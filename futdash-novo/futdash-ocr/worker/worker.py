# worker/worker.py
import os
import json
import time
import ssl
import logging
import traceback
import tempfile
from pathlib import Path

import cv2
import numpy as np
import redis
import requests

from core.core_logic import HudProcessor  # sua lógica principal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("worker")


# -------------------------------------------------------------------
# 1) Allowlist para deserialização segura (PyTorch 2.6+ / weights_only)
# -------------------------------------------------------------------
def _allow_ultralytics_unpickling():
    """
    Registra classes do Ultralytics/PyTorch necessárias para deserializar checkpoints
    quando torch.load usa weights_only=True (comportamento padrão no PyTorch 2.6+).
    Cobri um conjunto amplo para YOLOv8.
    """
    import torch
    import torch.nn as nn
    from torch.serialization import add_safe_globals

    # Ultralytics
    import ultralytics
    from ultralytics.nn import modules as um, tasks as ut

    add_safe_globals([
        # núcleo YOLO (Ultralytics)
        ut.DetectionModel,

        # containers PyTorch (caminhos curtos e completos)
        nn.Sequential,
        nn.modules.container.Sequential,
        nn.ModuleList,
        nn.modules.container.ModuleList,
        nn.ModuleDict,
        nn.modules.container.ModuleDict,

        # blocos/camadas YOLOv8 mais comuns
        um.conv.Conv,
        um.conv.Concat,          # <---- IMPORTANTE (erro atual)
        um.block.C2f,
        um.block.Bottleneck,
        um.block.SPPF,
        um.block.C3 if hasattr(um.block, "C3") else nn.Identity,
        um.block.C3x if hasattr(um.block, "C3x") else nn.Identity,
        um.head.Detect,

        # camadas PyTorch comuns
        nn.Conv2d, nn.BatchNorm2d, nn.SyncBatchNorm,
        nn.SiLU, nn.ReLU, nn.LeakyReLU,
        nn.Upsample, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Dropout, nn.Identity,
    ])


# ---------------------------------------------------------
# 2) Redis: aceita REDIS_URL OU HOST/PORT/USER/PASS (Upstash)
# ---------------------------------------------------------
def connect_redis():
    url = (os.getenv("REDIS_URL") or "").strip()
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT", "6379"))
    user = os.getenv("REDIS_USER")
    pw   = os.getenv("REDIS_PASS")

    try:
        if url and (url.startswith("redis://") or url.startswith("rediss://")):
            extra = {"decode_responses": True}
            if url.startswith("rediss://"):
                # Upstash com TLS
                extra["ssl_cert_reqs"] = ssl.CERT_NONE
            # Loga só o host para não expor credenciais
            log.info(f"[worker] Redis via URL: {url.split('@')[-1]}")
            return redis.Redis.from_url(url, **extra)

        if host and user and pw:
            log.info(f"[worker] Redis via HOST: {host}")
            return redis.Redis(
                host=host, port=port, username=user, password=pw,
                ssl=True, ssl_cert_reqs=ssl.CERT_NONE, decode_responses=True
            )

        raise RuntimeError("Credenciais Redis ausentes (defina REDIS_URL OU HOST/PORT/USER/PASS).")

    except Exception as e:
        log.error(f"[worker] Falha ao conectar Redis: {e}")
        raise


# -----------------------------
# 3) Configurações de execução
# -----------------------------
r = connect_redis()

QUEUE_PLUS  = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE  = os.getenv("QUEUE_FREE", "qv2:default")
STORE_DIR   = Path(os.getenv("STORE_DIR", "/workspace/data"))
RESULTS_DIR = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "15"))
IDLE_EXIT_SEC = IDLE_EXIT_MIN * 60


# ------------------------------------------------
# 4) Utilitários de I/O: download e leitura imagem
# ------------------------------------------------
def download_to_temp(url: str) -> str:
    """Baixa a imagem via HTTP e retorna caminho de arquivo temporário local."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp.write(resp.content)
        return tmp.name


def read_image_from_job(job: dict):
    """
    Lê a imagem do job. Se houver 'download_url' (URL pública da API), baixa.
    Senão tenta 'image_path' (caminho local no pod — normalmente não será válido).
    """
    download_url = job.get("download_url")
    image_path   = job.get("image_path")

    if download_url:
        tmp_path = download_to_temp(download_url)
        try:
            img = cv2.imread(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return img

    # fallback: tentar caminho local
    if image_path:
        return cv2.imread(image_path)

    return None


# ----------------------------------------
# 5) Processamento do job (YOLO + seu core)
# ----------------------------------------
def process_job(payload: str, processor: HudProcessor):
    job = json.loads(payload)
    job_id = job.get("job_id")

    try:
        log.info(f"[worker] Processando job {job_id} ...")
        r.hset(f"job:{job_id}", mapping={"status": "processing"})

        frame = read_image_from_job(job)
        if frame is None:
            raise RuntimeError("Falha ao ler imagem (nem URL nem path local válidos).")

        results = processor.process_frame(frame)

        out_path = RESULTS_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        r.hset(f"job:{job_id}", mapping={"status": "done"})
        log.info(f"[worker] ✅ Job {job_id} concluído.")

    except Exception as e:
        log.error(f"[worker] Erro no job {job_id}: {e}")
        traceback.print_exc()
        r.hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})


# ----------------------
# 6) Loop principal
# ----------------------
def main():
    # habilita allowlist do pickle/torch antes de carregar o YOLO
    _allow_ultralytics_unpickling()

    log.info("HudProcessor: inicializando...")
    processor = HudProcessor(ckpt_path="weights/best.pt")
    # o HudProcessor decide cpu/gpu; pode logar internamente também
    log.info("[worker] device ativo: cpu/gpu conforme core")

    log.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    last_job_ts = time.time()

    while True:
        # tenta primeiro a fila “rápida” e depois a padrão
        payload = r.lpop(QUEUE_PLUS)
        if payload is None:
            payload = r.lpop(QUEUE_FREE)

        if payload:
            last_job_ts = time.time()
            process_job(payload, processor)
        else:
            time.sleep(1)
            if time.time() - last_job_ts > IDLE_EXIT_SEC:
                log.info(f"[worker] idle por {IDLE_EXIT_MIN} min. Encerrando.")
                break


if __name__ == "__main__":
    main()
