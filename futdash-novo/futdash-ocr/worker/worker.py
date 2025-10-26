# worker/worker.py
import os
import json
import time
import ssl
import logging
import traceback
import tempfile
import random
from pathlib import Path

import cv2
import redis
import requests
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from core.core_logic import HudProcessor  # sua lógica principal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("worker")

# =========================
# 0) Allowlist para YOLO/PT
# =========================
def _allow_ultralytics_unpickling():
    """
    Registra classes do Ultralytics/PyTorch necessárias para deserializar checkpoints
    quando torch.load usa weights_only=True (comportamento padrão no PyTorch 2.6+).
    """
    import torch
    import torch.nn as nn
    import ultralytics
    from ultralytics.nn import modules as um, tasks as ut

    torch.serialization.add_safe_globals([
        # núcleo YOLO
        ut.DetectionModel,
        # containers
        nn.Sequential, nn.modules.container.Sequential, nn.ModuleList,
        # blocos/camadas YOLOv8
        um.conv.Conv,
        um.conv.Concat,
        um.block.C2f, um.block.Bottleneck, um.block.SPPF, um.block.DFL,
        um.head.Detect,
        # camadas PyTorch comuns
        nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.ReLU, nn.LeakyReLU,
        nn.Upsample, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Dropout,
    ])


# =======================================
# 1) Redis com keepalive + reconexão
# =======================================
def connect_redis():
    url  = (os.getenv("REDIS_URL") or "").strip()
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    user = os.getenv("REDIS_USER")
    pw   = os.getenv("REDIS_PASS")

    # Se veio URL, padroniza para rediss:// (TLS) e usa somente from_url
    if url:
        if url.startswith("redis://"):
            url = "rediss://" + url.split("://", 1)[1]
        return redis.Redis.from_url(url, decode_responses=True)

    # Se veio em partes, monta um rediss:// e usa from_url
    if host and user and pw:
        built = f"rediss://{user}:{pw}@{host}:{port}"
        return redis.Redis.from_url(built, decode_responses=True)

    raise RuntimeError("Credenciais Redis ausentes (REDIS_URL ou HOST/PORT/USER/PASS).")


    except Exception as e:
        log.error(f"[worker] Falha ao conectar Redis: {e}")
        raise

# conexão inicial
r = connect_redis()

def ensure_redis():
    """Garante uma conexão viva; reconecta se necessário."""
    global r
    try:
        r.ping()
        return r
    except Exception:
        time.sleep(0.5 + random.random())
        r = connect_redis()
        return r


# ============================
# 2) Configurações e diretórios
# ============================
QUEUE_PLUS  = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE  = os.getenv("QUEUE_FREE", "qv2:default")

STORE_DIR   = Path(os.getenv("STORE_DIR", "/workspace/data"))
RESULTS_DIR = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "15"))
IDLE_EXIT_SEC = IDLE_EXIT_MIN * 60


# ==================================
# 3) Utilitários de download/leitura
# ==================================
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
            try: os.unlink(tmp_path)
            except Exception: pass
        return img

    if image_path:
        return cv2.imread(image_path)

    return None


# ===========================
# 4) Processamento do Job
# ===========================
def process_job(payload: str, processor: HudProcessor):
    job = json.loads(payload)
    job_id = job.get("job_id")

    try:
        log.info(f"[worker] Processando job {job_id} ...")
        rr = ensure_redis()
        rr.hset(f"job:{job_id}", mapping={"status": "processing"})

        frame = read_image_from_job(job)
        if frame is None:
            # mais contexto no erro ajuda a debugar 404/paths
            raise RuntimeError(
                f"Falha ao ler imagem (url={job.get('download_url')}, path={job.get('image_path')})"
            )

        results = processor.process_frame(frame)

        out_path = RESULTS_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        rr.hset(f"job:{job_id}", mapping={"status": "done"})
        log.info(f"[worker] ✅ Job {job_id} concluído.")

    except requests.HTTPError as http_err:
        # pega, por exemplo, 404 ao baixar a imagem
        msg = f"HTTP {http_err.response.status_code} ao baixar {job.get('download_url')}"
        log.error(f"[worker] Erro no job {job_id}: {msg}")
        traceback.print_exc()
        ensure_redis().hset(f"job:{job_id}", mapping={"status": "failed", "error": msg})

    except Exception as e:
        log.error(f"[worker] Erro no job {job_id}: {e}")
        traceback.print_exc()
        ensure_redis().hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})


# ===========================
# 5) Loop principal
# ===========================
def main():
    _allow_ultralytics_unpickling()

    log.info("HudProcessor: inicializando...")
    processor = HudProcessor(ckpt_path="weights/best.pt")
    log.info("[worker] device ativo: cpu/gpu conforme core")

    log.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    last_job_ts = time.time()

    BRPOP_TIMEOUT = 10  # seg. (usa bloqueio em vez de polling agressivo)

    while True:
        try:
            rr = ensure_redis()

            item = rr.brpop(QUEUE_PLUS, timeout=BRPOP_TIMEOUT)
            if item is None:
                item = rr.brpop(QUEUE_FREE, timeout=BRPOP_TIMEOUT)

            if item:
                _queue, payload = item
                last_job_ts = time.time()
                process_job(payload, processor)
            # se item None, foi apenas timeout — segue para checar ociosidade

        except (ConnectionError, TimeoutError, RedisError) as e:
            log.warning(f"[worker] Redis caiu: {e}. Recolando...")
            time.sleep(0.5)  # pequeno backoff e reconecta no próximo ciclo

        # encerrar por ociosidade
        if time.time() - last_job_ts > IDLE_EXIT_SEC:
            log.info(f"[worker] idle por {IDLE_EXIT_MIN} min. Encerrando.")
            break


if __name__ == "__main__":
    main()


