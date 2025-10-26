# worker/worker.py
import os, json, time, ssl, logging, traceback, tempfile
from pathlib import Path

import cv2, numpy as np, redis, requests

# >>> Monkey patch: força torch.load(weights_only=False) globalmente <<<
import torch
_ORIG_TORCH_LOAD = torch.load
def _torch_load_no_weights_only(*args, **kwargs):
    # Desativa o modo seguro por pesos. Útil quando o checkpoint é confiável (seu .pt).
    kwargs["weights_only"] = False
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_no_weights_only  # APLICA O PATCH ANTES DE IMPORTAR/CRIAR O MODELO

from core.core_logic import HudProcessor  # sua lógica principal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("worker")

# ---------------------------------------------------------
# Redis: aceita REDIS_URL OU HOST/PORT/USER/PASS (Upstash)
# ---------------------------------------------------------
def connect_redis():
    url  = (os.getenv("REDIS_URL") or "").strip()
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT", "6379"))
    user = os.getenv("REDIS_USER")
    pw   = os.getenv("REDIS_PASS")

    try:
        if url and (url.startswith("redis://") or url.startswith("rediss://")):
            extra = {"decode_responses": True}
            if url.startswith("rediss://"):
                extra["ssl_cert_reqs"] = ssl.CERT_NONE  # Upstash TLS
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
# Configurações de execução
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
# Utilitários de I/O
# ------------------------------------------------
def download_to_temp(url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp.write(resp.content)
        return tmp.name

def read_image_from_job(job: dict):
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

# ----------------------------------------
# Processamento do job
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
# Loop principal
# ----------------------
def main():
    log.info("HudProcessor: inicializando...")
    # Com o monkey patch ativo, o YOLO carrega sem weights_only
    processor = HudProcessor(ckpt_path="weights/best.pt")
    log.info("[worker] device ativo: cpu/gpu conforme core")

    log.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    last_job_ts = time.time()

    while True:
        payload = r.lpop(QUEUE_PLUS) or r.lpop(QUEUE_FREE)
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
