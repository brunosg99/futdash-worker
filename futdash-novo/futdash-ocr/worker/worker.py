# worker/worker.py
import os
import json
import time
import logging
import tempfile
from pathlib import Path

import cv2
import redis
import requests

from core.core_logic import HudProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("worker")


# ---------------------------
# 1) Conexão Redis (Upstash)
# ---------------------------
def connect_redis():
    url  = (os.getenv("REDIS_URL") or "").strip()
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    user = os.getenv("REDIS_USER")
    pw   = os.getenv("REDIS_PASS")

    if url:
        # força TLS
        if url.startswith("redis://"):
            url = "rediss://" + url.split("://", 1)[1]
        return redis.Redis.from_url(url, decode_responses=True)

    if host and user and pw:
        built = f"rediss://{user}:{pw}@{host}:{port}"
        return redis.Redis.from_url(built, decode_responses=True)

    raise RuntimeError("Credenciais Redis ausentes (REDIS_URL ou HOST/PORT/USER/PASS).")


r = connect_redis()

QUEUE_PLUS  = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE  = os.getenv("QUEUE_FREE", "qv2:default")
STORE_DIR   = Path(os.getenv("STORE_DIR", "/workspace/data"))
RESULTS_DIR = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "15"))
IDLE_EXIT_SEC = IDLE_EXIT_MIN * 60


# --------------------------------------
# 2) Utilitários: baixar e ler a imagem
# --------------------------------------
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
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return img

    if image_path:
        return cv2.imread(image_path)

    return None


# ----------------------------
# 3) Processamento de um job
# ----------------------------
def process_job(payload: str, processor: HudProcessor):
    job = json.loads(payload)
    job_id = job.get("job_id")

    try:
        log.info(f"[worker] Processando job {job_id} ...")
        r.hset(f"job:{job_id}", mapping={"status": "processing"})

        frame = read_image_from_job(job)
        if frame is None:
            # marca erro amigável no Redis
            r.hset(
                f"job:{job_id}",
                mapping={
                    "status": "failed",
                    "error": f"Imagem indisponível. Tente acessar {job.get('download_url','(sem url)')} no navegador."
                },
            )
            log.error(f"[worker] Falha ao ler imagem (url={job.get('download_url')} path={job.get('image_path')}).")
            return

        results = processor.process_frame(frame)

        out_path = RESULTS_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        r.hset(f"job:{job_id}", mapping={"status": "done"})
        log.info(f"[worker] ✅ Job {job_id} concluído.")

    except requests.HTTPError as http_err:
        r.hset(
            f"job:{job_id}",
            mapping={
                "status": "failed",
                "error": f"HTTP {http_err.response.status_code} ao baixar {job.get('download_url')}",
            },
        )
        log.error(f"[worker] HTTPError no job {job_id}: {http_err}")
    except Exception as e:
        r.hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})
        log.exception(f"[worker] Erro no job {job_id}")


# ----------------------------
# 4) Loop principal do worker
# ----------------------------
def main():
    log.info("HudProcessor: inicializando...")
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
