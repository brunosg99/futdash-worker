import os, json, time, logging, ssl, traceback, tempfile
from pathlib import Path

import cv2
import numpy as np
import redis
import requests

from core.core_logic import HudProcessor  # seu core

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

# ===== Redis: aceita REDIS_URL ou HOST/PORT/USER/PASS =====
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
                extra["ssl_cert_reqs"] = ssl.CERT_NONE
            logging.info(f"[worker] Redis via URL: {url.split('@')[-1]}")
            return redis.Redis.from_url(url, **extra)
        elif host and user and pw:
            logging.info(f"[worker] Redis via HOST: {host}")
            return redis.Redis(
                host=host, port=port, username=user, password=pw,
                ssl=True, ssl_cert_reqs=ssl.CERT_NONE, decode_responses=True
            )
        else:
            raise RuntimeError("Credenciais Redis ausentes.")
    except Exception as e:
        logging.error(f"[worker] Falha ao conectar Redis: {e}")
        raise

r = connect_redis()

QUEUE_PLUS  = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE  = os.getenv("QUEUE_FREE", "qv2:default")
STORE_DIR   = Path(os.getenv("STORE_DIR", "./data"))
RESULTS_DIR = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "15"))
IDLE_EXIT_SEC = IDLE_EXIT_MIN * 60

# modelo (CPU mesmo, como você rodou)
processor = HudProcessor(ckpt_path="weights/best.pt")

def download_to_temp(url: str) -> str:
    """Baixa a imagem via HTTP e retorna o caminho temporário local."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp.write(resp.content)
        return tmp.name

def process_job(payload: str):
    job = json.loads(payload)
    job_id       = job.get("job_id")
    image_path   = job.get("image_path")    # caminho no servidor da API (não existe localmente)
    download_url = job.get("download_url")  # URL http da API (é o que usaremos)

    try:
        logging.info(f"[worker] Processando job {job_id} ...")
        r.hset(f"job:{job_id}", mapping={"status": "processing"})

        # baixa via HTTP se houver URL; caso contrário tenta caminho local
        if download_url:
            tmp_path = download_to_temp(download_url)
            frame = cv2.imread(tmp_path)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        else:
            frame = cv2.imread(image_path)

        if frame is None:
            raise RuntimeError(f"Falha ao ler imagem (url={download_url} path={image_path})")

        results = processor.process_frame(frame)
        out_path = RESULTS_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        r.hset(f"job:{job_id}", mapping={"status": "done"})
        logging.info(f"[worker] ✅ Job {job_id} concluído.")
    except Exception as e:
        logging.error(f"[worker] Erro no job {job_id}: {e}")
        traceback.print_exc()
        r.hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})

def main():
    logging.info(f"[worker] device ativo: cpu")
    logging.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    last_job_ts = time.time()

    while True:
        # tenta high, depois default
        payload = r.lpop(QUEUE_PLUS)
        if payload is None:
            payload = r.lpop(QUEUE_FREE)

        if payload:
            last_job_ts = time.time()
            process_job(payload)
        else:
            time.sleep(1)
            if time.time() - last_job_ts > IDLE_EXIT_SEC:
                logging.info(f"[worker] idle por {IDLE_EXIT_MIN} min. Encerrando.")
                break

if __name__ == "__main__":
    main()
