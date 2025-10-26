import os, json, time, logging, ssl, traceback, tempfile
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import redis
import requests

from core.core_logic import HudProcessor  # seu core

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

# =========================
# Helpers de tempo/ISO
# =========================
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_iso(s: str) -> datetime:
    # aceita "Z" ou ±hh:mm
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _diff_ms(a_iso: str, b_iso: str) -> int:
    return int(( _parse_iso(b_iso) - _parse_iso(a_iso) ).total_seconds() * 1000)

# =========================
# Redis
# =========================
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
            raise RuntimeError("Credenciais Redis ausentes (REDIS_URL ou HOST/PORT/USER/PASS).")
    except Exception as e:
        logging.error(f"[worker] Falha ao conectar Redis: {e}")
        raise

r = connect_redis()

QUEUE_PLUS  = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE  = os.getenv("QUEUE_FREE", "qv2:default")
STORE_DIR   = Path(os.getenv("STORE_DIR", "./data"))
RESULTS_DIR = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "30"))
IDLE_EXIT_SEC = IDLE_EXIT_MIN * 60

# =========================
# Modelo
# =========================
logging.info("HudProcessor: inicializando...")
processor = HudProcessor(ckpt_path="weights/best.pt")
try:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"HudProcessor: device={dev}, fp16={'True' if (dev=='cuda') else 'False'}")
except Exception:
    logging.info("HudProcessor: device=desconhecido")

# =========================
# I/O de imagem
# =========================
def download_to_temp(url: str) -> str:
    """Baixa a imagem via HTTP e retorna o caminho temporário local."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp.write(resp.content)
        return tmp.name

# =========================
# Job
# =========================
def process_job(payload: str):
    job = json.loads(payload)
    job_id       = job.get("job_id")
    image_path   = job.get("image_path")    # caminho físico (só existirá se worker e API compartilham storage)
    download_url = job.get("download_url")  # URL http da API (use este no RunPod)

    try:
        picked_at = utcnow_iso()
        # marca "processing" e picked_at
        r.hset(f"job:{job_id}", mapping={"status": "processing", "picked_at": picked_at})
        logging.info(f"[worker] Processando job {job_id} ...")

        # baixa via HTTP se houver URL; senão tenta caminho local
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

        done_at = utcnow_iso()

        # métricas (se a API tiver gravado enqueued_at)
        data = {"status": "done", "done_at": done_at}
        try:
            enq = r.hget(f"job:{job_id}", "enqueued_at")
            if enq:
                data["queue_ms"] = str(_diff_ms(enq, picked_at))
                data["run_ms"]   = str(_diff_ms(picked_at, done_at))
                data["total_ms"] = str(_diff_ms(enq, done_at))
        except Exception as _:
            pass

        r.hset(f"job:{job_id}", mapping=data)
        logging.info(f"[worker] ✅ Job {job_id} concluído.")
    except Exception as e:
        logging.error(f"[worker] Erro no job {job_id}: {e}")
        traceback.print_exc()
        r.hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})

# =========================
# Loop principal (BLPOP + auto-shutdown)
# =========================
def main():
    logging.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    logging.info(f"[worker] IDLE_EXIT_MIN={IDLE_EXIT_MIN}")

    idle_limit = IDLE_EXIT_SEC
    last_activity = time.monotonic()

    # BLPOP: bloqueia até 10s esperando item; checa idle e repete
    while True:
        try:
            # prioridade: PLUS primeiro
            res = r.blpop([QUEUE_PLUS, QUEUE_FREE], timeout=10)
        except Exception as e:
            logging.error(f"[worker] Erro no BLPOP: {e}")
            time.sleep(2)
            res = None

        if res:
            _queue, payload = res  # _queue = nome da fila que retornou
            last_activity = time.monotonic()
            process_job(payload)
            continue

        # timeout do BLPOP sem job; verificar ociosidade
        idle_for = time.monotonic() - last_activity
        if idle_for > idle_limit:
            logging.info(f"[worker] Nenhum job há {idle_for/60:.1f} min → encerrando pod.")
            os._exit(0)  # encerra o processo/container → RunPod para o Pod

if __name__ == "__main__":
    main()
