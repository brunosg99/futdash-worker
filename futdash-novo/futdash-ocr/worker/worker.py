# worker/worker.py
import os
import json
import time
import ssl
import logging
import tempfile
from pathlib import Path

import cv2
import redis
import requests

from core.core_logic import HudProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("worker")

# =========================
# 0) Configurações básicas
# =========================
QUEUE_PLUS   = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE   = os.getenv("QUEUE_FREE", "qv2:default")
STORE_DIR    = Path(os.getenv("STORE_DIR", "/workspace/data"))
RESULTS_DIR  = STORE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_EXIT_MIN = int(os.getenv("IDLE_EXIT_MIN", "15"))
IDLE_EXIT_SEC = max(30, IDLE_EXIT_MIN * 60)  # segurança mínima 30s

PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "https://futdash-api-v2.onrender.com")

# Para auto-stop do Pod (parar cobrança de GPU)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")      # ex: "rp_xxx"
RUNPOD_POD_ID  = os.getenv("RUNPOD_POD_ID")       # ex: "xxxxxxxxxxxxxxxx"
RUNPOD_API     = "https://api.runpod.io/v2/pods"


# ===================================
# 1) Conexão Redis (Upstash) c/ TLS
# ===================================
def connect_redis():
    """
    Cria conexão Redis (Upstash) usando TLS. Upstash costuma exigir rediss://
    e, em ambientes serverless, a verificação de certificado pode falhar em
    conexões longas — por isso ssl_cert_reqs=ssl.CERT_NONE.
    """
    url  = (os.getenv("REDIS_URL") or "").strip()
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    user = os.getenv("REDIS_USER")
    pw   = os.getenv("REDIS_PASS")

    # 1) URL direta (recomendado)
    if url:
        if url.startswith("redis://"):
            url = "rediss://" + url.split("://", 1)[1]
        return redis.Redis.from_url(url, ssl_cert_reqs=ssl.CERT_NONE, decode_responses=True)

    # 2) Montagem manual
    if host and user and pw:
        built = f"rediss://{user}:{pw}@{host}:{port}"
        return redis.Redis.from_url(built, ssl_cert_reqs=ssl.CERT_NONE, decode_responses=True)

    raise RuntimeError("Credenciais Redis ausentes (REDIS_URL ou HOST/PORT/USER/PASS).")


def redis_ping_safe(r: redis.Redis) -> bool:
    try:
        return bool(r.ping())
    except Exception as e:
        log.warning(f"[redis] ping falhou: {e}")
        return False


# Conexão global
r = connect_redis()
if not redis_ping_safe(r):
    log.warning("[redis] ping inicial falhou; seguiremos e o loop tentará novamente quando necessário.")


# ==========================================
# 2) Utilitários: baixar e ler a imagem
# ==========================================
def download_to_temp(url: str, max_attempts: int = 3, timeout: int = 60) -> str:
    """
    Baixa a imagem de download_url para um arquivo temporário. Faz alguns retries leves.
    Retorna o caminho do arquivo temporário.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                tmp.write(resp.content)
                return tmp.name
        except Exception as e:
            last_err = e
            log.warning(f"[download] tentativa {attempt}/{max_attempts} falhou ({e}); retry em 1s...")
            time.sleep(1)
    raise last_err if last_err else RuntimeError("Falha desconhecida no download.")


def read_image_from_job(job: dict):
    """
    Lê a imagem via download_url (preferencial) ou caminho local (apenas para modo local).
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

    if image_path:
        return cv2.imread(image_path)

    return None


# ==========================================
# 3) Auto-stop: parar o Pod no RunPod ($$$)
# ==========================================
def stop_runpod():
    """
    Chama a API do RunPod para parar o Pod atual (status Stopped).
    Isso zera a cobrança da GPU enquanto parado.
    """
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        log.info("[auto-stop] RUNPOD_API_KEY/RUNPOD_POD_ID ausentes; não é possível parar o Pod via API.")
        return False

    url = f"{RUNPOD_API}/stop"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"podId": RUNPOD_POD_ID}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        ok = 200 <= resp.status_code < 300
        log.info(f"[auto-stop] POST {url} => {resp.status_code} {resp.text[:200]}")
        return ok
    except Exception as e:
        log.error(f"[auto-stop] erro ao chamar RunPod stop: {e}")
        return False


def try_auto_stop_if_idle(idle_secs: float) -> bool:
    """
    Se passou do limite de idle e as filas continuam vazias, tenta parar o Pod.
    Retorna True se pediu o stop (independente do sucesso no HTTP).
    """
    if idle_secs < IDLE_EXIT_SEC:
        return False

    try:
        total_llen = (r.llen(QUEUE_PLUS) or 0) + (r.llen(QUEUE_FREE) or 0)
    except Exception as e:
        log.warning(f"[auto-stop] não consegui checar LLEN: {e}")
        total_llen = 0  # se não deu pra checar, vamos considerar zero e prosseguir com cautela

    if total_llen > 0:
        # Entrou job enquanto estava ocioso — não para
        return False

    # Parar Pod via API RunPod
    did_request = stop_runpod()
    if did_request:
        log.info("[auto-stop] Requisição de stop enviada; encerrando processo do worker.")
    else:
        log.info("[auto-stop] Falhou em requisitar stop; encerrando processo para evitar cobrança.")
    return True


# ==========================================
# 4) Processamento de um job
# ==========================================
def process_job(payload: str, processor: HudProcessor):
    job = json.loads(payload)
    job_id = job.get("job_id")

    try:
        log.info(f"[worker] Processando job {job_id} ...")
        r.hset(f"job:{job_id}", mapping={"status": "processing"})

        frame = read_image_from_job(job)
        if frame is None:
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
                "error": f"HTTP {http_err.response.status_code} ao baixar {job.get('download_url')}"
            },
        )
        log.error(f"[worker] HTTPError no job {job_id}: {http_err}")
    except Exception as e:
        r.hset(f"job:{job_id}", mapping={"status": "failed", "error": str(e)})
        log.exception(f"[worker] Erro no job {job_id}")


# ==========================================
# 5) Loop principal do worker
# ==========================================
def main():
    log.info("HudProcessor: inicializando...")
    processor = HudProcessor(ckpt_path="weights/best.pt")
    log.info("[worker] device ativo: cpu/gpu conforme core")

    log.info(f"[worker] Iniciando... filas: {QUEUE_PLUS}, {QUEUE_FREE}")
    last_job_ts = time.time()

    while True:
        try:
            payload = r.lpop(QUEUE_PLUS) or r.lpop(QUEUE_FREE)
        except Exception as e:
            log.warning(f"[redis] LPOP falhou ({e}); aguardando 1s e tentando de novo.")
            time.sleep(1)
            continue

        if payload:
            last_job_ts = time.time()
            process_job(payload, processor)
            continue

        # Fila vazia no momento
        time.sleep(1)
        idle_secs = time.time() - last_job_ts

        # Ao bater limite de ociosidade, para o Pod (RunPod API) e sai
        if try_auto_stop_if_idle(idle_secs):
            break

    log.info("[worker] encerrado.")

if __name__ == "__main__":
    main()
