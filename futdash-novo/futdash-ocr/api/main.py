# main.py — versão final robusta (Render + Upstash)
import os
import uuid
import time
import ssl
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import redis
import json

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ========= Config =========
QUEUE_PLUS = os.getenv("QUEUE_PLUS", "qv2:high")
QUEUE_FREE = os.getenv("QUEUE_FREE", "qv2:default")

STORE_DIR = Path(os.getenv("STORE_DIR", "/data"))
UPLOADS_DIR = STORE_DIR / "uploads"
RESULTS_DIR = STORE_DIR / "results"
for d in (UPLOADS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# ========= Redis (compatível com Upstash TLS ou local) =========
def connect_redis():
    """Tenta conectar usando REDIS_URL ou host/port/user/pass."""
    url = os.getenv("REDIS_URL", "").strip()
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT", "6379"))
    user = os.getenv("REDIS_USER")
    password = os.getenv("REDIS_PASS")

    try:
        if url and (url.startswith("redis://") or url.startswith("rediss://")):
            extra = {"decode_responses": True}
            if url.startswith("rediss://"):
                extra["ssl_cert_reqs"] = ssl.CERT_NONE
            logging.info(f"[redis] Conectando via URL: {url.split('@')[-1]}")
            return redis.Redis.from_url(url, **extra)

        elif host and user and password:
            logging.info(f"[redis] Conectando via HOST: {host}")
            return redis.Redis(
                host=host,
                port=port,
                username=user,
                password=password,
                ssl=True,
                ssl_cert_reqs=ssl.CERT_NONE,
                decode_responses=True,
            )
        else:
            logging.warning("[redis] Nenhuma variável válida encontrada (REDIS_URL ou HOST/USER/PASS).")
            return None
    except Exception as e:
        logging.error(f"[redis] Falha ao conectar: {e}")
        return None


r = connect_redis()
app = FastAPI(title="Futdash OCR API v2")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/diag/redis")
def diag_redis():
    """
    Diagnóstico explícito do Redis: tenta PING e SET/GET.
    Retorna erro detalhado se falhar (sem derrubar o serviço).
    """
    if r is None:
        return JSONResponse(status_code=500, content={"redis": "error", "detail": "Redis não configurado"})
    try:
        r.ping()
        k = f"diag:{int(time.time())}"
        r.set(k, "ok", ex=30)
        val = r.get(k)
        return {"redis": "ok", "set_get": val}
    except Exception as e:
        logging.exception("Diag Redis falhou")
        return JSONResponse(status_code=500, content={"redis": "error", "detail": str(e)})


@app.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    plan: str = Form(...),
    user_id: str = Form(...)
):
    """Salva a imagem e cria o job no Redis."""
    if r is None:
        raise HTTPException(status_code=503, detail="Redis não configurado.")

    try:
        filename = image.filename or ""
        ext = (filename.rsplit(".", 1)[-1] if "." in filename else "png").lower()
        if ext not in {"png", "jpg", "jpeg"}:
            ext = "png"

        job_id = str(uuid.uuid4())
        save_path = UPLOADS_DIR / f"{job_id}.{ext}"
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")
        save_path.write_bytes(data)

        queue = QUEUE_PLUS if plan.lower() in {"plus", "pro", "paid"} else QUEUE_FREE
        download_url = f"{PUBLIC_BASE_URL}/files/{job_id}" if PUBLIC_BASE_URL else ""

        payload = {
            "job_id": job_id,
            "image_path": str(save_path),
            "download_url": download_url,
            "plan": plan,
            "user_id": user_id,
            "created_at": int(time.time()),
        }

        try:
            r.ping()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Redis indisponível: {e}")

        r.hset(f"job:{job_id}", mapping={"status": "queued"})
        r.rpush(queue, json.dumps(payload))
        logging.info(f"[api] Job {job_id} enfileirado em {queue}")

        return {"job_id": job_id, "status": "queued"}

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Erro no /upload")
        return JSONResponse(status_code=500, content={"error": "upload_failed", "detail": str(e)})


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if r is None:
        raise HTTPException(503, "Redis não configurado.")
    st = r.hgetall(f"job:{job_id}")
    if not st:
        raise HTTPException(404, "Job não encontrado.")
    return {"job_id": job_id, **st}


@app.get("/files/{job_id}")
def get_file(job_id: str):
    """Expõe a imagem salva."""
    for ext in ("png", "jpg", "jpeg"):
        p = UPLOADS_DIR / f"{job_id}.{ext}"
        if p.exists():
            return FileResponse(p)
    raise HTTPException(404, "Arquivo não encontrado.")

