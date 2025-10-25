# enqueue_local.py — enfileira um job direto no Redis p/ o worker ler um arquivo local

import os, json, time, ssl, redis, uuid, pathlib

# ==== EDITE AQUI ====
TOKEN = "AXKAAAIncDJkNGVjNjQ0MzNmMmM0NDNjOGQ3MzlmMDAyMGJhYjNlNnAyMjkzMTI"  # cole aqui o token do Upstash (a parte entre 'default:' e '@')
IMG = r"C:\Users\bruno\Desktop\futdash-novo\futdash-ocr\teste3.jpeg"  # caminho do seu arquivo local
# =====================

HOST = "wired-wolf-29312.upstash.io"
PORT = 6379
USER = "default"

# validações rápidas
if not pathlib.Path(IMG).exists():
    raise SystemExit(f"Arquivo não encontrado: {IMG}")

# conecta no Redis (TLS)
r = redis.Redis(host=HOST, port=PORT, username=USER, password=TOKEN,
                ssl=True, ssl_cert_reqs=ssl.CERT_NONE, decode_responses=True)

# monta o job
job_id = str(uuid.uuid4())
payload = {
    "job_id": job_id,
    "image_path": IMG,      # o worker vai abrir este arquivo local
    "download_url": "",     # sem HTTP
    "plan": "free",
    "user_id": "debug",
    "created_at": int(time.time()),
}

# grava status e envia para a fila default
r.hset(f"job:{job_id}", mapping={"status": "queued"})
r.rpush("qv2:default", json.dumps(payload))

print("ENFILEIRADO:", job_id)
print("Pronto! Agora olhe a janela do worker: ele deve processar e salvar o JSON em data\\results\\<JOB_ID>.json")
