FROM python:3.11-slim

# deps mínimos p/ OpenCV ler imagens se um dia precisar
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

# diretórios para uploads e resultados
RUN mkdir -p /data/uploads /data/results

ENV PORT=10000
# start leve; Render vai setar $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
