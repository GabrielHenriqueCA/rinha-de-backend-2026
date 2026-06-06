# linux/amd64 obrigatório
FROM --platform=linux/amd64 python:3.12-slim AS build

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY resources/ ./resources/
COPY normalization.json mcc_risk.json ./
COPY preprocess.py app.py ./

# Constrói índice FAISS IVF fp16 no build — startup é só mmap.
RUN python preprocess.py

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2 PYTHONDONTWRITEBYTECODE=1
CMD ["sh", "-c", "granian --interface asgi --uds ${API_SOCKET:-/tmp/api.sock} --uds-permissions 666 --workers 1 --runtime-threads 1 --loop uvloop --backlog 4096 app:app"]
