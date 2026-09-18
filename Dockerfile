# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies separately to cache the layer
COPY requirements-online.txt .
RUN pip install --no-cache-dir -r requirements-online.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and config
COPY src/online/ src/online/
COPY src/audit/ src/audit/
COPY src/config_loader.py src/
COPY src/training_config.py src/
COPY configs/ configs/
COPY dashboard/ dashboard/

# Environment defaults - artifacts mounted as volumes
ENV CC_GGUF_PATH=/data/model.gguf
ENV CC_ADAPTER_DIR=/data/adapter/
ENV CC_INDEX_DIR=/data/faiss_index/
ENV CC_DB_PATH=/data/continual_counsel.db

# Explicitly disable CUDA/GPU
ENV CUDA_VISIBLE_DEVICES=""
ENV GGMl_USE_METAL="0"

EXPOSE 8000

CMD ["uvicorn", "src.online.api:app", "--host", "0.0.0.0", "--port", "8000"]
