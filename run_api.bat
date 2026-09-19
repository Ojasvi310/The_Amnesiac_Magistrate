@echo off
set CC_GGUF_PATH=exports\regime_q1\model.gguf
set CC_ADAPTER_DIR=exports\regime_q1\adapter
python -m uvicorn src.online.api:app --host 0.0.0.0 --port 8000
