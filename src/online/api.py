"""
FastAPI application for Continual Counsel online inference.

Constraint compliance:
  - No HTTP client imports (no requests, httpx, aiohttp, urllib.request).
  - FastAPI listens for inbound connections; it never initiates outbound calls.
  - All heavy model/index objects are loaded once at startup via lifespan.

Configuration (in priority order):
  1. Environment variables: CC_GGUF_PATH, CC_ADAPTER_DIR, CC_INDEX_DIR,
     CC_DB_PATH, CC_EMBED_MODEL_ID, CC_CONFIG_PATH.
  2. YAML config file at CC_CONFIG_PATH (default: config/online.yaml).
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.audit.log import AuditLog
from src.audit.registry import AdapterRegistry
from src.online.confidence_gate import InsufficientGrounding, escalation_response
from src.online.infer import InferenceEngine, InferenceResult


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    regime_filter: Optional[str] = None


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    status: str
    retrieved_regime_tags: list[str]
    index_version: str
    adapter_version: str
    verified: bool
    flagged_claims: list[str]
    timestamp: str


class EscalationResponse(BaseModel):
    query_id: str
    status: str = "escalate"
    message: str


# ---------------------------------------------------------------------------
# App state (populated during lifespan startup)
# ---------------------------------------------------------------------------

class _AppState:
    engine: InferenceEngine
    audit_log: AuditLog
    adapter_registry: AdapterRegistry
    adapter_version: str
    index_version: str


_state = _AppState()


def _load_config() -> dict:
    """
    Merge YAML config file with environment variable overrides.
    ENV vars always win over the config file.
    """
    config_path = Path(os.environ.get("CC_CONFIG_PATH", "config/online.yaml"))
    cfg: dict = {}
    if config_path.exists():
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}

    overrides = {
        "gguf_path": os.environ.get("CC_GGUF_PATH"),
        "adapter_dir": os.environ.get("CC_ADAPTER_DIR"),
        "index_dir": os.environ.get("CC_INDEX_DIR"),
        "db_path": os.environ.get("CC_DB_PATH"),
        "embed_model_id": os.environ.get("CC_EMBED_MODEL_ID"),
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_config()

    cfg.setdefault("gguf_path", "exports/dummy_bundle/model.gguf")
    cfg.setdefault("adapter_dir", "exports/dummy_bundle/adapter")
    cfg.setdefault("index_dir", "data/faiss_index")
    cfg.setdefault("db_path", "continual_counsel.db")
    
    required = ["gguf_path", "adapter_dir", "index_dir"]

    db_path = cfg.get("db_path", "data/audit.db")
    embed_model_id = cfg.get("embed_model_id", "sentence-transformers/all-MiniLM-L6-v2")

    _state.engine = InferenceEngine(
        gguf_path=cfg["gguf_path"],
        adapter_dir=cfg["adapter_dir"],
        index_dir=cfg["index_dir"],
        embed_model_id=embed_model_id,
        base_cfg=cfg,
    )
    _state.audit_log = AuditLog(db_path=db_path)
    _state.adapter_registry = AdapterRegistry(db_path=db_path)
    _state.adapter_version = _state.engine._adapter_hash
    _state.index_version = _state.engine._index_hash

    yield
    # No explicit teardown needed; llama_cpp frees memory on GC.


app = FastAPI(title="Continual Counsel", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=None)
async def post_query(req: QueryRequest):
    """
    Answer a legal query using the RAG pipeline.

    Returns QueryResponse on success, or EscalationResponse when the retrieval
    confidence gate fires. Both are HTTP 200 -- escalation is a valid outcome.
    """
    query_id = str(uuid.uuid4())

    try:
        result: InferenceResult = _state.engine.answer(
            query=req.query,
            regime_filter=req.regime_filter,
        )
    except InsufficientGrounding as exc:
        esc = escalation_response(req.query)
        _state.audit_log.append(
            query_id=query_id,
            query_text=req.query,
            retrieved_chunk_ids=[],
            regime_tags=[],
            index_hash=_state.index_version,
            adapter_version_hash=_state.adapter_version,
            confidence_scores=[exc.max_score],
            verification_verdict="escalated",
            final_answer="",
        )
        return EscalationResponse(
            query_id=query_id,
            status="escalate",
            message=esc["message"],
        )

    regime_tags = list({c.regime for c in result.retrieved_chunks})
    chunk_ids = [
        f"{c.regime}::{c.section}::{i}"
        for i, c in enumerate(result.retrieved_chunks)
    ]
    verdict = "verified" if result.verification_result.verified else "flagged"

    _state.audit_log.append(
        query_id=query_id,
        query_text=req.query,
        retrieved_chunk_ids=chunk_ids,
        regime_tags=regime_tags,
        index_hash=result.index_hash,
        adapter_version_hash=result.adapter_version_hash,
        confidence_scores=result.retrieval_scores,
        verification_verdict=verdict,
        final_answer=result.answer,
    )

    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        status="ok",
        retrieved_regime_tags=regime_tags,
        index_version=result.index_hash,
        adapter_version=result.adapter_version_hash,
        verified=result.verification_result.verified,
        flagged_claims=result.verification_result.flagged_claims,
        timestamp=result.timestamp,
    )


@app.get("/audit/{query_id}")
async def get_audit(query_id: str):
    """Return the audit log entry for a given query_id."""
    record = _state.audit_log.get_record(query_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No audit record for {query_id!r}")

    return {
        "query_id": record.query_id,
        "query_text": record.query_text,
        "retrieved_chunk_ids": json.loads(record.retrieved_chunk_ids),
        "regime_tags": json.loads(record.regime_tags),
        "index_hash": record.index_hash,
        "adapter_version_hash": record.adapter_version_hash,
        "confidence_scores": json.loads(record.confidence_scores),
        "verification_verdict": record.verification_verdict,
        "final_answer": record.final_answer,
        "timestamp": record.timestamp,
        "record_hash": record.record_hash,
        "prev_record_hash": record.prev_record_hash,
    }


@app.get("/health")
async def health():
    """Liveness check; reports adapter and index version hashes."""
    return {
        "status": "ok",
        "adapter_version": _state.adapter_version,
        "index_version": _state.index_version,
        "model_loaded": _state.engine._llm is not None,
    }


@app.get("/adapter/lineage")
async def adapter_lineage():
    """
    Return the full merge lineage of the currently loaded adapter by walking
    the adapter registry from the current hash back through parent hashes.
    """
    lineage = _state.adapter_registry.get_lineage(_state.adapter_version)
    return {
        "current_adapter_hash": _state.adapter_version,
        "lineage": [
            {
                "adapter_version_hash": r.adapter_version_hash,
                "regime": r.regime,
                "timestamp": r.timestamp,
                "base_model_id": r.base_model_id,
                "merge_lineage": json.loads(r.merge_lineage_json or "[]"),
                "benchmark_scores": json.loads(r.benchmark_scores_json or "{}"),
                "repo_commit_hash": r.repo_commit_hash,
                "registered_at": r.registered_at,
            }
            for r in lineage
        ],
    }
