"""
Main inference orchestrator for Continual Counsel.

Dependency-constraint note: this file imports ONLY:
  - llama_cpp (llama-cpp-python) -- a C++ binding, NOT an HTTP client.
    Llama() loads a local GGUF file via mmap; it never opens a socket.
  - sentence_transformers -- local embedding model, no outbound calls.
  - faiss -- local index, no network access.
  - src.online.* -- our own modules, all similarly constrained.
  - Standard library: hashlib, json, datetime, dataclasses, pathlib, os.

No torch, no peft, no transformers (HuggingFace), no bitsandbytes,
no accelerate, no HTTP client of any kind (requests, httpx, aiohttp, urllib).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from sentence_transformers import SentenceTransformer

from src.online.confidence_gate import (
    InsufficientGrounding,
    check_retrieval_confidence,
)
from src.online.retrieval import (
    IndexMetadata,
    embed_query,
    load_index,
    query_index,
)
from src.online.verify import VerificationResult, re_generate_if_flagged, verify_answer


@dataclass
class InferenceResult:
    query: str
    answer: str
    retrieved_chunks: list[IndexMetadata]
    retrieval_scores: list[float]
    index_hash: str
    adapter_version_hash: str
    verification_result: VerificationResult
    was_regenerated: bool
    timestamp: str


def adapter_version_hash(adapter_dir: str | Path) -> str:
    """
    Compute a stable version identifier for a PEFT adapter by hashing
    adapter_config.json and all .safetensors files in lexicographic order.

    The adapter files themselves live in adapter_dir (written by the offline
    training pipeline). We only read them here -- no PEFT import needed.
    """
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")

    h = hashlib.sha256()
    # Hash the config first so the version changes whenever hyperparams change.
    h.update(config_path.read_bytes())

    for sf_path in sorted(adapter_dir.glob("*.safetensors")):
        # Stream in 64 KiB blocks to avoid loading large tensors into RAM.
        with sf_path.open("rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)

    return h.hexdigest()


class InferenceEngine:
    """
    Ties together the GGUF model, FAISS retrieval index, and embedding model
    into a single answer() call.

    All heavy objects are loaded once at construction time. The engine is
    intentionally stateless between calls so that it is safe to share across
    FastAPI request handlers.
    """

    def __init__(
        self,
        gguf_path: str | Path,
        adapter_dir: str | Path,
        index_dir: str | Path,
        embed_model_id: str,
        base_cfg: dict,
    ) -> None:
        """
        Parameters
        ----------
        gguf_path:
            Path to the quantized GGUF model file produced by the offline pipeline.
        adapter_dir:
            Directory containing adapter_config.json and .safetensors weights.
            Used only for version hashing; actual adapter fusion happens at GGUF
            export time (the weights are already baked into the GGUF).
        index_dir:
            Directory containing index.faiss and chunks.json.
        embed_model_id:
            SentenceTransformer model name or local path for query embedding.
        base_cfg:
            Arbitrary config dict (from YAML or env); used for generation params
            such as top_k, confidence_threshold, max_new_tokens.
        """
        gguf_path = Path(gguf_path)

        # 1. Load the frozen base model (GGUF) + LoRA adapter if they exist
        gguf_p = Path(gguf_path)
        if not gguf_p.exists() or Llama is None:
            print(f"WARNING: Llama missing or {gguf_p} not found. Running in MOCK inference mode.")
            self.model = None
        else:
            self.model = Llama(
                model_path=str(gguf_p),
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )

        self._llm = self.model
        self._index, self._chunks, self._index_hash = load_index(index_dir)
        self._embed_model = SentenceTransformer(embed_model_id)
        self._adapter_hash = adapter_version_hash(adapter_dir)
        self._cfg = base_cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, chunks: list[IndexMetadata]) -> list[dict]:
        context_blocks = []
        for i, c in enumerate(chunks, start=1):
            context_blocks.append(
                f"[{i}] (regime={c.regime}, section={c.section}, "
                f"effective={c.effective_date})\n{c.text}"
            )
        context = "\n\n".join(context_blocks)
        
        system_msg = (
            "You are Continual Counsel, a legal assistant specialising in regulatory compliance. "
            "Answer the question using ONLY the provided context. If the context does not contain "
            "sufficient information, say so explicitly rather than guessing.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Format your entire answer using clear, readable Markdown.\n"
            "2. Use **bold text** to highlight key deadlines, percentages, or critical terms.\n"
            "3. Use bullet points if you are listing multiple conditions or rules.\n"
            "4. Always explicitly cite the Regulation (e.g., Regulation A) and Section in your text."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

    def _generate(self, prompt) -> str:
        if self.model is None:
            return "[MOCK ANSWER] Based on the retrieved context, the notification deadline is 72 hours under Regulation A, or 48 hours for an AI incident under Regulation B."

        max_tokens = int(self._cfg.get("max_new_tokens", 512))
        temperature = float(self._cfg.get("temperature", 0.1))
        
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["message"]["content"].strip()

    def _llm_verifier(self, prompt: str) -> str:
        """Thin wrapper so verify_answer can call the LLM without knowing internals."""
        return self._generate(prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        regime_filter: Optional[str] = None,
    ) -> InferenceResult:
        """
        Full RAG pipeline:
          1. Embed query
          2. Retrieve top-k chunks (with optional regime filter)
          3. Confidence gate (raises InsufficientGrounding if score too low)
          4. Build prompt and generate
          5. Self-verify; regenerate once if flagged
          6. Return structured InferenceResult

        Raises
        ------
        InsufficientGrounding
            Caller (API layer) must catch this and return an escalation response.
        """
        top_k = int(self._cfg.get("top_k", 5))
        threshold = float(self._cfg.get("confidence_threshold", 0.35))

        # Step 1: embed
        qvec = embed_query(query, self._embed_model)

        # Step 2: retrieve
        raw_results = query_index(
            qvec,
            self._index,
            self._chunks,
            top_k=top_k,
            regime_filter=regime_filter,
        )

        # Step 3: gate -- raises InsufficientGrounding if max score < threshold
        try:
            raw_results = check_retrieval_confidence(raw_results, threshold)
        except InsufficientGrounding as exc:
            # Re-raise with the actual query text so callers can log it properly.
            raise InsufficientGrounding(
                max_score=exc.max_score,
                threshold=exc.threshold,
                query=query,
            ) from None

        retrieved_chunks = [m for m, _ in raw_results]
        retrieval_scores = [s for _, s in raw_results]

        # Step 4: generate
        prompt = self._build_prompt(query, retrieved_chunks)
        first_answer = self._generate(prompt)

        # Step 5: verify
        vresult = verify_answer(
            first_answer,
            retrieved_chunks,
            llm_fn=self._llm_verifier,
        )

        # Step 6: conditional regeneration
        if vresult.flagged_claims:
            # Regenerate with a stricter prompt that explicitly lists the flagged claims.
            flagged_str = "\n".join(f"- {c}" for c in vresult.flagged_claims)
            import copy
            strict_prompt = copy.deepcopy(prompt)
            strict_prompt[-1]["content"] += (
                "\n\nWARNING: The following claims in a previous draft could not be "
                "verified against the context. Do NOT repeat them unless the "
                f"context explicitly supports them:\n{flagged_str}"
            )
            final_answer, was_regenerated = re_generate_if_flagged(
                vresult,
                regenerate_fn=lambda: self._generate(strict_prompt),
            )
        else:
            final_answer = first_answer
            was_regenerated = False

        return InferenceResult(
            query=query,
            answer=final_answer if final_answer else first_answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_scores=retrieval_scores,
            index_hash=self._index_hash,
            adapter_version_hash=self._adapter_hash,
            verification_result=vresult,
            was_regenerated=was_regenerated,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
