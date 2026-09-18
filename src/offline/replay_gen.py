"""Golden replay capture (Step A) for the continual-counsel offline pipeline.

NOTE: This module is intended to run on a Colab GPU. The model calls require
      a CUDA device. Mark integration tests with @pytest.mark.colab_only.
"""
from __future__ import annotations

# @pytest.mark.colab_only -- GPU / model-generation required
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config_loader import BaseConfig
from src.training_config import TrainingConfig

log = logging.getLogger(__name__)

# Legal terms used as cheap conclusion signals for two-seed self-consistency.
# The small model (Qwen2.5-1.5B) is not reliable enough for nuanced semantic comparison,
# so we fall back to checking whether a fixed vocabulary of outcome-bearing terms
# appears or is absent in both seeds identically.
_LEGAL_CONCLUSION_TERMS = [
    "must", "shall", "required", "prohibited", "may not", "is exempt",
    "compliance", "violation", "penalty", "mandatory", "permissible",
    "not required", "optional", "subject to", "pursuant to",
]


def _extract_conclusion_signal(text: str) -> frozenset:
    """Return the frozenset of legal conclusion terms present in text (lowercased)."""
    low = text.lower()
    return frozenset(term for term in _LEGAL_CONCLUSION_TERMS if term in low)


def _seeds_agree(resp_a: str, resp_b: str) -> bool:
    """True if both seed responses share the same conclusion-term fingerprint."""
    return _extract_conclusion_signal(resp_a) == _extract_conclusion_signal(resp_b)


def _load_faiss_snapshot(faiss_snapshot_path: str) -> tuple:
    snap = Path(faiss_snapshot_path)
    index_path = snap / "index.faiss"
    chunks_path = snap / "chunks.json"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found: {chunks_path}")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, chunks


def _is_grounded(
    answer: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks: list,
    top_k: int,
) -> bool:
    """Embed the answer and check whether at least one top-k retrieved chunk
    contains any of the answer's legal conclusion terms.

    If the retrieval result is empty (degenerate index), we discard the pair to
    be conservative rather than letting unverified content into the golden set.
    """
    if index.ntotal == 0:
        log.warning("FAISS index is empty -- discarding pair as ungrounded.")
        return False

    vec = embed_model.encode([answer], normalize_embeddings=True).astype(np.float32)
    distances, neighbour_ids = index.search(vec, min(top_k, index.ntotal))
    retrieved_text = " ".join(chunks[i]["text"] for i in neighbour_ids[0] if i >= 0)

    if not retrieved_text.strip():
        return False

    answer_terms = _extract_conclusion_signal(answer)
    if not answer_terms:
        # No legal conclusion terms in answer -- can't verify grounding; discard.
        return False

    retrieved_terms = _extract_conclusion_signal(retrieved_text)
    # Require at least one conclusion term to appear in retrieved context.
    return bool(answer_terms & retrieved_terms)


def _generate_with_seed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    seed: int,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response with a fixed seed for reproducibility."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    # Strip the input tokens from the output.
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def capture_golden_set(
    model: Any,
    tokenizer: Any,
    regime_dir: str,
    prompt_bank_path: str,
    faiss_snapshot_path: str,
    config: BaseConfig,
    train_cfg: TrainingConfig,
) -> Path:
    """Step A: generate and filter a golden replay set for a compliance regime.

    Saves between train_cfg.replay.golden_set_min and golden_set_max QA pairs
    (whichever is reached first after exhausting the prompt bank) to:
        data/replay_cache/<regime>/golden/golden_set.jsonl

    Filtering pipeline per candidate pair:
        1. Two-seed generation with self-consistency check (legal conclusion terms).
        2. Retrieval-grounding check via FAISS snapshot.

    Returns the path to the saved .jsonl file.
    """
    regime_dir = Path(regime_dir)
    prompt_bank_path = Path(prompt_bank_path)
    faiss_snapshot_path = Path(faiss_snapshot_path)

    if not prompt_bank_path.exists():
        raise FileNotFoundError(f"Prompt bank not found: {prompt_bank_path}")

    prompt_bank: list = json.loads(prompt_bank_path.read_text(encoding="utf-8"))
    if not prompt_bank:
        raise ValueError("Prompt bank is empty.")

    index, chunks = _load_faiss_snapshot(str(faiss_snapshot_path))

    embed_model_id = config.retrieval.embed_model_id
    log.info("Loading embedding model %s for grounding filter", embed_model_id)
    embed_model = SentenceTransformer(embed_model_id)

    regime_name = regime_dir.name
    out_dir = Path(config.paths.replay_cache) / regime_name / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "golden_set.jsonl"

    target_min = train_cfg.replay.golden_set_min
    target_max = train_cfg.replay.golden_set_max
    top_k = config.retrieval.top_k

    # Shuffle prompt bank so repeated calls (e.g. retries) explore different orderings.
    shuffled_bank = prompt_bank[:]
    random.shuffle(shuffled_bank)

    collected = []
    skipped_consistency = 0
    skipped_grounding = 0

    for template in shuffled_bank:
        if len(collected) >= target_max:
            break

        regulation = template.get("regulation", regime_name)
        section = template.get("section", "General")
        instruction = template.get("instruction", "")

        if not instruction:
            log.warning("Skipping template with empty instruction field.")
            continue

        prompt = instruction.format(regulation=regulation, section=section)

        resp_a = _generate_with_seed(model, tokenizer, prompt, seed=42)
        resp_b = _generate_with_seed(model, tokenizer, prompt, seed=137)

        if not _seeds_agree(resp_a, resp_b):
            skipped_consistency += 1
            log.debug("Pair discarded: seed disagreement.")
            continue

        # Use seed-A response as the canonical answer (first seed is more conservative).
        if not _is_grounded(resp_a, embed_model, index, chunks, top_k):
            skipped_grounding += 1
            log.debug("Pair discarded: grounding check failed.")
            continue

        collected.append({
            "prompt": prompt,
            "response": resp_a,
            "regime": regime_name,
            "seed_agreement": True,
            "grounded": True,
        })

    if len(collected) < target_min:
        log.warning(
            "Only %d pairs collected for regime %s (minimum %d). "
            "Consider expanding the prompt bank.",
            len(collected), regime_name, target_min,
        )

    log.info(
        "Golden set stats for %s: collected=%d skipped_consistency=%d skipped_grounding=%d",
        regime_name, len(collected), skipped_consistency, skipped_grounding,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for record in collected:
            f.write(json.dumps(record) + "\n")

    log.info("Golden set saved to %s", out_path)
    return out_path


def refresh_golden_sets(
    model: Any,
    tokenizer: Any,
    all_regimes: list,
    config: BaseConfig,
    train_cfg: TrainingConfig,
) -> dict:
    """Config-gated golden-set refresh. Off by default (train_cfg.replay.refresh_enabled).

    For each regime, generates a fresh drift-summary golden set and computes the
    fraction of items that differ from the existing set's conclusion fingerprints.
    Pre-refresh sets are kept under golden_set_<timestamp>.jsonl.bak.

    Returns a dict mapping regime_name -> drift_fraction.
    """
    if not train_cfg.replay.refresh_enabled:
        log.info("Golden set refresh is disabled (replay.refresh_enabled=False). Skipping.")
        return {}

    drift_report: dict = {}

    for regime_name in all_regimes:
        golden_dir = Path(config.paths.replay_cache) / regime_name / "golden"
        existing_path = golden_dir / "golden_set.jsonl"

        if not existing_path.exists():
            log.warning("No existing golden set for %s; skipping refresh.", regime_name)
            continue

        # Backup before overwriting.
        timestamp = int(time.time())
        backup_path = golden_dir / f"golden_set_{timestamp}.jsonl.bak"
        existing_path.rename(backup_path)
        log.info("Backed up existing golden set to %s", backup_path)

        # Load existing fingerprints.
        existing_records = [
            json.loads(line)
            for line in backup_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing_fingerprints = {
            frozenset(_extract_conclusion_signal(r["response"]))
            for r in existing_records
        }

        # Locate regime dir and FAISS snapshot.
        regime_dir = Path(config.paths.data_regimes) / regime_name
        faiss_snapshot_path = regime_dir / "faiss_snapshot"

        # Determine prompt bank path; fall back to a regime-level bank.
        prompt_bank_path = regime_dir / "prompt_bank.json"
        if not prompt_bank_path.exists():
            log.warning(
                "No prompt bank found for %s at %s; skipping refresh.",
                regime_name, prompt_bank_path,
            )
            # Restore backup so the golden set is not lost.
            backup_path.rename(existing_path)
            continue

        try:
            new_path = capture_golden_set(
                model=model,
                tokenizer=tokenizer,
                regime_dir=regime_dir,
                prompt_bank_path=prompt_bank_path,
                faiss_snapshot_path=faiss_snapshot_path,
                config=config,
                train_cfg=train_cfg,
            )
        except Exception as exc:
            log.error("Refresh failed for %s: %s. Restoring backup.", regime_name, exc)
            backup_path.rename(existing_path)
            continue

        new_records = [
            json.loads(line)
            for line in new_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        new_fingerprints = {
            frozenset(_extract_conclusion_signal(r["response"]))
            for r in new_records
        }

        # Drift = fraction of new fingerprints not in the old set.
        if new_fingerprints:
            drift = len(new_fingerprints - existing_fingerprints) / len(new_fingerprints)
        else:
            drift = 0.0

        drift_report[regime_name] = drift
        log.info("Drift fraction for %s: %.3f", regime_name, drift)

    return drift_report
