"""DPO polishing pass (Step C optional) for the continual-counsel offline pipeline."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

import torch

from src.config_loader import BaseConfig
from src.training_config import DPOConfig

log = logging.getLogger(__name__)


def _retrieve_context(faiss_snapshot_path: str, query: str, top_k: int = 3) -> str:
    """Retrieve top-k text chunks relevant to the query from a FAISS snapshot."""
    import json
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    snap = Path(faiss_snapshot_path)
    index = faiss.read_index(str(snap / "index.faiss"))
    chunks = json.loads((snap / "chunks.json").read_text(encoding="utf-8"))

    if index.ntotal == 0:
        return ""

    # Use a lightweight model; the embedding model used here is intentionally
    # the same one as during replay capture to keep retrieval semantics consistent.
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(vec, min(top_k, index.ntotal))
    return " ".join(chunks[i]["text"] for i in ids[0] if i >= 0)


def _generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single response at the specified temperature."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.9,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_contrastive_pairs(
    model: Any,
    tokenizer: Any,
    held_out_dataset: Any,
    faiss_snapshot_path: str,
    n_pairs: int = 100,
) -> list:
    """Generate contrastive preference pairs for DPO training.

    For each held-out prompt:
        chosen:   generated at high temperature with retrieved context prepended
                  (retrieval-supported, more likely to be grounded)
        rejected: generated without context (higher hallucination probability)

    The asymmetry between context-grounded and context-free generation is the
    signal we want DPO to amplify: the model should prefer retrieval-consistent
    answers over plausible-sounding but ungrounded ones.

    Returns a list of dicts: {prompt, chosen, rejected}.
    """
    pairs = []
    dataset_list = list(held_out_dataset)[:n_pairs]

    if not dataset_list:
        log.warning("Held-out dataset is empty; no contrastive pairs generated.")
        return []

    for item in dataset_list:
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        # Retrieval-grounded chosen response.
        context = _retrieve_context(faiss_snapshot_path, prompt, top_k=3)
        grounded_prompt = (
            f"Context:\n{context}\n\nQuestion:\n{prompt}" if context else prompt
        )
        try:
            chosen = _generate_response(model, tokenizer, grounded_prompt, temperature=0.8)
        except Exception as exc:
            log.warning("Chosen generation failed for prompt: %s", exc)
            continue

        # Hallucination-susceptible rejected response (no context, same temperature).
        try:
            rejected = _generate_response(model, tokenizer, prompt, temperature=0.8)
        except Exception as exc:
            log.warning("Rejected generation failed for prompt: %s", exc)
            continue

        # Discard trivially identical pairs -- DPO needs a meaningful preference gap.
        if chosen.strip() == rejected.strip():
            log.debug("Skipping pair with identical chosen/rejected responses.")
            continue

        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    log.info("Generated %d contrastive pairs (requested %d).", len(pairs), n_pairs)
    return pairs


def run_dpo_pass(
    model: Any,
    tokenizer: Any,
    contrastive_pairs: list,
    dpo_cfg: DPOConfig,
) -> Any:
    """Run a DPO training pass using TRL's DPO trainer.

    Returns the updated model. The model is modified in-place internally but we
    return it for clarity and to allow callers to swap references.
    """
    from datasets import Dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    if not contrastive_pairs:
        log.warning("No contrastive pairs provided; skipping DPO pass.")
        return model

    dpo_dataset = Dataset.from_list(contrastive_pairs)

    trl_config = TRLDPOConfig(
        output_dir="/tmp/dpo_output",
        num_train_epochs=dpo_cfg.num_epochs,
        per_device_train_batch_size=2,
        # DPO with very small batches benefits from gradient accumulation to get
        # stable preference gradient estimates.
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=dpo_cfg.beta,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None triggers implicit reference via the frozen base model copy
        args=trl_config,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )

    log.info("Starting DPO pass (beta=%.3f, epochs=%d).", dpo_cfg.beta, dpo_cfg.num_epochs)
    trainer.train()
    log.info("DPO pass complete.")
    return model


def dpo_with_revalidation(
    model: Any,
    tokenizer: Any,
    contrastive_pairs: list,
    dpo_cfg: DPOConfig,
    validate_fn: Callable,
) -> tuple:
    """Option (a): run DPO then re-run the Step D gate; fall back if BWT degrades.

    Workflow:
        1. Snapshot the pre-DPO model state.
        2. Run DPO pass.
        3. Call validate_fn(model) -> dict of benchmark scores.
        4. If BWT degrades (validate_fn returns indication of failure), restore
           the pre-DPO state and return (pre_dpo_model, accepted=False).
        5. Otherwise return (dpo_model, accepted=True).

    The validate_fn is expected to return a dict that includes a key "passed"
    (bool) at the top level, or raise an exception indicating failure. The dict
    is interpreted as: if d.get("passed", True) is False, DPO is rejected.

    Returns:
        (model, accepted: bool)
    """
    # Deep-copy the adapter state before DPO modifies it.
    # We copy only named parameters to avoid duplicating quantised base weights.
    pre_dpo_state = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    try:
        model = run_dpo_pass(model, tokenizer, contrastive_pairs, dpo_cfg)
    except Exception as exc:
        log.error("DPO pass raised an exception: %s. Reverting to pre-DPO state.", exc)
        _restore_state(model, pre_dpo_state)
        return model, False

    # Re-validate using the caller-supplied gate function.
    try:
        result = validate_fn(model)
    except Exception as exc:
        log.error("validate_fn raised after DPO: %s. Reverting.", exc)
        _restore_state(model, pre_dpo_state)
        return model, False

    passed = result.get("passed", True)
    if not passed:
        log.warning(
            "DPO revalidation gate failed (BWT degraded). Reverting to pre-DPO adapter."
        )
        _restore_state(model, pre_dpo_state)
        return model, False

    log.info("DPO revalidation gate passed. Keeping DPO-updated adapter.")
    return model, True


def _restore_state(model: Any, state_dict: dict) -> None:
    """Restore the model's trainable parameters from a previously snapshotted state dict."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
    log.info("Model state restored from pre-DPO snapshot.")
