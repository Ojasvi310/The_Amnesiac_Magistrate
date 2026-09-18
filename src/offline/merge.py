"""TIES-merge (Step E) for the continual-counsel offline pipeline.

Implemented from scratch using only numpy and torch -- no mergekit dependency
in the main path. mergekit is optionally used in _compare_with_mergekit() for
test-time validation of our implementation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch

log = logging.getLogger(__name__)


def ties_trim(delta: torch.Tensor, keep_top_pct: float) -> torch.Tensor:
    """Zero out the bottom (1 - keep_top_pct) parameters by absolute magnitude.

    Keeping only the top-magnitude values reduces interference between adapters
    while preserving the directions that contributed most to each regime's learning.

    keep_top_pct should be in (0, 1]. Values outside this range are clamped.
    """
    keep_top_pct = max(1e-6, min(1.0, keep_top_pct))
    if keep_top_pct == 1.0:
        return delta.clone()

    flat = delta.abs().flatten()
    k = max(1, int(flat.numel() * keep_top_pct))
    # kth-largest value as threshold.
    threshold = torch.topk(flat, k, largest=True).values[-1]

    trimmed = delta.clone()
    trimmed[delta.abs() < threshold] = 0.0
    return trimmed


def ties_sign_elect(trimmed_deltas: list) -> torch.Tensor:
    """Majority-vote sign election per element across all adapter deltas.

    For each parameter position, count positive and negative non-zero votes.
    The elected sign is +1 if positives >= negatives, else -1.
    Ties (equal counts) are broken by summing the magnitudes: whichever sign
    has greater total magnitude wins.

    All tensors in trimmed_deltas must have the same shape.
    """
    if not trimmed_deltas:
        raise ValueError("trimmed_deltas must be non-empty.")

    shape = trimmed_deltas[0].shape
    device = trimmed_deltas[0].device

    pos_count = torch.zeros(shape, device=device)
    neg_count = torch.zeros(shape, device=device)
    pos_mag = torch.zeros(shape, device=device)
    neg_mag = torch.zeros(shape, device=device)

    for delta in trimmed_deltas:
        pos_mask = delta > 0
        neg_mask = delta < 0
        pos_count = pos_count + pos_mask.float()
        neg_count = neg_count + neg_mask.float()
        pos_mag = pos_mag + delta * pos_mask.float()
        neg_mag = neg_mag + delta.abs() * neg_mask.float()

    # Where counts are equal, use magnitude as tiebreaker.
    elected = torch.where(
        pos_count > neg_count,
        torch.ones(shape, device=device),
        torch.where(
            neg_count > pos_count,
            -torch.ones(shape, device=device),
            # Tie -- use sign of whichever magnitude is larger.
            torch.where(pos_mag >= neg_mag,
                        torch.ones(shape, device=device),
                        -torch.ones(shape, device=device)),
        ),
    )
    return elected


def ties_disjoint_average(trimmed_deltas: list, elected_signs: torch.Tensor) -> torch.Tensor:
    """Average only parameters whose sign matches the elected sign.

    Parameters that are non-zero but disagree with the elected sign are excluded
    from the average, preventing sign-conflicted cancellations. Zero-trimmed
    values are also excluded (they don't contribute to the mean).
    """
    if not trimmed_deltas:
        raise ValueError("trimmed_deltas must be non-empty.")

    device = trimmed_deltas[0].device
    accumulated = torch.zeros_like(trimmed_deltas[0], device=device)
    counts = torch.zeros_like(trimmed_deltas[0], device=device)

    for delta in trimmed_deltas:
        # Mask: parameter is non-zero AND its sign matches the elected sign.
        sign_match = (delta * elected_signs) > 0
        accumulated = accumulated + delta * sign_match.float()
        counts = counts + sign_match.float()

    # Avoid division by zero for positions where no adapter agreed.
    safe_counts = counts.clone()
    safe_counts[safe_counts == 0] = 1.0
    merged = accumulated / safe_counts
    # Zero out positions where no adapter agreed (already zero in accumulated,
    # but make it explicit).
    merged[counts == 0] = 0.0
    return merged


def _load_adapter_flat(adapter_dir: str) -> dict:
    """Load A and B LoRA matrices from an adapter directory as flat float32 tensors.

    Looks for either adapter_model.bin or adapter_model.safetensors.
    Returns a dict: param_name -> tensor (on CPU, float32).
    """
    adapter_dir = Path(adapter_dir)

    # Prefer safetensors if available (faster, safer).
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_dir}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )

    # Convert all to float32 on CPU for numerically stable merging.
    return {k: v.float().cpu() for k, v in state.items() if "lora_A" in k or "lora_B" in k}


def ties_merge(adapter_dirs: list, keep_top_pct: float) -> dict:
    """Load all adapters and apply TIES trim/sign-elect/disjoint-average per parameter.

    Returns a merged state dict: param_name -> merged tensor.

    Only processes parameters that appear in ALL adapters; parameters missing
    from any adapter are skipped with a warning (can occur if target_modules differ).
    """
    if not adapter_dirs:
        raise ValueError("adapter_dirs must be non-empty.")

    log.info("Loading %d adapters for TIES merge.", len(adapter_dirs))
    all_states = [_load_adapter_flat(d) for d in adapter_dirs]

    # Intersect parameter names.
    common_keys = set(all_states[0].keys())
    for state in all_states[1:]:
        common_keys &= set(state.keys())

    missing_any = set(all_states[0].keys()) - common_keys
    if missing_any:
        log.warning(
            "The following parameters are not present in all adapters and will be skipped: %s",
            sorted(missing_any),
        )

    merged_state: dict = {}

    for key in sorted(common_keys):
        deltas = [state[key] for state in all_states]

        # Trim each adapter independently before sign election.
        trimmed = [ties_trim(d, keep_top_pct) for d in deltas]

        elected_signs = ties_sign_elect(trimmed)
        merged = ties_disjoint_average(trimmed, elected_signs)
        merged_state[key] = merged

    log.info("TIES merge complete: %d parameter tensors merged.", len(merged_state))
    return merged_state


def merge_and_validate(
    adapter_dirs: list,
    base_model_id: str,
    output_dir: str,
    validate_fn: Callable,
    train_cfg: Any,
    base_cfg: Any,
) -> tuple:
    """Orchestrate TIES merge, save the merged adapter, then validate.

    Steps:
        1. TIES-merge all adapter dirs.
        2. Save the merged state dict alongside a minimal adapter_config.json
           copied from the first adapter (all should share the same PEFT config).
        3. Call validate_fn(merged_adapter_dir) -> dict of benchmark scores.
        4. Return (merged_adapter_dir, accepted: bool, scores).
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_state = ties_merge(adapter_dirs, train_cfg.merge.ties_keep_top_pct)

    # Save merged weights.
    merged_weights_path = output_dir / "adapter_model.bin"
    torch.save(merged_state, str(merged_weights_path))
    log.info("Merged adapter weights saved to %s", merged_weights_path)

    # Copy PEFT config from first adapter so the merged dir is a valid adapter dir.
    for config_filename in ["adapter_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = Path(adapter_dirs[0]) / config_filename
        if src.exists():
            shutil.copy2(str(src), str(output_dir / config_filename))

    # Validate the merged adapter.
    try:
        scores = validate_fn(str(output_dir))
    except Exception as exc:
        log.error("Validation failed for merged adapter: %s", exc)
        return output_dir, False, {}

    passed = scores.get("passed", True) if isinstance(scores, dict) else True
    log.info("Merged adapter validation: passed=%s", passed)
    return output_dir, passed, scores


def _compare_with_mergekit(
    adapter_dirs: list,
    keep_top_pct: float,
    our_result: dict,
) -> None:
    """Test-time comparison: run mergekit's TIES on the same inputs and log L2 difference.

    This function is NEVER called in the main pipeline path. It exists to
    validate our from-scratch TIES implementation against the reference library.
    Only called from tests when mergekit is available.
    """
    try:
        import mergekit  # noqa: F401
    except ImportError:
        log.info("mergekit not importable; skipping reference comparison.")
        return

    log.info("mergekit is available; running reference TIES for comparison.")

    try:
        from mergekit.merge import MergeConfig, run_merge
        import tempfile, os

        merge_cfg = MergeConfig(
            merge_method="ties",
            models=[{"model": d} for d in adapter_dirs],
            parameters={"density": keep_top_pct},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_merge(merge_cfg, out_path=tmpdir)
            # Load mergekit result and compute per-parameter L2 vs our result.
            mk_state = _load_adapter_flat(tmpdir)
            for key in our_result:
                if key in mk_state:
                    l2 = (our_result[key] - mk_state[key]).norm().item()
                    log.info("mergekit comparison [%s]: L2 diff = %.6f", key, l2)
                else:
                    log.warning("Key %s not found in mergekit output.", key)
    except Exception as exc:
        log.warning("mergekit comparison failed: %s", exc)
