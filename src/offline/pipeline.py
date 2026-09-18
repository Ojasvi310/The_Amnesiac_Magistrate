"""Top-level offline training orchestrator (Steps A -> F) for continual-counsel."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _log_startup_info() -> None:
    """Log Python version and GPU info at pipeline startup."""
    log.info("Python version: %s", sys.version)
    try:
        import torch
        if torch.cuda.is_available():
            log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            log.info("No CUDA device available.")
    except ImportError:
        log.warning("torch not importable; cannot detect GPU.")


def _load_configs(repo_root: Path, config_profile: str) -> tuple:
    """Load base and training configs; return (base_cfg, train_cfg)."""
    from src.config_loader import load_base_config
    from src.training_config import load_training_config

    base_cfg = load_base_config(profile=config_profile, repo_root=repo_root)
    train_cfg = load_training_config(path=repo_root / "configs" / "training.yaml")
    return base_cfg, train_cfg


def _count_accepted_adapters(runs_dir: Path) -> int:
    """Count the number of previously accepted regime adapters in runs/."""
    if not runs_dir.exists():
        return 0
    # Each accepted regime has a summary.json with "accepted": true.
    count = 0
    for summary_file in runs_dir.glob("*/summary.json"):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            if data.get("accepted", False):
                count += 1
        except Exception:
            pass
    return count


def _collect_prior_adapter_dirs(runs_dir: Path) -> list:
    """Collect adapter directories from all previously accepted runs."""
    adapter_dirs = []
    for summary_file in sorted(runs_dir.glob("*/summary.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            if data.get("accepted", False):
                adapter_dir = summary_file.parent / "adapter"
                if adapter_dir.exists():
                    adapter_dirs.append(str(adapter_dir))
        except Exception as exc:
            log.warning("Could not read summary from %s: %s", summary_file, exc)
    return adapter_dirs


def _collect_prior_scores(runs_dir: Path) -> dict:
    """Collect peak benchmark scores per regime from prior accepted runs."""
    from src.offline.validate import BenchmarkResult

    prior_scores: dict = {}
    for scores_file in sorted(runs_dir.glob("*/validation_scores.json")):
        try:
            data = json.loads(scores_file.read_text(encoding="utf-8"))
            for regime_name, result_data in data.items():
                existing = prior_scores.get(regime_name)
                current_acc = result_data.get("accuracy", 0.0)
                if existing is None or current_acc > existing.accuracy:
                    prior_scores[regime_name] = BenchmarkResult(
                        regime_name=regime_name,
                        accuracy=current_acc,
                        n_correct=result_data.get("n_correct", 0),
                        n_total=result_data.get("n_total", 0),
                        timestamp=result_data.get("timestamp", ""),
                    )
        except Exception as exc:
            log.warning("Could not read scores from %s: %s", scores_file, exc)
    return prior_scores


def run_regime_update(
    regime_name: str,
    input_bundle_dir: str,
    config_profile: str,
    repo_root: str,
) -> dict:
    """Orchestrate the full offline continual-learning pipeline for one regime.

    Steps:
        A  -> Capture golden replay set (pre-training, using FAISS from input bundle)
        B+C -> Compose training batch and train O-LoRA adapter (retry loop on BWT fail)
        D  -> Validate; reject if BWT gate fails after max retries
        E  -> TIES merge if cadence reached; validate merged adapter
        F  -> Export GGUF and assemble bundle if export cadence says so

    Returns a run summary dict; also writes it to runs/<regime>_<timestamp>/summary.json.
    """
    _configure_logging()
    _log_startup_info()

    repo_root = Path(repo_root)
    input_bundle_dir = Path(input_bundle_dir)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log.info("=== run_regime_update regime=%s profile=%s ===", regime_name, config_profile)

    base_cfg, train_cfg = _load_configs(repo_root, config_profile)

    paths = {
        "data_regimes": repo_root / base_cfg.paths.data_regimes,
        "replay_cache": repo_root / base_cfg.paths.replay_cache,
        "exports": repo_root / base_cfg.paths.exports,
        "runs": repo_root / base_cfg.paths.runs,
    }

    run_dir = paths["runs"] / f"{regime_name}_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    regime_dir = paths["data_regimes"] / regime_name
    faiss_snapshot_path = input_bundle_dir / "faiss_snapshot"
    if not faiss_snapshot_path.exists():
        # Fallback: look for faiss_snapshot inside the regime dir.
        faiss_snapshot_path = regime_dir / "faiss_snapshot"

    prompt_bank_path = input_bundle_dir / "prompt_bank.json"
    if not prompt_bank_path.exists():
        prompt_bank_path = regime_dir / "prompt_bank.json"

    base_model_id = base_cfg.profile.model_id

    # -------------------------------------------------------------------------
    # Load tokenizer and base model early; they're shared across steps A-D.
    # -------------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from peft import PeftModel

    log.info("Loading base model %s for inference steps.", base_model_id)
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        log.warning("4-bit loading failed (%s); falling back to fp32 CPU load.", exc)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Step A (pre-training): capture golden replay set
    # -------------------------------------------------------------------------
    log.info("Step A: Capturing golden set for %s.", regime_name)
    from src.offline.replay_gen import capture_golden_set

    golden_set_path = None
    if prompt_bank_path.exists() and faiss_snapshot_path.exists():
        try:
            golden_set_path = capture_golden_set(
                model=base_model,
                tokenizer=tokenizer,
                regime_dir=str(regime_dir),
                prompt_bank_path=str(prompt_bank_path),
                faiss_snapshot_path=str(faiss_snapshot_path),
                config=base_cfg,
                train_cfg=train_cfg,
            )
            log.info("Step A complete: golden set at %s.", golden_set_path)
        except Exception as exc:
            log.warning("Step A failed: %s. Continuing without golden set.", exc)
    else:
        log.warning(
            "Skipping Step A: prompt bank (%s) or FAISS snapshot (%s) not found.",
            prompt_bank_path, faiss_snapshot_path,
        )

    # -------------------------------------------------------------------------
    # Step B+C: train O-LoRA adapter (with retry loop on BWT failure)
    # -------------------------------------------------------------------------
    log.info("Steps B+C: Training O-LoRA adapter for %s.", regime_name)
    from src.offline.train_adapter import OrthonormalBasis, compose_training_batch, train_lora, bump_lambda
    from src.offline.validate import (
        run_benchmark_suite, check_bwt_gate, save_benchmark_results, BenchmarkResult
    )

    # Load the basis checkpoint if one exists from a prior run.
    basis = OrthonormalBasis()
    basis_checkpoint_path = paths["runs"] / "basis_checkpoint.pt"
    if basis_checkpoint_path.exists():
        try:
            saved_Q = torch.load(str(basis_checkpoint_path), map_location="cpu")
            basis._Q = saved_Q
            log.info("Loaded basis checkpoint from %s (rank=%d).", basis_checkpoint_path, basis.rank)
        except Exception as exc:
            log.warning("Failed to load basis checkpoint: %s. Starting fresh.", exc)

    training_batch = compose_training_batch(
        regime_dir=str(regime_dir),
        replay_cache_dir=str(paths["replay_cache"]),
        regime_name=regime_name,
        replay_ratio=train_cfg.replay.ratio,
    )

    current_lambda = train_cfg.orthogonality.lambda_weight
    adapter_dir = None
    prior_scores = _collect_prior_scores(paths["runs"])
    all_known_regimes = list(prior_scores.keys()) + [regime_name]

    bwt_passed = False
    failed_regimes_final: list = []
    bwt_final = 0.0
    current_scores: dict = {}

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        log.info(
            "Training attempt %d/%d (lambda=%.4f).", attempt + 1, MAX_RETRIES, current_lambda
        )
        # Temporarily override lambda for this attempt.
        original_lambda = train_cfg.orthogonality.lambda_weight
        train_cfg.orthogonality.lambda_weight = current_lambda

        try:
            adapter_dir = train_lora(
                base_model_id=base_model_id,
                training_batch=training_batch,
                basis=basis,
                train_cfg=train_cfg,
                base_cfg=base_cfg,
                regime_name=regime_name,
                run_dir=str(run_dir),
            )
        except Exception as exc:
            log.error("Training failed on attempt %d: %s", attempt + 1, exc)
            train_cfg.orthogonality.lambda_weight = original_lambda
            break

        train_cfg.orthogonality.lambda_weight = original_lambda

        # Step D: validate
        log.info("Step D: Running benchmark suite (attempt %d).", attempt + 1)
        try:
            eval_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            current_scores = run_benchmark_suite(
                model=eval_model,
                tokenizer=tokenizer,
                all_regimes=all_known_regimes,
                benchmarks_dir=str(paths["data_regimes"]),
            )
        except Exception as exc:
            log.error("Benchmark suite failed on attempt %d: %s", attempt + 1, exc)
            break

        save_benchmark_results(current_scores, str(paths["runs"]), regime_name)

        bwt_passed, failed_regimes_final, bwt_final = check_bwt_gate(
            current_scores=current_scores,
            prior_scores=prior_scores,
            threshold=train_cfg.gates.bwt_degradation_threshold,
        )

        if bwt_passed:
            log.info("Step D: BWT gate passed on attempt %d.", attempt + 1)
            break

        log.warning(
            "Step D: BWT gate failed (failed_regimes=%s, bwt=%.4f). "
            "Bumping lambda and retrying.",
            failed_regimes_final, bwt_final,
        )
        current_lambda = bump_lambda(current_lambda, train_cfg)

    if not bwt_passed:
        log.error(
            "BWT gate failed after %d attempts. Regime %s NOT accepted.", MAX_RETRIES, regime_name
        )
        summary = _write_summary(
            run_dir=run_dir,
            regime_name=regime_name,
            timestamp_str=timestamp_str,
            accepted=False,
            bwt=bwt_final,
            failed_regimes=failed_regimes_final,
            adapter_dir=adapter_dir,
            bundle_dir=None,
        )
        return summary

    # Persist updated basis.
    if basis._Q is not None:
        torch.save(basis._Q, str(basis_checkpoint_path))
        log.info("Basis checkpoint saved to %s.", basis_checkpoint_path)

    # -------------------------------------------------------------------------
    # DPO polishing pass (optional, Steps C DPO)
    # -------------------------------------------------------------------------
    if train_cfg.dpo.enabled and adapter_dir is not None:
        log.info("DPO pass enabled; generating contrastive pairs.")
        from src.offline.dpo_polish import generate_contrastive_pairs, dpo_with_revalidation
        from datasets import Dataset

        eval_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

        # Build a small held-out dataset from the benchmark pairs.
        held_out_pairs = []
        bench_path = paths["data_regimes"] / regime_name / "benchmark.json"
        if bench_path.exists():
            held_out_pairs = json.loads(bench_path.read_text(encoding="utf-8"))[:50]
        held_out_ds = Dataset.from_list(held_out_pairs) if held_out_pairs else Dataset.from_list([])

        if faiss_snapshot_path.exists():
            contrastive_pairs = generate_contrastive_pairs(
                model=eval_model,
                tokenizer=tokenizer,
                held_out_dataset=held_out_ds,
                faiss_snapshot_path=str(faiss_snapshot_path),
                n_pairs=100,
            )

            def _quick_validate(m: Any) -> dict:
                """Lightweight post-DPO gate: re-check BWT on the updated model."""
                scores = run_benchmark_suite(
                    model=m,
                    tokenizer=tokenizer,
                    all_regimes=all_known_regimes,
                    benchmarks_dir=str(paths["data_regimes"]),
                )
                passed, _, _ = check_bwt_gate(scores, prior_scores, train_cfg.gates.bwt_degradation_threshold)
                return {"passed": passed, "scores": {k: v.accuracy for k, v in scores.items()}}

            eval_model, dpo_accepted = dpo_with_revalidation(
                model=eval_model,
                tokenizer=tokenizer,
                contrastive_pairs=contrastive_pairs,
                dpo_cfg=train_cfg.dpo,
                validate_fn=_quick_validate,
            )
            if dpo_accepted:
                eval_model.save_pretrained(str(adapter_dir))
                log.info("DPO-updated adapter saved.")
            else:
                log.info("DPO rejected; keeping pre-DPO adapter.")
        else:
            log.warning("FAISS snapshot not found; skipping DPO pass.")

    # -------------------------------------------------------------------------
    # Step E: TIES merge if cadence reached
    # -------------------------------------------------------------------------
    merged_adapter_dir = None
    merge_accepted = False
    cadence = train_cfg.merge.cadence

    if isinstance(cadence, int):
        n_accepted = _count_accepted_adapters(paths["runs"])
        should_merge = (n_accepted + 1) % cadence == 0  # +1 for current
    else:
        # cadence == "never" or any non-integer string disables merging.
        should_merge = False

    if should_merge and adapter_dir is not None:
        log.info("Step E: TIES merge cadence reached. Merging adapters.")
        from src.offline.merge import merge_and_validate

        prior_adapter_dirs = _collect_prior_adapter_dirs(paths["runs"])
        all_adapter_dirs = prior_adapter_dirs + [str(adapter_dir)]

        if len(all_adapter_dirs) > 1:
            merged_output_dir = paths["runs"] / f"{regime_name}_{timestamp_str}_merged"

            def _validate_merged(merged_dir: str) -> dict:
                merged_model = PeftModel.from_pretrained(base_model, merged_dir)
                scores = run_benchmark_suite(
                    model=merged_model,
                    tokenizer=tokenizer,
                    all_regimes=all_known_regimes,
                    benchmarks_dir=str(paths["data_regimes"]),
                )
                passed, _, bwt = check_bwt_gate(
                    scores, prior_scores, train_cfg.merge.post_merge_bwt_tolerance
                )
                return {"passed": passed, "bwt": bwt}

            merged_adapter_dir, merge_accepted, merge_scores = merge_and_validate(
                adapter_dirs=all_adapter_dirs,
                base_model_id=base_model_id,
                output_dir=str(merged_output_dir),
                validate_fn=_validate_merged,
                train_cfg=train_cfg,
                base_cfg=base_cfg,
            )

            if merge_accepted:
                # Reset basis from merged adapter to prevent rank explosion.
                import safetensors.torch as sft
                from pathlib import Path as _P
                merged_state_path = _P(str(merged_adapter_dir)) / "adapter_model.bin"
                if merged_state_path.exists():
                    merged_state = torch.load(str(merged_state_path), map_location="cpu")
                    merged_A = [v for k, v in merged_state.items() if "lora_A" in k]
                    if merged_A:
                        basis.reset_from_merged(merged_A)
                        torch.save(basis._Q, str(basis_checkpoint_path))
                log.info("Basis reset from merged adapter.")
            else:
                log.warning("Merged adapter failed validation; keeping per-regime adapter.")
                merged_adapter_dir = None
        else:
            log.info("Only one adapter available; skipping TIES merge.")

    # The active adapter for export is the merged one (if accepted) or the per-regime one.
    active_adapter_dir = str(merged_adapter_dir) if merge_accepted and merged_adapter_dir else str(adapter_dir)

    # -------------------------------------------------------------------------
    # Step F: GGUF export and bundle assembly
    # -------------------------------------------------------------------------
    bundle_dir = None
    export_cadence = train_cfg.export.cadence

    should_export = (
        export_cadence == "every_adapter"
        or (export_cadence == "merge_only" and merge_accepted)
    )

    if should_export and active_adapter_dir:
        log.info("Step F: Exporting GGUF and assembling bundle.")
        from src.offline.export import merge_to_fp16, convert_to_gguf, assemble_bundle

        llama_cpp_dir = str(repo_root / base_cfg.paths.llama_cpp_dir)
        quant_level = base_cfg.profile.gguf_quant_level

        merged_fp16_dir = run_dir / "merged_fp16"
        gguf_output_path = run_dir / f"model_{quant_level}.gguf"

        try:
            merge_to_fp16(
                adapter_dir=active_adapter_dir,
                base_model_id=base_model_id,
                output_dir=str(merged_fp16_dir),
                base_cfg=base_cfg,
            )
            convert_to_gguf(
                merged_model_dir=str(merged_fp16_dir),
                llama_cpp_dir=llama_cpp_dir,
                output_path=str(gguf_output_path),
                quant_level=quant_level,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            log.error("GGUF export failed: %s. Bundle will be assembled without GGUF.", exc)
            gguf_output_path = None

        import subprocess as _sp
        try:
            commit_hash = _sp.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            commit_hash = "unknown"

        golden_set_dir = str(paths["replay_cache"] / regime_name / "golden")
        basis_ckpt = str(basis_checkpoint_path) if basis_checkpoint_path.exists() else ""

        bundle_dir = assemble_bundle(
            regime=regime_name,
            timestamp_str=timestamp_str,
            adapter_dir=active_adapter_dir,
            gguf_path=str(gguf_output_path) if gguf_output_path else "",
            basis_checkpoint=basis_ckpt,
            golden_set_dir=golden_set_dir,
            benchmark_results=current_scores,
            commit_hash=commit_hash,
            python_version=sys.version,
            exports_root=str(paths["exports"]),
        )
        log.info("Bundle assembled at %s.", bundle_dir)

    summary = _write_summary(
        run_dir=run_dir,
        regime_name=regime_name,
        timestamp_str=timestamp_str,
        accepted=True,
        bwt=bwt_final,
        failed_regimes=[],
        adapter_dir=active_adapter_dir,
        bundle_dir=str(bundle_dir) if bundle_dir else None,
    )

    log.info("=== run_regime_update COMPLETE: regime=%s accepted=True ===", regime_name)
    return summary


def _write_summary(
    run_dir: Path,
    regime_name: str,
    timestamp_str: str,
    accepted: bool,
    bwt: float,
    failed_regimes: list,
    adapter_dir: Any,
    bundle_dir: Any,
) -> dict:
    """Write and return the run summary JSON."""
    summary = {
        "regime": regime_name,
        "timestamp": timestamp_str,
        "accepted": accepted,
        "bwt": bwt,
        "failed_regimes": failed_regimes,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "bundle_dir": str(bundle_dir) if bundle_dir else None,
        "python_version": sys.version,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Run summary written to %s.", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual Counsel offline training pipeline."
    )
    parser.add_argument(
        "--regime",
        required=True,
        help="Name of the compliance regime to train on (e.g. 'gdpr', 'hipaa').",
    )
    parser.add_argument(
        "--input-bundle",
        required=True,
        help=(
            "Path to the input bundle directory containing FAISS snapshot, "
            "prompt bank, and any prior artifacts."
        ),
    )
    parser.add_argument(
        "--profile",
        default="proxy",
        help="Config profile name (default: proxy). Controls model_id and VRAM budget.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Absolute or relative path to the repo root (default: current directory).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    summary = run_regime_update(
        regime_name=args.regime,
        input_bundle_dir=args.input_bundle,
        config_profile=args.profile,
        repo_root=str(repo_root),
    )

    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary.get("accepted", False) else 1)


if __name__ == "__main__":
    main()
