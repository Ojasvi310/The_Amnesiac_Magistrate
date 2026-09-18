"""Benchmark gate (Step D) for the continual-counsel offline pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    regime_name: str
    accuracy: float
    n_correct: int
    n_total: int
    timestamp: str


def load_benchmark(regime_name: str, benchmarks_dir: str) -> list:
    """Load held-out QA pairs for a regime.

    Looks for data/regimes/<regime>/benchmark.json.
    If not found, falls back to generating trivial pairs from docs.json by
    treating each chunk as a self-answering question. This fallback is a best-
    effort measure and should be replaced with real held-out data in production.

    Returns a list of dicts with keys 'prompt' and 'expected_keywords'.
    """
    bench_path = Path(benchmarks_dir) / regime_name / "benchmark.json"
    if bench_path.exists():
        pairs = json.loads(bench_path.read_text(encoding="utf-8"))
        log.info("Loaded %d benchmark pairs for %s from %s", len(pairs), regime_name, bench_path)
        return pairs

    log.warning(
        "No benchmark.json found for %s at %s. Generating from docs.json (fallback).",
        regime_name, bench_path,
    )

    docs_path = Path(benchmarks_dir) / regime_name / "docs.json"
    if not docs_path.exists():
        log.error("No docs.json found for %s either. Returning empty benchmark.", regime_name)
        return []

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    pairs = []
    for doc in docs[:50]:  # cap at 50 to keep evaluation tractable
        text = doc.get("text", "")
        if not text:
            continue
        # Synthetic benchmark item: prompt asks for a summary, keywords are
        # the first few meaningful words from the source text.
        words = [w for w in text.split() if len(w) > 4][:5]
        pairs.append({
            "prompt": f"What does the following {regime_name} clause require?\n{text[:300]}",
            "expected_keywords": words,
        })
    log.info("Generated %d synthetic benchmark pairs for %s.", len(pairs), regime_name)
    return pairs


def _score_response(response: str, expected_keywords: list) -> bool:
    """Simple keyword-match scorer.

    Returns True if at least half of the expected keywords appear in the response.
    This is intentionally lightweight because the model is small and benchmark
    speed matters; semantic similarity scoring is overkill here.
    """
    if not expected_keywords:
        return True
    low_response = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in low_response)
    return hits >= max(1, len(expected_keywords) // 2)


def run_benchmark_suite(
    model: Any,
    tokenizer: Any,
    all_regimes: list,
    benchmarks_dir: str,
    max_new_tokens: int = 256,
) -> dict:
    """Run inference on all regimes and return per-regime BenchmarkResult.

    Returns a dict mapping regime_name -> BenchmarkResult.
    """
    results = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    for regime_name in all_regimes:
        pairs = load_benchmark(regime_name, benchmarks_dir)
        if not pairs:
            log.warning("Empty benchmark for %s; skipping.", regime_name)
            continue

        n_correct = 0
        n_total = len(pairs)

        for item in pairs:
            prompt = item.get("prompt", "")
            expected_keywords = item.get("expected_keywords", [])

            if not prompt:
                n_total -= 1
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            if _score_response(response, expected_keywords):
                n_correct += 1

        accuracy = n_correct / n_total if n_total > 0 else 0.0
        results[regime_name] = BenchmarkResult(
            regime_name=regime_name,
            accuracy=accuracy,
            n_correct=n_correct,
            n_total=n_total,
            timestamp=timestamp,
        )
        log.info(
            "Benchmark %s: accuracy=%.3f (%d/%d)",
            regime_name, accuracy, n_correct, n_total,
        )

    return results


def compute_bwt(current_scores: dict, prior_scores: dict) -> float:
    """Backward Transfer: average accuracy drop across all prior regimes.

    BWT = (1/N) * sum_i (current_i - prior_peak_i)

    Negative BWT means the model has forgotten prior knowledge (forgetting).
    A value of 0 means no forgetting; positive means backward improvement (unusual).

    prior_scores maps regime_name -> BenchmarkResult (the peak score before the
    current regime update).
    """
    if not prior_scores:
        return 0.0

    drops = []
    for regime_name, prior_result in prior_scores.items():
        current_result = current_scores.get(regime_name)
        if current_result is None:
            log.warning("No current score for prior regime %s; skipping in BWT.", regime_name)
            continue
        drop = current_result.accuracy - prior_result.accuracy
        drops.append(drop)

    if not drops:
        return 0.0

    bwt = sum(drops) / len(drops)
    return bwt


def check_bwt_gate(
    current_scores: dict,
    prior_scores: dict,
    threshold: float,
) -> tuple:
    """Check whether backward transfer exceeds the acceptable degradation threshold.

    Returns (passed: bool, failed_regimes: list[str], bwt: float).

    A regime fails if its accuracy has dropped by more than `threshold` relative
    to its prior peak. The overall gate passes only if no regime exceeds the threshold.
    """
    bwt = compute_bwt(current_scores, prior_scores)
    failed_regimes = []

    for regime_name, prior_result in prior_scores.items():
        current_result = current_scores.get(regime_name)
        if current_result is None:
            log.warning("Missing current score for %s; treating as failed.", regime_name)
            failed_regimes.append(regime_name)
            continue

        drop = prior_result.accuracy - current_result.accuracy
        if drop > threshold:
            log.warning(
                "BWT gate FAIL: %s dropped %.3f (threshold %.3f).",
                regime_name, drop, threshold,
            )
            failed_regimes.append(regime_name)

    passed = len(failed_regimes) == 0
    log.info(
        "BWT gate: passed=%s bwt=%.4f failed_regimes=%s",
        passed, bwt, failed_regimes,
    )
    return passed, failed_regimes, bwt


def save_benchmark_results(results: dict, run_dir: str, regime_name: str) -> Path:
    """Persist benchmark results to runs/<regime>_<timestamp>/validation_scores.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(run_dir)
    out_dir = run_dir / f"{regime_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_scores.json"

    serialisable = {
        name: asdict(result)
        for name, result in results.items()
    }
    out_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    log.info("Validation scores saved to %s", out_path)
    return out_path
