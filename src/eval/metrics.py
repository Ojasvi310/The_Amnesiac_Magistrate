"""Evaluation metrics for Continual Counsel."""
from __future__ import annotations

import json
import os
import psutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def compute_accuracy(benchmark_results: dict[str, list[float]]) -> dict[str, float]:
    """Mean accuracy per regime."""
    return {
        regime: float(np.mean(scores)) if scores else 0.0
        for regime, scores in benchmark_results.items()
    }


def compute_bwt(accuracy_history: list[dict[str, float]]) -> float:
    """Backward transfer (BWT).
    
    BWT = (1/T-1) * sum_{i=1}^{T-1} (a_{T,i} - a_{i,i})
    where a_{T,i} is current accuracy on regime i and a_{i,i} was accuracy 
    right after training on regime i.
    """
    if len(accuracy_history) < 2:
        return 0.0

    current_scores = accuracy_history[-1]
    T = len(accuracy_history)
    regimes_seen = list(current_scores.keys())
    
    bwt_sum = 0.0
    count = 0
    
    # We can only compute BWT for regimes we've trained on and evaluated subsequently.
    for i in range(T - 1):
        regime = regimes_seen[i]
        # a_{i,i}: accuracy on regime i immediately after it was introduced
        peak_acc = accuracy_history[i].get(regime, 0.0)
        # a_{T,i}: current accuracy on regime i
        curr_acc = current_scores.get(regime, 0.0)
        
        bwt_sum += (curr_acc - peak_acc)
        count += 1
        
    return bwt_sum / max(count, 1)


def compute_fwt(accuracy_history: list[dict[str, float]]) -> float:
    """Forward transfer (FWT).
    
    FWT = (1/T-1) * sum_{i=2}^T (a_{i-1, i} - random_baseline)
    """
    if len(accuracy_history) < 2:
        return 0.0

    fwt_sum = 0.0
    count = 0
    # This assumes evaluating on a regime before training on it.
    # We approximate random baseline as 0.0 for open-ended QA.
    baseline = 0.0 
    
    # Since our pipeline might not evaluate future regimes before they arrive,
    # FWT might be zero or require access to pre-training evaluation.
    # Implementing standard formulation based on available history.
    for i in range(1, len(accuracy_history)):
        regimes_seen = list(accuracy_history[i].keys())
        new_regime = regimes_seen[-1] # assumption: last key is the new one
        pre_train_acc = accuracy_history[i-1].get(new_regime, baseline)
        fwt_sum += (pre_train_acc - baseline)
        count += 1
        
    return fwt_sum / max(count, 1)


def compute_adapter_footprint_mb(adapter_dir: str | Path) -> float:
    """Sum of .safetensors file sizes in MB."""
    path = Path(adapter_dir)
    if not path.exists():
        return 0.0
        
    total_bytes = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return total_bytes / (1024 * 1024)


def compute_basis_rank(basis_checkpoint_path: str | Path) -> int:
    """Load basis_checkpoint.npz and return rank (number of basis vectors)."""
    path = Path(basis_checkpoint_path)
    if not path.exists():
        return 0
    try:
        data = np.load(path)
        # assuming 'Q' or 'basis' array inside
        key = list(data.keys())[0]
        q = data[key]
        if q.ndim == 2:
            return q.shape[1]
        return 0
    except Exception:
        return 0


def measure_inference_latency(engine: Any, test_queries: list[str], n_warmup: int = 2) -> dict[str, float]:
    """Measures wall-clock latency for inference queries."""
    if not test_queries:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    # Warmup
    for query in test_queries[:n_warmup]:
        engine.answer(query)
        
    latencies = []
    for query in test_queries[n_warmup:]:
        t0 = time.time()
        engine.answer(query)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)
        
    if not latencies:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        
    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies))
    }


def measure_memory_mb() -> float:
    """Current RSS memory of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def compute_hallucination_rate(eval_results: list[dict]) -> float:
    """Fraction of answers citing nonexistent clauses or attributing rule to wrong regime."""
    if not eval_results:
        return 0.0
        
    hallucinated = 0
    for res in eval_results:
        if res.get("fell_into_trap", False) or not res.get("regime_correct", True):
            hallucinated += 1
            
    return hallucinated / len(eval_results)


def compute_golden_drift(old_golden_path: str | Path, new_golden_path: str | Path) -> float | None:
    """Fraction of pairs whose key conclusion changed between old and new golden set."""
    p_old = Path(old_golden_path)
    p_new = Path(new_golden_path)
    
    if not p_old.exists() or not p_new.exists():
        return None
        
    def load_jsonl(p: Path) -> dict:
        data = {}
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            item = json.loads(line)
            data[item["prompt"]] = item
        return data
        
    old_data = load_jsonl(p_old)
    new_data = load_jsonl(p_new)
    
    common_prompts = set(old_data.keys()) & set(new_data.keys())
    if not common_prompts:
        return None
        
    drift_count = 0
    for prompt in common_prompts:
        # Simplified drift check: exact match of response. 
        # In real scenario, use self_consistency_filter logic.
        if old_data[prompt].get("response") != new_data[prompt].get("response"):
            drift_count += 1
            
    return drift_count / len(common_prompts)
