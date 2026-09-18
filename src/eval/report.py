"""Report generation for evaluation results."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.eval.metrics import (
    compute_accuracy,
    compute_bwt,
    compute_fwt,
    compute_adapter_footprint_mb,
    compute_basis_rank,
)


@dataclass
class RunReport:
    regime: str
    timestamp: str
    accuracy: dict[str, float]
    bwt: float
    fwt: float
    footprint_mb: float
    basis_rank: int
    hallucination_rate: float
    confusion_score: float
    latency_ms_p50: float
    golden_drift: float | None


def generate_report(
    regime: str, 
    timestamp: str, 
    runs_dir: str | Path, 
    data_root: str | Path
) -> RunReport:
    """Generate report from a specific run."""
    run_path = Path(runs_dir) / f"{regime}_{timestamp}"
    
    # Defaults
    acc = {}
    bwt = 0.0
    fwt = 0.0
    footprint_mb = 0.0
    basis_rank = 0
    hallucination_rate = 0.0
    confusion_score = 0.0
    latency_ms_p50 = 0.0
    golden_drift = None
    
    # Load validation scores
    val_scores_path = run_path / "validation_scores.json"
    if val_scores_path.exists():
        raw_scores = json.loads(val_scores_path.read_text())
        acc = {k: v.get("accuracy", 0.0) if isinstance(v, dict) else v for k, v in raw_scores.items() if k != "bwt"}
        # Compute BWT
        # Without historical context here, we rely on the pre-computed BWT in the run if available
        bwt = raw_scores.get("bwt", 0.0)

    # Footprint
    adapter_dir = run_path / "adapter"
    if adapter_dir.exists():
        footprint_mb = compute_adapter_footprint_mb(adapter_dir)
        
    # Basis rank
    basis_path = run_path / "basis_checkpoint.npz"
    if basis_path.exists():
        basis_rank = compute_basis_rank(basis_path)
        
    return RunReport(
        regime=regime,
        timestamp=timestamp,
        accuracy=acc,
        bwt=bwt,
        fwt=fwt,
        footprint_mb=footprint_mb,
        basis_rank=basis_rank,
        hallucination_rate=hallucination_rate,
        confusion_score=confusion_score,
        latency_ms_p50=latency_ms_p50,
        golden_drift=golden_drift,
    )


def save_report(report: RunReport, reports_dir: str | Path) -> None:
    """Save report to disk."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(asdict(report), indent=2))


def load_all_reports(runs_dir: str | Path) -> list[RunReport]:
    """Load all run reports, sorted by timestamp."""
    runs_dir = Path(runs_dir)
    reports = []
    
    if not runs_dir.exists():
        return reports
        
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            report_path = run_dir / "report.json"
            if report_path.exists():
                try:
                    data = json.loads(report_path.read_text())
                    reports.append(RunReport(**data))
                except Exception:
                    pass
                    
    reports.sort(key=lambda r: r.timestamp)
    return reports


def reports_to_dataframe(reports: list[RunReport]) -> Any:
    """Convert a list of reports into a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        return None
        
    data = []
    for r in reports:
        row = asdict(r)
        # Flatten accuracy
        for regime, acc in r.accuracy.items():
            row[f"acc_{regime}"] = acc
        del row["accuracy"]
        data.append(row)
        
    return pd.DataFrame(data)
