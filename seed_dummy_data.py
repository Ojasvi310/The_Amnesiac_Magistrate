import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from src.audit.registry import AdapterRegistry
from src.eval.report import RunReport, save_report

db_path = "continual_counsel.db"
registry = AdapterRegistry(db_path)

# 1. Create a mock bundle
bundle_dir = Path("exports/dummy_bundle")
bundle_dir.mkdir(parents=True, exist_ok=True)

manifest = {
    "regime": "regime_q1",
    "timestamp": "2024-01-15T10:00:00Z",
    "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "commit_hash": "abc123def",
    "python_version": "3.11",
}
import hashlib
manifest_json = json.dumps(manifest)
(bundle_dir / "manifest.json").write_text(manifest_json)
(bundle_dir / "manifest.sha256").write_text(hashlib.sha256(manifest_json.encode()).hexdigest())

registry.register(
    bundle_dir=bundle_dir,
    manifest=manifest,
    adapter_version_hash="sha256:dummy_adapter_v1",
    training_data_hash="sha256:dummy_data_hash",
    hyperparams={"r": 16, "lora_alpha": 32},
    benchmark_scores={"regime_q1": 0.85, "bwt": 0.0},
    merge_lineage=[],
    index_hash="sha256:dummy_index"
)

# 2. Create a fake evaluation report
dummy_report = RunReport(
    regime="regime_q1",
    timestamp="2024-01-15T10:00:00",
    accuracy={"regime_q1": 0.85, "regime_q2": 0.0},
    bwt=0.0,
    fwt=0.0,
    footprint_mb=45.5,
    basis_rank=16,
    hallucination_rate=0.05,
    confusion_score=0.9,
    latency_ms_p50=125.0,
    golden_drift=0.01
)
runs_dir = Path("runs/regime_q1_2024-01-15T10-00-00")
runs_dir.mkdir(parents=True, exist_ok=True)
save_report(dummy_report, runs_dir)

# 3. Add a fake audit log query
from src.audit.log import AuditLog
audit = AuditLog(db_path)
audit.append(
    query_id=str(uuid.uuid4()),
    query_text="What is the data retention policy?",
    retrieved_chunk_ids=["chunk_1", "chunk_2"],
    regime_tags=["regime_q1"],
    index_hash="sha256:dummy_index",
    adapter_version_hash="sha256:dummy_adapter_v1",
    confidence_scores=[0.89, 0.75],
    verification_verdict="verified",
    final_answer="The policy requires 7 years of retention."
)

print("Dummy data seeded successfully!")
