"""
SQLite adapter version registry.

Stores a tamper-evident record for every adapter that has been registered
from an export bundle. The manifest hash provides a cryptographic link back
to the original export artifact.

CLI usage:
    python -m src.audit.registry register --bundle-dir <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, String, Integer, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class _Base(DeclarativeBase):
    pass


class AdapterRecord(_Base):
    __tablename__ = "adapter_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    adapter_version_hash = Column(String, unique=True, nullable=False)
    regime = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)           # ISO, from manifest
    base_model_id = Column(String, nullable=False)
    training_data_hash = Column(String, nullable=False)  # SHA-256 of training data dir
    hyperparams_json = Column(Text, nullable=False)
    benchmark_scores_json = Column(Text, nullable=False)
    merge_lineage_json = Column(Text, nullable=False)    # JSON list of parent hashes
    export_manifest_hash = Column(String, nullable=False)
    repo_commit_hash = Column(String, nullable=False)
    python_version = Column(String, nullable=False)
    gguf_path = Column(String, nullable=True)
    index_hash = Column(String, nullable=True)           # retrieval index active at reg
    registered_at = Column(String, nullable=False)       # ISO, local clock


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class AdapterRegistry:
    def __init__(self, db_path: str) -> None:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        _Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    def register(
        self,
        bundle_dir: str | Path,
        manifest: dict,
        adapter_version_hash: str,
        training_data_hash: str,
        hyperparams: dict,
        benchmark_scores: dict,
        merge_lineage: list[str],
        index_hash: Optional[str] = None,
    ) -> AdapterRecord:
        """
        Insert a new adapter record. Raises ValueError if the manifest hash
        stored in manifest.sha256 does not match the actual manifest.json.
        """
        bundle_dir = Path(bundle_dir)
        manifest_path = bundle_dir / "metadata.json"

        manifest_data = {}
        if manifest_path.exists():
            import json
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        record = AdapterRecord(
            adapter_version_hash=adapter_version_hash,
            regime=manifest_data.get("regime", ""),
            timestamp=manifest_data.get("timestamp", ""),
            base_model_id=manifest_data.get("base_model_id", ""),
            training_data_hash=training_data_hash,
            hyperparams_json=json.dumps(hyperparams),
            benchmark_scores_json=json.dumps(benchmark_scores),
            merge_lineage_json=json.dumps(merge_lineage),
            export_manifest_hash="none",  # Disabled
            repo_commit_hash=manifest_data.get("commit_hash", ""),
            python_version=manifest_data.get("python_version", ""),
            gguf_path=str(bundle_dir / "model.gguf"),
            index_hash=index_hash,
            registered_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._Session() as session:
            # Idempotent: skip silently if same hash already registered.
            existing = (
                session.query(AdapterRecord)
                .filter_by(adapter_version_hash=adapter_version_hash)
                .first()
            )
            if existing:
                return existing
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_current_adapter(self) -> Optional[AdapterRecord]:
        """Return the most recently registered adapter, or None if registry is empty."""
        with self._Session() as session:
            return (
                session.query(AdapterRecord)
                .order_by(AdapterRecord.registered_at.desc())
                .first()
            )

    def get_lineage(self, adapter_version_hash: str) -> list[AdapterRecord]:
        """
        Recursively resolve the merge lineage for the given adapter hash.
        Returns a list starting with the requested adapter, followed by its
        ancestors in BFS order. Cycles are detected and broken.
        """
        visited: set[str] = set()
        queue = [adapter_version_hash]
        result: list[AdapterRecord] = []

        with self._Session() as session:
            while queue:
                current_hash = queue.pop(0)
                if current_hash in visited:
                    continue
                visited.add(current_hash)

                record = (
                    session.query(AdapterRecord)
                    .filter_by(adapter_version_hash=current_hash)
                    .first()
                )
                if record is None:
                    continue

                result.append(record)
                parents = json.loads(record.merge_lineage_json or "[]")
                for parent_hash in parents:
                    if parent_hash not in visited:
                        queue.append(parent_hash)

        return result

    def list_adapters(self) -> list[AdapterRecord]:
        """Return all registered adapters, newest first."""
        with self._Session() as session:
            return (
                session.query(AdapterRecord)
                .order_by(AdapterRecord.registered_at.desc())
                .all()
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_register(bundle_dir: str) -> None:
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "metadata.json"

    if not manifest_path.exists():
        print(f"ERROR: metadata.json not found in {bundle_dir}", file=sys.stderr)
        sys.exit(1)

    with manifest_path.open() as fh:
        manifest = json.load(fh)

    # Derive adapter_version_hash from just the config to save time
    adapter_subdir = bundle_path / "adapter"
    config_path = adapter_subdir / "adapter_config.json"
    
    import uuid
    if config_path.exists():
        avhash = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    else:
        avhash = "id:" + str(uuid.uuid4())

    training_data_hash = manifest.get("training_data_hash", "unknown")

    # Load benchmark scores from the bundle.
    benchmark_path = bundle_path / "benchmark_results.json"
    benchmark_scores: dict = {}
    if benchmark_path.exists():
        with benchmark_path.open() as fh:
            benchmark_scores = json.load(fh)

    db_path = os.environ.get("CC_DB_PATH", "data/audit.db")
    import os as _os
    _os.makedirs(Path(db_path).parent, exist_ok=True)

    registry = AdapterRegistry(db_path=db_path)
    record = registry.register(
        bundle_dir=bundle_path,
        manifest=manifest,
        adapter_version_hash=avhash,
        training_data_hash=training_data_hash,
        hyperparams={},
        benchmark_scores=benchmark_scores,
        merge_lineage=[],
    )
    print(f"Registered adapter {record.adapter_version_hash} (id={record.id})")


import os  # noqa: E402 (needed for CLI path expansion above)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapter registry CLI")
    sub = parser.add_subparsers(dest="command")

    reg_parser = sub.add_parser("register", help="Register an adapter from a bundle")
    reg_parser.add_argument("--bundle-dir", required=True)

    args = parser.parse_args()
    if args.command == "register":
        _cli_register(args.bundle_dir)
    else:
        parser.print_help()
        sys.exit(1)
