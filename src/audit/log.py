"""
Tamper-evident append-only audit log stored in SQLite.

Each record contains a SHA-256 hash of its own fields concatenated with the
hash of the previous record (like a simplistic hash chain). Any post-hoc
modification to a record's data will break verify_chain().
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class _Base(DeclarativeBase):
    pass


class AuditRecord(_Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String, unique=True, nullable=False)
    query_text = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(Text, nullable=False)    # JSON list[str]
    regime_tags = Column(Text, nullable=False)            # JSON list[str]
    index_hash = Column(String, nullable=False)
    adapter_version_hash = Column(String, nullable=False)
    confidence_scores = Column(Text, nullable=False)      # JSON list[float]
    verification_verdict = Column(String, nullable=False) # "verified"|"flagged"|"escalated"
    final_answer = Column(Text, nullable=False)
    timestamp = Column(String, nullable=False)            # ISO UTC
    record_hash = Column(String, nullable=False)
    prev_record_hash = Column(String, nullable=True)


def _compute_record_hash(
    query_id: str,
    query_text: str,
    retrieved_chunk_ids: str,
    regime_tags: str,
    index_hash: str,
    adapter_version_hash: str,
    confidence_scores: str,
    verification_verdict: str,
    final_answer: str,
    timestamp: str,
    prev_record_hash: Optional[str],
) -> str:
    """
    Produce a deterministic SHA-256 over all mutable fields so that tampering
    with any single field invalidates the hash.
    """
    payload = "\x00".join([
        query_id,
        query_text,
        retrieved_chunk_ids,
        regime_tags,
        index_hash,
        adapter_version_hash,
        confidence_scores,
        verification_verdict,
        final_answer,
        timestamp,
        prev_record_hash or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, db_path: str) -> None:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        _Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    def _last_record_hash(self, session: Session) -> Optional[str]:
        """Return the record_hash of the most recently inserted row, or None."""
        last = (
            session.query(AuditRecord)
            .order_by(AuditRecord.id.desc())
            .first()
        )
        return last.record_hash if last else None

    def append(
        self,
        query_id: str,
        query_text: str,
        retrieved_chunk_ids: list[str],
        regime_tags: list[str],
        index_hash: str,
        adapter_version_hash: str,
        confidence_scores: list[float],
        verification_verdict: str,
        final_answer: str,
    ) -> AuditRecord:
        """
        Append a new audit entry, chaining its hash from the previous record.
        The timestamp is set to UTC now at insertion time.
        """
        ts = datetime.now(tz=timezone.utc).isoformat()

        chunk_ids_json = json.dumps(retrieved_chunk_ids)
        regime_tags_json = json.dumps(regime_tags)
        scores_json = json.dumps([float(s) for s in confidence_scores])

        with self._Session() as session:
            prev_hash = self._last_record_hash(session)

            record_hash = _compute_record_hash(
                query_id=query_id,
                query_text=query_text,
                retrieved_chunk_ids=chunk_ids_json,
                regime_tags=regime_tags_json,
                index_hash=index_hash,
                adapter_version_hash=adapter_version_hash,
                confidence_scores=scores_json,
                verification_verdict=verification_verdict,
                final_answer=final_answer,
                timestamp=ts,
                prev_record_hash=prev_hash,
            )

            record = AuditRecord(
                query_id=query_id,
                query_text=query_text,
                retrieved_chunk_ids=chunk_ids_json,
                regime_tags=regime_tags_json,
                index_hash=index_hash,
                adapter_version_hash=adapter_version_hash,
                confidence_scores=scores_json,
                verification_verdict=verification_verdict,
                final_answer=final_answer,
                timestamp=ts,
                record_hash=record_hash,
                prev_record_hash=prev_hash,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_record(self, query_id: str) -> Optional[AuditRecord]:
        """Return the AuditRecord for query_id, or None if not found."""
        with self._Session() as session:
            return (
                session.query(AuditRecord)
                .filter_by(query_id=query_id)
                .first()
            )

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """
        Walk all records in insertion order and recompute each record_hash.
        Returns (True, None) if the chain is intact; (False, first_broken_id)
        if any record has been tampered with.

        This is O(n) in the number of audit records.
        """
        with self._Session() as session:
            records = (
                session.query(AuditRecord)
                .order_by(AuditRecord.id.asc())
                .all()
            )

        prev_hash: Optional[str] = None

        for record in records:
            expected = _compute_record_hash(
                query_id=record.query_id,
                query_text=record.query_text,
                retrieved_chunk_ids=record.retrieved_chunk_ids,
                regime_tags=record.regime_tags,
                index_hash=record.index_hash,
                adapter_version_hash=record.adapter_version_hash,
                confidence_scores=record.confidence_scores,
                verification_verdict=record.verification_verdict,
                final_answer=record.final_answer,
                timestamp=record.timestamp,
                prev_record_hash=prev_hash,
            )

            if record.record_hash != expected:
                return False, record.id

            # The stored prev_record_hash must match what we tracked.
            if record.prev_record_hash != prev_hash:
                return False, record.id

            prev_hash = record.record_hash

        return True, None
