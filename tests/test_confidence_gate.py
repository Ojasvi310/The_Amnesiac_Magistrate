"""Tests for the retrieval confidence gate in online/confidence_gate.py."""
from __future__ import annotations

import pytest


try:
    from src.online.confidence_gate import (
        InsufficientGrounding,
        check_retrieval_confidence,
        escalation_response,
    )
except ImportError:
    pytest.skip("src.online.confidence_gate not importable", allow_module_level=True)


def _make_results(top_score: float):
    """Make a fake top-k result list with the given max score."""
    return [
        ({"text": "Section 2.1...", "regime": "regime_q1"}, top_score),
        ({"text": "Section 3.1...", "regime": "regime_q1"}, top_score * 0.8),
    ]


class TestConfidenceGate:
    def test_gate_raises_below_threshold(self):
        """Score 0.3, threshold 0.45 → InsufficientGrounding is raised."""
        results = _make_results(0.3)
        with pytest.raises(InsufficientGrounding):
            check_retrieval_confidence(results, threshold=0.45)

    def test_gate_passes_above_threshold(self):
        """Score 0.6, threshold 0.45 → returns results unchanged."""
        results = _make_results(0.6)
        returned = check_retrieval_confidence(results, threshold=0.45)
        assert returned is results or returned == results

    def test_gate_at_exact_threshold_passes(self):
        """Score exactly at threshold → passes (threshold is exclusive lower bound)."""
        results = _make_results(0.45)
        # Should NOT raise
        returned = check_retrieval_confidence(results, threshold=0.45)
        assert returned is not None

    def test_gate_raises_on_empty_results(self):
        """Empty result list → InsufficientGrounding (no evidence at all)."""
        with pytest.raises(InsufficientGrounding):
            check_retrieval_confidence([], threshold=0.45)

    def test_escalation_response_structure(self):
        """escalation_response returns dict with required keys and correct status."""
        resp = escalation_response("What is the penalty for late notification?")
        assert resp["status"] == "escalate"
        assert "message" in resp
        assert "query" in resp
        assert "Insufficient" in resp["message"] or "escalate" in resp["message"].lower()

    def test_gate_is_not_just_a_warning(self):
        """Confirm the gate raises an exception, not just prints/logs."""
        results = _make_results(0.1)
        raised = False
        try:
            check_retrieval_confidence(results, threshold=0.45)
        except InsufficientGrounding:
            raised = True
        except Exception as e:
            pytest.fail(f"Expected InsufficientGrounding, got {type(e).__name__}: {e}")
        assert raised, "Gate must raise InsufficientGrounding, not just log a warning."
