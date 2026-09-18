"""Tests for the replay grounding filter and self-consistency filter in replay_gen.py.

These tests isolate the filter logic from the GPU-dependent golden capture function,
using mocked FAISS queries and synthetic text to verify the filtering logic.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


try:
    from src.offline.replay_gen import (
        grounding_filter,
        self_consistency_filter,
    )
except ImportError:
    pytest.skip("src.offline.replay_gen not importable (may require offline deps)", allow_module_level=True)


class TestGroundingFilter:
    def test_rejects_ungrounded_answer(self):
        """Answer whose key terms appear in no retrieved chunk is discarded."""
        answer = "The penalty is 500 years in prison under Section 99."
        chunks = [
            {"text": "Section 2.1 requires a 72-hour notification window.", "regime": "regime_q1"},
            {"text": "Financial records must be retained for 7 years.", "regime": "regime_q1"},
        ]
        # "500 years in prison" and "Section 99" appear in no chunk
        result = grounding_filter(answer, chunks, key_terms=["500 years", "Section 99", "prison"])
        assert result is False, "Ungrounded answer should be rejected."

    def test_keeps_grounded_answer(self):
        """Answer whose key terms appear in retrieved chunks is kept."""
        answer = "Under Section 2.1, notification must occur within 72 hours."
        chunks = [
            {"text": "Section 2.1 requires a 72-hour notification window.", "regime": "regime_q1"},
        ]
        result = grounding_filter(answer, chunks, key_terms=["72-hour", "Section 2.1"])
        assert result is True, "Grounded answer should be kept."

    def test_partial_match_sufficient(self):
        """If at least one key term appears in at least one chunk, it passes."""
        answer = "The 72-hour clock begins at confirmed discovery under Section 2.1."
        chunks = [
            {"text": "The 7-year retention period applies to financial records.", "regime": "regime_q1"},
            {"text": "72-hour notification is required upon breach discovery.", "regime": "regime_q1"},
        ]
        result = grounding_filter(answer, chunks, key_terms=["72-hour"])
        assert result is True


class TestSelfConsistencyFilter:
    def test_rejects_disagreeing_responses(self):
        """Two responses with conflicting key terms are rejected."""
        response_a = "The penalty is 2% of annual turnover."
        response_b = "The penalty is 5% of annual global turnover."
        # Key conclusion terms differ: "2%" vs "5%"
        result = self_consistency_filter(response_a, response_b, key_terms=["2%", "5%"])
        assert result is False, "Disagreeing responses should be rejected."

    def test_keeps_agreeing_responses(self):
        """Two responses agreeing on key terms are kept."""
        response_a = "Notification must be sent within 72 hours of discovery."
        response_b = "The data controller has 72 hours from discovery to notify the authority."
        result = self_consistency_filter(response_a, response_b, key_terms=["72"])
        assert result is True, "Agreeing responses should be kept."

    def test_both_must_contain_term_for_agreement(self):
        """Term present in only one response → not considered an agreement."""
        response_a = "The penalty is 2% of annual turnover."
        response_b = "Penalties may apply for late notification."
        # "2%" only in response_a
        result = self_consistency_filter(response_a, response_b, key_terms=["2%"])
        assert result is False, "Term in only one response should not count as agreement."
