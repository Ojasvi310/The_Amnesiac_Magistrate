"""Tests for DPO option (a) re-validation path in dpo_polish.py.

Option (a): after DPO, re-run the Step D benchmark gate. If BWT degrades past
threshold, restore the pre-DPO model state (saved before the DPO pass) and
return accepted=False. These tests verify the restore path works correctly.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


try:
    from src.offline.dpo_polish import dpo_with_revalidation
except ImportError:
    pytest.skip("src.offline.dpo_polish not importable", allow_module_level=True)


class TinyModel(nn.Module):
    """Minimal model for testing state dict save/restore."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=False)


class TestDPORevalidation:
    def test_accepts_good_result(self):
        """If validate_fn returns good BWT, model is updated and accepted=True."""
        model = TinyModel()
        original_weights = model.layer.weight.data.clone()

        def fake_dpo_pass(m, tok, pairs, cfg):
            # Simulate DPO modifying model weights
            with torch.no_grad():
                m.layer.weight.data += 0.01
            return m

        def good_validate_fn(m):
            return {"regime_q1": 0.85, "regime_q2": 0.80, "bwt": 0.00}

        tokenizer = MagicMock()
        contrastive_pairs = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
        dpo_cfg = MagicMock()
        dpo_cfg.beta = 0.1

        with patch("src.offline.dpo_polish.run_dpo_pass", side_effect=fake_dpo_pass):
            result_model, accepted = dpo_with_revalidation(
                model, tokenizer, contrastive_pairs, dpo_cfg,
                validate_fn=good_validate_fn,
            )

        assert accepted is True
        # Weights should be updated (DPO was applied)
        assert not torch.allclose(result_model.layer.weight.data, original_weights), \
            "Model should have updated weights when DPO is accepted."

    def test_rejects_bad_bwt_and_restores(self):
        """If validate_fn returns BWT regression, pre-DPO model is restored and accepted=False."""
        model = TinyModel()
        original_weights = model.layer.weight.data.clone()

        def fake_dpo_pass(m, tok, pairs, cfg):
            with torch.no_grad():
                m.layer.weight.data += 0.5  # large modification
            return m

        def bad_validate_fn(m):
            # Simulate severe forgetting
            return {"regime_q1": 0.60, "regime_q2": 0.50, "bwt": -0.20, "passed": False}

        tokenizer = MagicMock()
        contrastive_pairs = [{"prompt": "q", "chosen": "a", "rejected": "b"}]
        dpo_cfg = MagicMock()

        with patch("src.offline.dpo_polish.run_dpo_pass", side_effect=fake_dpo_pass):
            result_model, accepted = dpo_with_revalidation(
                model, tokenizer, contrastive_pairs, dpo_cfg,
                validate_fn=bad_validate_fn,
            )

        assert accepted is False
        # Weights must be restored to pre-DPO values
        assert torch.allclose(result_model.layer.weight.data, original_weights, atol=1e-6), \
            "Model weights must be restored to pre-DPO values when BWT gate fails."

    def test_restores_exact_state_dict(self):
        """Verify layer-by-layer that the restored model exactly matches pre-DPO state."""
        model = TinyModel()
        pre_dpo_sd = {k: v.clone() for k, v in model.state_dict().items()}

        def fake_dpo_pass(m, tok, pairs, cfg):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.fill_(999.0)  # obviously wrong values
            return m

        def bad_validate_fn(m):
            return {"regime_q1": 0.50, "bwt": -0.30, "passed": False}

        tokenizer = MagicMock()
        dpo_cfg = MagicMock()

        with patch("src.offline.dpo_polish.run_dpo_pass", side_effect=fake_dpo_pass):
            result_model, accepted = dpo_with_revalidation(
                model, tokenizer, [], dpo_cfg,
                validate_fn=bad_validate_fn,
            )

        assert accepted is False
        post_sd = result_model.state_dict()
        for key in pre_dpo_sd:
            assert torch.allclose(post_sd[key], pre_dpo_sd[key], atol=1e-7), \
                f"Parameter '{key}' was not correctly restored."
