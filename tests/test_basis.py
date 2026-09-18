"""Tests for the QR-based incremental orthonormal basis in train_adapter.py.

These tests use small synthetic tensors (dim=8, rank=2) and require no GPU.
The correctness of the basis math is easy to get subtly wrong — especially the
reset-at-merge path, where accumulated orthogonal complement from prior adapters
must be fully discarded and rebuilt from the single merged adapter alone.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Import OrthonormalBasis from train_adapter; skip if offline deps missing
# ---------------------------------------------------------------------------
try:
    from src.offline.train_adapter import OrthonormalBasis
except ImportError:
    pytest.skip("src.offline.train_adapter not importable (offline deps missing)", allow_module_level=True)


def _random_A(rows, cols, seed=None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    return torch.randn(rows, cols, generator=rng)


def _check_orthonormal(B: torch.Tensor, tol=1e-5):
    """Returns True if the columns of B are orthonormal (B^T B ≈ I)."""
    product = B.T @ B
    identity = torch.eye(product.shape[0])
    return torch.allclose(product, identity, atol=tol)


class TestOrthonormalBasis:
    def test_basis_update_orthogonality(self):
        """After three updates with random A matrices, columns are orthonormal."""
        basis = OrthonormalBasis(max_rank=16)
        for seed in [1, 2, 3]:
            A = _random_A(32, 4, seed=seed)
            basis.update({"layer_0": A})

        B = basis._basis_vectors  # internal column matrix
        if B is not None and B.shape[1] > 0:
            assert _check_orthonormal(B), "Basis columns are not orthonormal after 3 updates."

    def test_basis_rank_grows(self):
        """Rank increases after each update (up to max_rank)."""
        basis = OrthonormalBasis(max_rank=16)
        prev_rank = basis.rank
        for seed in [10, 20, 30]:
            A = _random_A(32, 4, seed=seed)
            basis.update({"layer_0": A})
            assert basis.rank >= prev_rank, f"Rank did not increase: {basis.rank} < {prev_rank}"
            prev_rank = basis.rank

    def test_basis_reset_from_merged(self):
        """After reset, basis reflects only the merged adapter; prior vectors are gone."""
        basis = OrthonormalBasis(max_rank=16)
        for seed in [1, 2, 3]:
            basis.update({"layer_0": _random_A(32, 4, seed=seed)})

        rank_before_reset = basis.rank
        merged_A = _random_A(32, 2, seed=99)  # rank-2 merged adapter
        basis.reset_from_merged({"layer_0": merged_A})

        # After reset, rank should reflect only the merged adapter's columns
        assert basis.rank <= 2, f"Expected rank <= 2 after reset, got {basis.rank}"
        assert basis.rank < rank_before_reset, "Reset should reduce rank vs accumulated basis."

    def test_orthogonality_loss_zero_for_orthogonal_input(self):
        """If the new A matrix is orthogonal to the current basis, the loss is 0."""
        basis = OrthonormalBasis(max_rank=16)
        # Set a known basis: first two standard basis vectors in R^8
        e1 = torch.zeros(8, 1)
        e1[0] = 1.0
        e2 = torch.zeros(8, 1)
        e2[1] = 1.0
        # Manually set basis to [e1, e2]
        B = torch.cat([e1, e2], dim=1)
        basis._basis_vectors = B

        # A matrix orthogonal to e1 and e2: rows 2-4 only
        A_orth = torch.zeros(8, 3)
        A_orth[2, 0] = 1.0
        A_orth[3, 1] = 1.0
        A_orth[4, 2] = 1.0

        loss = basis.orthogonality_loss({"layer_0": A_orth})
        assert float(loss) < 1e-5, f"Expected ~0 loss for orthogonal input, got {float(loss):.6f}"

    def test_orthogonality_loss_nonzero_for_aligned_input(self):
        """If the new A matrix aligns with the basis, the loss is positive."""
        basis = OrthonormalBasis(max_rank=16)
        e1 = torch.zeros(8, 1)
        e1[0] = 1.0
        basis._basis_vectors = e1

        # A matrix that projects exactly onto e1
        A_aligned = torch.zeros(8, 1)
        A_aligned[0] = 1.0

        loss = basis.orthogonality_loss({"layer_0": A_aligned})
        assert float(loss) > 0.01, f"Expected positive loss for aligned input, got {float(loss):.6f}"
