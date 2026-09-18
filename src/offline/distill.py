"""Feature-level distillation loss for continual-learning training."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def hidden_state_distill_loss(
    teacher_hiddens: list,
    student_hiddens: list,
    layer_indices: Optional[list] = None,
) -> torch.Tensor:
    """MSE loss between teacher and student hidden states at specified transformer layers.

    We match hidden states rather than output logit distributions (KL divergence)
    because a confident model can produce nearly-identical token probabilities while
    its internal representations drift substantially -- the logit surface flattens
    near argmax so KL misses the drift entirely. Hidden-state MSE catches that drift
    directly in representation space, which is what we care about for continual learning.

    Args:
        teacher_hiddens: List of tensors, one per layer, shape (B, T, H).
                         Typically obtained from model(..., output_hidden_states=True).hidden_states.
        student_hiddens: Same structure as teacher_hiddens. Must have the same number of
                         layers and the same hidden dimension H (both are the same base
                         model, different adapter state).
        layer_indices:   Which layers to compute the loss over. 0 = embedding layer,
                         1..N = transformer layers. If None, all layers are used.

    Returns:
        Scalar tensor; mean MSE across selected layers, averaged over batch and sequence.
    """
    if len(teacher_hiddens) != len(student_hiddens):
        raise ValueError(
            "Teacher has {} hidden-state tensors but student has {}. "
            "Both models must share the same architecture.".format(
                len(teacher_hiddens), len(student_hiddens)
            )
        )

    indices = layer_indices if layer_indices is not None else list(range(len(teacher_hiddens)))

    if not indices:
        raise ValueError("layer_indices must be non-empty (or None to use all layers).")

    total_loss = torch.tensor(0.0, device=student_hiddens[0].device, dtype=student_hiddens[0].dtype)

    for idx in indices:
        if idx >= len(teacher_hiddens):
            raise IndexError(
                "layer_index {} is out of range for a model with {} hidden-state tensors.".format(
                    idx, len(teacher_hiddens)
                )
            )
        t = teacher_hiddens[idx].detach()  # teacher grads must not flow
        s = student_hiddens[idx]

        if t.shape != s.shape:
            raise ValueError(
                "Shape mismatch at layer {}: teacher {} vs student {}.".format(
                    idx, t.shape, s.shape
                )
            )

        # Mean over batch, sequence, and hidden dim -- produces a single comparable scalar
        # regardless of the layer hidden size (though all transformer layers share H here).
        total_loss = total_loss + F.mse_loss(s, t, reduction="mean")

    return total_loss / len(indices)
