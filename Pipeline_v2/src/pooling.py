"""Pooling functions that reduce a [batch, seq_len, hidden] hidden-state tensor to
[batch, hidden]. Every function here is attention-mask-aware.

v1's bug (see defect_report.md S2-04/S2-05 and HANDOVER.md): every v1 notebook called
`hidden_states.mean(dim=1)` directly on padded batches with no mask applied, so
pad/EOS-token activations silently entered the pooled vector for every batch containing
more than one sequence length. That is exactly what `masked_mean` below exists to
prevent — padded positions must never contribute to a pooled vector, in any function
in this module.
"""
from __future__ import annotations

import torch


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over real (non-padded) tokens only.

    hidden_states: [batch, seq_len, hidden], any float dtype.
    attention_mask: [batch, seq_len], 1 for real tokens, 0 for padding.
    Returns: [batch, hidden], float32.
    """
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden_states leading dims {tuple(hidden_states.shape[:2])} != "
            f"attention_mask shape {tuple(attention_mask.shape)}"
        )
    hidden_states = hidden_states.to(torch.float32)
    mask = attention_mask.to(torch.float32).unsqueeze(-1)  # [batch, seq_len, 1]
    token_counts = mask.sum(dim=1)  # [batch, 1]
    if torch.any(token_counts == 0):
        raise ValueError("masked_mean: at least one sequence has zero real tokens")
    summed = (hidden_states * mask).sum(dim=1)  # [batch, hidden]
    return summed / token_counts


def last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Activation of the last real (non-padded) token of each sequence.

    Correct under both left- and right-padding: the last real token is found from the
    attention mask directly, not assumed to sit at a fixed offset.

    hidden_states: [batch, seq_len, hidden], any float dtype.
    attention_mask: [batch, seq_len], 1 for real tokens, 0 for padding.
    Returns: [batch, hidden], float32.
    """
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden_states leading dims {tuple(hidden_states.shape[:2])} != "
            f"attention_mask shape {tuple(attention_mask.shape)}"
        )
    hidden_states = hidden_states.to(torch.float32)
    mask = attention_mask.to(torch.long)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("last_token: at least one sequence has zero real tokens")
    seq_len = mask.shape[1]
    positions = torch.arange(seq_len, device=mask.device).unsqueeze(0)  # [1, seq_len]
    # Last real-token index = highest position where mask == 1.
    masked_positions = torch.where(mask.bool(), positions, torch.full_like(positions, -1))
    last_idx = masked_positions.max(dim=1).values  # [batch]
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, last_idx]


def center(vectors: torch.Tensor, reference_mean: torch.Tensor) -> torch.Tensor:
    """Subtract a reference mean (e.g. the neutral-set mean) from each pooled vector.

    vectors: [n, hidden]. reference_mean: [hidden] or [1, hidden].
    """
    return vectors - reference_mean.reshape(1, -1)


def standardize(vectors: torch.Tensor, reference_mean: torch.Tensor, reference_std: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
    """Center and scale by a reference std (e.g. computed on the neutral/train set only).

    vectors: [n, hidden]. reference_mean, reference_std: [hidden] or [1, hidden].
    Per-dimension std of zero (a dead/constant dimension) is floored at eps rather than
    producing inf/NaN.
    """
    mean = reference_mean.reshape(1, -1)
    std = reference_std.reshape(1, -1).clamp(min=eps)
    return (vectors - mean) / std
