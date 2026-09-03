"""Pure-tensor unit tests for src/pooling.py — no model required.

These pin down the pooling functions' correctness in isolation (mask handling, both
padding sides) before the integration test in test_pooling_batch_equivalence.py checks
them against a real model's batching behavior.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import pooling  # noqa: E402


def test_masked_mean_ignores_right_padding():
    # batch of 2: seq 0 has 3 real tokens + 2 pad; seq 1 has 5 real tokens.
    hidden = torch.zeros(2, 5, 4)
    hidden[0, :3] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hidden[0, 3:] = 999.0  # padding — must be excluded
    hidden[1, :5] = 2.0
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    out = pooling.masked_mean(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(out[1], torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_masked_mean_ignores_left_padding():
    hidden = torch.zeros(1, 4, 3)
    hidden[0, 0] = 999.0  # left pad — must be excluded
    hidden[0, 1:] = torch.tensor([1.0, 1.0, 1.0])
    mask = torch.tensor([[0, 1, 1, 1]])

    out = pooling.masked_mean(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0, 1.0]))


def test_last_token_right_padding():
    hidden = torch.zeros(1, 4, 2)
    hidden[0, 2] = torch.tensor([5.0, 6.0])  # last real token, index 2
    hidden[0, 3] = 999.0  # padding
    mask = torch.tensor([[1, 1, 1, 0]])

    out = pooling.last_token(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([5.0, 6.0]))


def test_last_token_left_padding():
    hidden = torch.zeros(1, 4, 2)
    hidden[0, 0] = 999.0  # left pad
    hidden[0, 3] = torch.tensor([7.0, 8.0])  # last real token, index 3
    mask = torch.tensor([[0, 1, 1, 1]])

    out = pooling.last_token(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([7.0, 8.0]))


def test_masked_mean_rejects_all_padding_sequence():
    hidden = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        pooling.masked_mean(hidden, mask)


def test_last_token_rejects_all_padding_sequence():
    hidden = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        pooling.last_token(hidden, mask)


def test_center_and_standardize():
    vectors = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    mean = torch.tensor([1.0, 2.0])
    centered = pooling.center(vectors, mean)
    assert torch.allclose(centered, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    std = torch.tensor([1.0, 2.0])
    standardized = pooling.standardize(vectors, mean, std)
    assert torch.allclose(standardized, torch.tensor([[1.0, 1.0], [5.0, 3.0]]))


def test_standardize_floors_zero_std():
    vectors = torch.tensor([[5.0]])
    mean = torch.tensor([0.0])
    std = torch.tensor([0.0])
    out = pooling.standardize(vectors, mean, std, eps=1e-8)
    assert torch.isfinite(out).all()
