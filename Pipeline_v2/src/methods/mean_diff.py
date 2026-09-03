"""Shared mean-difference / projection-onto-midpoint machinery, used by
content_pole.py (Method 2), tone_pole.py (Method 1's scoring step), and
neutral_origin.py's validation-direction scorer. Factored out once rather than
reimplemented three times, since all three reduce to the same geometry: two class
means define a direction and a midpoint, and a new point's score is its signed
projection onto that direction relative to the midpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanDiffPole:
    positive_mean: np.ndarray   # e.g. harmful-class mean
    negative_mean: np.ndarray    # e.g. neutral/helpful-class mean
    midpoint: np.ndarray
    direction: np.ndarray          # unit vector, positive_mean - negative_mean


def fit_mean_diff_pole(positive_matrix: np.ndarray, negative_matrix: np.ndarray) -> MeanDiffPole:
    if positive_matrix.ndim != 2 or negative_matrix.ndim != 2:
        raise ValueError("fit_mean_diff_pole expects 2D [n, hidden] matrices")
    positive_mean = positive_matrix.mean(axis=0)
    negative_mean = negative_matrix.mean(axis=0)
    midpoint = (positive_mean + negative_mean) / 2.0
    raw_direction = positive_mean - negative_mean
    norm = np.linalg.norm(raw_direction)
    if norm == 0:
        raise ValueError("fit_mean_diff_pole: positive and negative class means are identical")
    direction = raw_direction / norm
    return MeanDiffPole(positive_mean=positive_mean, negative_mean=negative_mean,
                         midpoint=midpoint, direction=direction)


def project(activations: np.ndarray, midpoint: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Signed distance of each row of `activations` from `midpoint` along `direction`
    (a unit vector). Positive => on the `positive_mean` side. Threshold is 0 by
    construction — no threshold is fit or selected on any split.
    """
    return (activations - midpoint.reshape(1, -1)) @ direction


def predict_from_pole(pole: MeanDiffPole, activations: np.ndarray) -> np.ndarray:
    """Shared by content_pole.py and tone_pole.py: score > 0 (positive-class side of
    the midpoint) => "harmful", else "neutral"/"helpful" (both methods' negative class
    is the non-harmful pole, so the label string is always this pair)."""
    scores = project(activations, pole.midpoint, pole.direction)
    return np.where(scores > 0, "harmful", "neutral")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between `a` [n, hidden] and a single vector `b`
    [hidden]."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b), 1e-12, None)
    return a_norm @ b_norm
