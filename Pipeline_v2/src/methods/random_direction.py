"""Null control for the vector methods — spec requirement 4: "ideally a random-
direction control for the vector methods themselves." A method that doesn't clearly
beat a random direction through the same midpoint isn't demonstrating anything about
content or tone; it's demonstrating that projecting onto *some* direction in a
high-dimensional space picks up class structure by chance.

Same scoring geometry as mean_diff.py (a midpoint + a unit direction) so it is
evaluated identically to Methods 1/2/3, with the direction replaced by a random unit
vector instead of a learned one. The midpoint is the training data's own class
midpoint (reusing whatever midpoint the method being controlled for used) so this
isolates the effect of a random DIRECTION specifically, not a random reference point.
"""
from __future__ import annotations

import numpy as np

from src.methods.mean_diff import project


def random_unit_direction(hidden_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=hidden_dim)
    return vec / np.linalg.norm(vec)


def score(midpoint: np.ndarray, direction: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return project(activations, midpoint, direction)


def predict(midpoint: np.ndarray, direction: np.ndarray, activations: np.ndarray) -> np.ndarray:
    scores = score(midpoint, direction, activations)
    return np.where(scores > 0, "harmful", "neutral")
