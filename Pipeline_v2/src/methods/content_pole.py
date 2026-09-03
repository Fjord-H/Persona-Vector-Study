"""Method 2 — content-pole (v1's approach, done honestly this time).

Vector = mean(harmbench activations) - mean(neutral activations), built on the TRAIN
split only. Score = signed projection onto the midpoint-relative direction (see
mean_diff.py) — equivalent to "position relative to the midpoint between the two class
means," one of the two scoring options the spec names for this method. Threshold is 0
by construction (the midpoint), so no separate threshold selection step is needed or
performed on any split.
"""
from __future__ import annotations

import numpy as np

from src.methods.mean_diff import MeanDiffPole, fit_mean_diff_pole, predict_from_pole, project


def fit_content_pole(harmful_train_matrix: np.ndarray, neutral_train_matrix: np.ndarray) -> MeanDiffPole:
    return fit_mean_diff_pole(positive_matrix=harmful_train_matrix, negative_matrix=neutral_train_matrix)


def score(pole: MeanDiffPole, activations: np.ndarray) -> np.ndarray:
    return project(activations, pole.midpoint, pole.direction)


def predict(pole: MeanDiffPole, activations: np.ndarray) -> np.ndarray:
    """Returns an array of "harmful"/"neutral" predictions (score > 0 => harmful)."""
    return predict_from_pole(pole, activations)
