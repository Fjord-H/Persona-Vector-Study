"""Method 3 — neutral-origin (primary candidate, Design 1).

Origin = mean(neutral_set_300.csv activations) on the TRAIN split ONLY. No harmful
example is ever used to construct the origin — that is the entire point of this method
relative to Method 2, and it is enforced structurally here: `fit_neutral_origin` takes
only a neutral-class matrix as input, there is no code path by which a harmful example
can reach it.

Two scoring modes, both named in the spec ("cosine distance from origin, or projection
onto the origin-to-harmful-mean direction computed separately for validation, not for
vector construction"):

- `score_by_origin_distance`: unsigned cosine distance from the origin. Direction-
  agnostic (doesn't know which way is "harmful").
- `score_by_validation_direction`: a direction from origin -> validation-split harmful
  mean, used only to score/classify, never fed back into the origin. This is the
  harmful set being used "only to validate that harmful prompts land consistently
  off-origin," not to build the vector — the validation-direction vector is discarded
  after scoring, it never touches `origin`.

BOTH modes need a validation-fit decision threshold, not just the first. It's tempting
to assume `score_by_validation_direction` can use a fixed threshold of 0 the way
content_pole's midpoint-projection does (see mean_diff.py) — Method 2's threshold of 0
is principled there because its reference point is a true midpoint between two class
means. Method 3's reference point (`origin`) is NOT a midpoint; it IS the negative
(neutral) class's own train mean. Scoring relative to a threshold of 0 at that point
puts the neutral class's own true center almost exactly on the decision boundary, so
roughly half of neutral test examples get misclassified by noise alone — this was
caught empirically by tests/test_eval_pipeline_synthetic.py (a strong synthetic signal
still produced ~64% accuracy until this was fixed) before it could reach real data.
Both scoring modes here are therefore paired with the same validation-fit threshold
helper, `fit_threshold`.

Both modes are implemented and reported; the spec explicitly leaves the choice between
them open ("do not let Method 3's implementation quietly become Method 2 with extra
steps" — reporting both, clearly labeled, is the way to keep that distinction visible
rather than picking one silently).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.methods.mean_diff import cosine_similarity, project


@dataclass(frozen=True)
class NeutralOriginModel:
    origin: np.ndarray  # mean of TRAIN-split neutral activations only


def fit_neutral_origin(neutral_train_matrix: np.ndarray) -> NeutralOriginModel:
    if neutral_train_matrix.ndim != 2:
        raise ValueError("fit_neutral_origin expects a 2D [n, hidden] matrix")
    return NeutralOriginModel(origin=neutral_train_matrix.mean(axis=0))


def score_by_origin_distance(model: NeutralOriginModel, activations: np.ndarray) -> np.ndarray:
    """1 - cosine_similarity(activation, origin): larger => further from the neutral origin."""
    return 1.0 - cosine_similarity(activations, model.origin)


def fit_validation_direction(model: NeutralOriginModel, harmful_val_matrix: np.ndarray) -> np.ndarray:
    """Direction from the (train-only) origin to the validation-split harmful mean.
    Used only for scoring; never assigned back onto `model.origin`."""
    harmful_val_mean = harmful_val_matrix.mean(axis=0)
    raw_direction = harmful_val_mean - model.origin
    norm = np.linalg.norm(raw_direction)
    if norm == 0:
        raise ValueError("fit_validation_direction: validation harmful mean equals the origin")
    return raw_direction / norm


def score_by_validation_direction(model: NeutralOriginModel, direction: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return project(activations, model.origin, direction)


def fit_threshold(scores_val: np.ndarray, labels_val: np.ndarray) -> float:
    """Grid search over observed validation scores for the accuracy-maximizing
    threshold (labels_val: array of "harmful"/"neutral"). Fit on validation only, per
    spec requirement 2 — never called with test data anywhere in this codebase. Shared
    by both scoring modes above; neither has a threshold that can be assumed a priori.
    """
    is_harmful = np.asarray(labels_val) == "harmful"
    candidates = np.unique(scores_val)
    best_threshold, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = ((scores_val > t) == is_harmful).mean()
        if acc > best_acc:
            best_acc, best_threshold = acc, t
    return float(best_threshold)


def predict_with_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores > threshold, "harmful", "neutral")
