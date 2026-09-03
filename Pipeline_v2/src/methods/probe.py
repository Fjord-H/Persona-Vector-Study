"""Logistic-regression probe on activations — the second open method question in the
spec: "Cosine-to-class-mean vs. logistic regression probe. Run a probe as a second arm
alongside the mean-diff/cosine approach for whichever method wins the A/B test above.
Comparing them is itself a finding worth reporting, not a decision to make silently."

Trained on the TRAIN split's activation matrix, evaluated on test — same split
discipline as every other method in this project. Which method's activations feed this
probe (content-pole's train matrices, or neutral-origin's) is a choice made by the
caller in eval/subtest_a.py once the A/B-gap winner is known; this module is
method-agnostic, it just fits/scores a probe on whatever [n, hidden] matrix it's given.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_and_score_probe(
    train_matrix: np.ndarray, train_labels: np.ndarray, test_matrix: np.ndarray, test_labels: np.ndarray,
    seed: int = 42,
) -> dict:
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_matrix, train_labels)
    predictions = clf.predict(test_matrix)
    return {
        "accuracy": float((predictions == np.asarray(test_labels)).mean()),
        "predictions": predictions,
        "n_train": len(train_matrix),
        "n_test": len(test_matrix),
    }
