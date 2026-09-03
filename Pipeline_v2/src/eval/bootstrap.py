"""Bootstrap confidence intervals — spec requirement 3: "Bootstrap confidence intervals
on every reported number. A layer or method 'winning' by less than its own CI width is
noise (this killed v1's Llama Layer 7 result, 63.86% +/- 6.45)."
"""
from __future__ import annotations

import numpy as np


def bootstrap_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """Percentile bootstrap over the per-example correctness indicator. Returns
    {accuracy, ci_low, ci_high, n, n_bootstrap}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    n = len(y_true)
    if n == 0:
        raise ValueError("bootstrap_accuracy_ci: empty input")

    correct = (y_true == y_pred).astype(np.float64)
    point_estimate = correct.mean()

    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = correct[idx].mean()

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(resampled_means, [alpha, 1.0 - alpha])

    return {
        "accuracy": float(point_estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_width": float(ci_high - ci_low),
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
    }


def cis_overlap(a: dict, b: dict) -> bool:
    """True if two bootstrap_accuracy_ci results' CIs overlap at all — a quick check
    for whether an apparent "winner" is actually distinguishable, per spec requirement
    3. Non-overlapping CIs are necessary but not sufficient for a rigorous significance
    claim; this is a fast sanity flag, not a hypothesis test.
    """
    return a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]
