"""r(score, char_length) check — spec: "Check r(score, char_length) directly on the
extracted activations before trusting any downstream method-comparison result... If a
variant shows |r(score, length)| approaching v1's -0.911, treat that variant's results
as unreliable for reporting... prefer whichever variant shows the weakest length
correlation alongside the strongest label correlation."

This is deliberately NOT folded silently into the method-comparison report — every
caller in eval/subtest_a.py and eval/subtest_b.py must call
`length_correlation_report` and surface `flagged=True` results explicitly rather than
reporting a number without it, per the spec's "flag rather than report" instruction.
"""
from __future__ import annotations

import numpy as np

# v1's worst observed length-channel correlation (GPT-2 layer 6, defect_report.md
# S2-04). Used as the flag threshold's reference point, not an arbitrary cutoff.
V1_WORST_LENGTH_CORRELATION = -0.911
FLAG_THRESHOLD = 0.7  # |r(score, length)| at or above this is treated as v1-level dominance


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def length_correlation_report(scores: np.ndarray, texts: list[str], labels: np.ndarray) -> dict:
    """labels: array-like of 0/1 (or any two-valued numeric/bool array) aligned with
    scores and texts. Returns r(score, length), r(score, label), and a `flagged` bool.
    """
    lengths = np.array([len(t) for t in texts])
    numeric_labels = np.asarray(labels, dtype=np.float64)

    r_length = _pearsonr(scores, lengths)
    r_label = _pearsonr(scores, numeric_labels)
    flagged = abs(r_length) >= FLAG_THRESHOLD

    return {
        "r_score_length": r_length,
        "r_score_label": r_label,
        "n": len(scores),
        "flagged": flagged,
        "flag_reason": (
            f"|r(score,length)|={abs(r_length):.3f} >= {FLAG_THRESHOLD} "
            f"(v1's dominated case was {V1_WORST_LENGTH_CORRELATION})"
            if flagged else None
        ),
    }


def pick_least_length_dominated_variant(reports_by_variant: dict[str, dict]) -> str:
    """Given {variant_name: length_correlation_report(...)}, returns the variant name
    with the weakest |r(score, length)|, breaking ties by the strongest |r(score,label)|.
    Per spec: "prefer whichever variant shows the weakest length correlation alongside
    the strongest label correlation."
    """
    if not reports_by_variant:
        raise ValueError("pick_least_length_dominated_variant: no variants given")
    return min(
        reports_by_variant,
        key=lambda name: (abs(reports_by_variant[name]["r_score_length"]),
                           -abs(reports_by_variant[name]["r_score_label"])),
    )
