"""Sub-test B (tone-varying, content-fixed): calm vs. hostile paraphrases of the same
underlying request. Applies the FROZEN method configurations `eval.subtest_a.run_subtest_a`
already selected (pooling variant, layer, fitted pole/threshold/direction) — sub-test B
has no train/val split of its own and selects nothing; it exists purely to test how a
configuration already chosen on sub-test A generalizes to tone variation.

Per data/subtest_b_MANIFEST.md and explicit instruction from Fjord: the neutral arm
(N=24 pairs = 48 items) is statistically reportable with a bootstrap CI. The harmful
arm has only 3 verified-clean pairs (6 items) out of 9 sourced — bootstrap CI at N=3 is
uninformative (binomial 95% CI for 3/3 spans roughly [29%, 100%]), so it is NEVER
returned as a bootstrap-CI'd, reportable accuracy number here. It comes back as a
separate, explicitly-labeled qualitative block (`reportable: False`) with the raw
per-pair outcomes, for the paper to cite as a case study only. Do not merge it into the
neutral arm's numbers, and do not run any headline result off of it alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import activation_store
from src.config import (
    SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES,
    SUBTEST_B_HARMFUL_CSV,
    SUBTEST_B_NEUTRAL_CSV,
)
from src.eval.bootstrap import bootstrap_accuracy_ci
from src.eval.length_correlation import length_correlation_report
from src.eval.subtest_a import FittedMethodResult


def load_neutral_arm_items() -> tuple[list[str], list[str], list[str]]:
    """Returns (item_ids, texts, labels) for the 24 calm + 24 hostile = 48-item
    reportable neutral arm. Ground-truth label is "neutral" for every item — these are
    tone paraphrases of a NEUTRAL request, so a content-sensitive detector should call
    all 48 "neutral" regardless of phrasing."""
    df = pd.read_csv(SUBTEST_B_NEUTRAL_CSV)
    item_ids, texts, labels = [], [], []
    for _, row in df.iterrows():
        item_ids.append(f"{row['PairID']}__calm")
        texts.append(row["calm_prompt"])
        labels.append("neutral")
        item_ids.append(f"{row['PairID']}__hostile")
        texts.append(row["hostile_prompt"])
        labels.append("neutral")
    return item_ids, texts, labels


def load_harmful_arm_clean_items() -> tuple[list[str], list[str], list[str]]:
    """Returns (item_ids, texts, labels) for ONLY the 3 verified-clean sourced pairs
    (6 items) — see SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES / data/subtest_b_MANIFEST.md.
    Ground-truth label is "harmful" for every item (both calm and hostile phrasings of
    a harmful request)."""
    df = pd.read_csv(SUBTEST_B_HARMFUL_CSV)
    item_ids, texts, labels = [], [], []
    for row_idx in SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES:
        row = df.iloc[row_idx]
        item_ids.append(f"STB_H_{row_idx:03d}__calm")
        texts.append(row["calm_prompt"])
        labels.append("harmful")
        item_ids.append(f"STB_H_{row_idx:03d}__hostile")
        texts.append(row["hostile_prompt"])
        labels.append("harmful")
    return item_ids, texts, labels


def _tone_of(item_id: str) -> str:
    return "calm" if item_id.endswith("__calm") else "hostile"


def _evaluate_arm(cache_dir, model_key, formatting_variant, fitted: FittedMethodResult,
                   item_ids, texts, labels, seed) -> dict:
    matrix = activation_store.load_layer_matrix(
        cache_dir, model_key, formatting_variant, fitted.pooling_variant, fitted.layer, item_ids
    )
    predictions = fitted.predict_fn(matrix)
    scores = fitted.score_fn(matrix)
    labels_arr = np.asarray(labels)

    tones = np.array([_tone_of(iid) for iid in item_ids])
    calm_mask, hostile_mask = tones == "calm", tones == "hostile"

    return {
        "overall": bootstrap_accuracy_ci(labels_arr, predictions, seed=seed),
        "calm_only_accuracy": float((predictions[calm_mask] == labels_arr[calm_mask]).mean()),
        "hostile_only_accuracy": float((predictions[hostile_mask] == labels_arr[hostile_mask]).mean()),
        "n_calm": int(calm_mask.sum()),
        "n_hostile": int(hostile_mask.sum()),
        "length_correlation": length_correlation_report(
            scores, texts, np.array([1.0 if lbl == "harmful" else 0.0 for lbl in labels])
        ),
    }


def _qualitative_harmful_arm(cache_dir, model_key, formatting_variant, fitted: FittedMethodResult,
                              item_ids, texts, labels) -> dict:
    matrix = activation_store.load_layer_matrix(
        cache_dir, model_key, formatting_variant, fitted.pooling_variant, fitted.layer, item_ids
    )
    predictions = fitted.predict_fn(matrix)
    per_item = [
        {"item_id": iid, "tone": _tone_of(iid), "predicted": str(pred), "true_label": true,
         "correct": bool(pred == true)}
        for iid, pred, true in zip(item_ids, predictions, labels)
    ]
    return {
        "reportable": False,
        "note": (
            "N=3 verified-clean sourced pairs (6 items). Bootstrap CI at this N is "
            "uninformative (binomial 95% CI for 3/3 correct spans roughly [29%, 100%]) "
            "-- see data/subtest_b_MANIFEST.md. Qualitative case-study supplement only; "
            "do not cite an accuracy percentage from this block as a result."
        ),
        "n_pairs": len(item_ids) // 2,
        "per_item": per_item,
    }


def run_subtest_b(
    cache_dir, model_key: str, formatting_variant: str, fitted_methods: dict[str, FittedMethodResult],
    seed: int = 42,
) -> dict:
    """fitted_methods: the dict returned by eval.subtest_a.run_subtest_a (already fit
    and layer/pooling-selected on sub-test A). Returns, per method name:
    {"neutral_arm": <reportable, bootstrap-CI'd>, "harmful_arm_qualitative": <N=3, not a
    reportable result>}.
    """
    neutral_ids, neutral_texts, neutral_labels = load_neutral_arm_items()
    harmful_ids, harmful_texts, harmful_labels = load_harmful_arm_clean_items()

    results = {}
    for method_name, fitted in fitted_methods.items():
        results[method_name] = {
            "neutral_arm": _evaluate_arm(
                cache_dir, model_key, formatting_variant, fitted,
                neutral_ids, neutral_texts, neutral_labels, seed,
            ),
            "harmful_arm_qualitative": _qualitative_harmful_arm(
                cache_dir, model_key, formatting_variant, fitted,
                harmful_ids, harmful_texts, harmful_labels,
            ),
        }
    return results


def summarize_a_b_gap(subtest_a_results: dict[str, FittedMethodResult], subtest_b_results: dict) -> dict:
    """The spec's win condition: "report accuracy ... separately on sub-test A and
    sub-test B, plus the gap between them ... Best method = smallest A/B gap with
    competitive absolute accuracy on A." Gap is computed against sub-test B's
    REPORTABLE neutral arm only — the harmful arm never contributes a number here.
    """
    summary = {}
    for method_name, fitted in subtest_a_results.items():
        if method_name not in subtest_b_results:
            continue
        a_acc = fitted.test_result["accuracy"]
        b_acc = subtest_b_results[method_name]["neutral_arm"]["overall"]["accuracy"]
        summary[method_name] = {
            "subtest_a_accuracy": a_acc,
            "subtest_a_ci": (fitted.test_result["ci_low"], fitted.test_result["ci_high"]),
            "subtest_b_neutral_arm_accuracy": b_acc,
            "subtest_b_neutral_arm_ci": (
                subtest_b_results[method_name]["neutral_arm"]["overall"]["ci_low"],
                subtest_b_results[method_name]["neutral_arm"]["overall"]["ci_high"],
            ),
            "a_b_gap": a_acc - b_acc,
        }
    return summary
