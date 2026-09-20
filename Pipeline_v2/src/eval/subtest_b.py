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
from sklearn.metrics import roc_auc_score

from src import activation_store, splits
from src.config import (
    SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES,
    SUBTEST_B_HARMFUL_CSV,
    SUBTEST_B_NEUTRAL_CSV,
)
from src.eval.bootstrap import bootstrap_accuracy_ci, wilson_interval
from src.eval.length_correlation import length_correlation_report
from src.eval.subtest_a import FittedMethodResult
from src.methods import probe as probe_mod


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

    # Label-polarity sanity: score_fn must return HIGHER values for "harmful" items.
    # Binary numeric: 1 = harmful, 0 = neutral. Consistent with content_pole.score (signed
    # projection onto harmful - neutral direction, threshold=0) and neutral_origin.score.
    # History note: v1 had a dim=0/dim=1 sign bug (defect_report.md #3); verify polarity
    # explicitly here every time so a regression would be caught by the test suite.
    numeric_labels = np.array([1.0 if lbl == "harmful" else 0.0 for lbl in labels])
    unique_classes = np.unique(numeric_labels)

    # AUROC — threshold-free discrimination. Only computable when both classes present.
    # For the neutral arm (all-neutral), auroc is None and we instead report
    # fraction_predicted_harmful to distinguish inversion from threshold artifact.
    # For a combined arm: auroc < 0.5 → true inversion; auroc ~0.5 → threshold artifact;
    # auroc > 0.5 → correct ranking.
    if len(unique_classes) > 1:
        auroc = float(roc_auc_score(numeric_labels, scores))
        polarity_note = (
            "inverted: score_fn ranks neutral higher than harmful" if auroc < 0.5 else
            "correct: score_fn ranks harmful higher than neutral"
        )
    else:
        auroc = None
        polarity_note = (
            "single-class arm — AUROC undefined; see fraction_predicted_harmful "
            "to distinguish inversion from threshold artifact"
        )

    n_total = len(predictions)
    n_predicted_harmful = int((predictions == "harmful").sum())
    bootstrap = bootstrap_accuracy_ci(labels_arr, predictions, seed=seed)

    # Wilson score interval on the same successes/n as the bootstrap
    successes = int(round(bootstrap["accuracy"] * n_total))
    wilson = wilson_interval(successes, n_total)

    return {
        "overall": bootstrap,
        "wilson": wilson,
        "auroc": auroc,
        "polarity_note": polarity_note,
        "fraction_predicted_harmful": float(n_predicted_harmful / n_total),
        "n_predicted_harmful": n_predicted_harmful,
        "calm_only_accuracy": float((predictions[calm_mask] == labels_arr[calm_mask]).mean()),
        "hostile_only_accuracy": float((predictions[hostile_mask] == labels_arr[hostile_mask]).mean()),
        "n_calm": int(calm_mask.sum()),
        "n_hostile": int(hostile_mask.sum()),
        "length_correlation": length_correlation_report(scores, texts, numeric_labels),
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


def run_probe_on_subtest_b(
    cache_dir, model_key: str, formatting_variant: str, fitted: FittedMethodResult,
    seed: int = 42,
) -> dict:
    """Trains a logistic-regression probe on sub-test A's TRAIN-split activations
    (same layer and pooling variant as `fitted`) and evaluates it on sub-test B's
    neutral arm (N=48 items), so the probe comparison is apples-to-apples with the
    nearest-centroid content_pole result on sub-test B.

    Rationale (reviewer Task 2): the existing notebook 02 probe evaluation tests the
    probe only on sub-test A's test split — a tuned discriminative model (probe) vs. an
    untuned geometric heuristic (content_pole) on DIFFERENT held-out sets is not a fair
    comparison. This function fixes that by giving both methods the same sub-test B
    evaluation set, making the comparison TF-IDF vs. content_pole vs. probe on the same
    tone-varying items.

    Returns: accuracy + Wilson CI on sub-test B neutral arm, plus training set metadata.
    Does NOT re-select layer or pooling — uses whatever `fitted` specifies.
    """
    splits_df = splits.load_frozen_splits(seed=seed)
    subset = splits_df[splits_df["split"] == "train"]
    train_harmful_ids = subset[subset["label"] == "harmful"]["PromptID"].tolist()
    train_neutral_ids = subset[subset["label"] == "neutral"]["PromptID"].tolist()
    train_ids = train_harmful_ids + train_neutral_ids
    train_labels = np.array(["harmful"] * len(train_harmful_ids) + ["neutral"] * len(train_neutral_ids))

    load = lambda ids: activation_store.load_layer_matrix(  # noqa: E731
        cache_dir, model_key, formatting_variant, fitted.pooling_variant, fitted.layer, ids
    )
    train_matrix = load(train_ids)

    neutral_ids, _, neutral_labels = load_neutral_arm_items()
    neutral_matrix = load(neutral_ids)

    result = probe_mod.fit_and_score_probe(
        train_matrix, train_labels, neutral_matrix, np.asarray(neutral_labels), seed=seed,
    )
    n = result["n_test"]
    successes = int(round(result["accuracy"] * n))
    return {
        "accuracy": result["accuracy"],
        "wilson": wilson_interval(successes, n),
        "n_train": result["n_train"],
        "n_test": n,
        "pooling_variant": fitted.pooling_variant,
        "layer": fitted.layer,
        "note": (
            "Probe trained on sub-test A train split, evaluated on sub-test B neutral arm. "
            "Layer and pooling inherited from fitted content_pole (no new selection)."
        ),
    }


def run_subtest_b(
    cache_dir, model_key: str, formatting_variant: str, fitted_methods: dict[str, FittedMethodResult],
    seed: int = 42,
) -> dict:
    """fitted_methods: the dict returned by eval.subtest_a.run_subtest_a (already fit
    and layer/pooling-selected on sub-test A). Returns, per method name:
    {"neutral_arm": <reportable, bootstrap-CI'd>, "harmful_arm_qualitative": <N=3, not a
    reportable result>, "probe_on_subtest_b": <probe accuracy on neutral arm>}.
    """
    neutral_ids, neutral_texts, neutral_labels = load_neutral_arm_items()
    harmful_ids, harmful_texts, harmful_labels = load_harmful_arm_clean_items()

    results = {}
    for method_name, fitted in fitted_methods.items():
        neutral_arm = _evaluate_arm(
            cache_dir, model_key, formatting_variant, fitted,
            neutral_ids, neutral_texts, neutral_labels, seed,
        )
        # Combined-arm AUROC: neutral (N=48) + clean harmful (N=6) together.
        # This is the only way to get a threshold-free discrimination score that spans
        # both classes, since the neutral arm alone has a single class. The harmful arm
        # contribution is small (N=6) but sufficient to anchor the AUROC computation.
        # Interpretation: auroc < 0.5 → true score inversion (base-model collapse is a
        # confident backwards prediction); auroc ~0.5 → threshold artifact (scores are
        # uninformative on this distribution); auroc > 0.5 → correct discrimination.
        combined_arm = _evaluate_arm(
            cache_dir, model_key, formatting_variant, fitted,
            neutral_ids + harmful_ids, neutral_texts + harmful_texts,
            neutral_labels + harmful_labels, seed,
        )
        results[method_name] = {
            "neutral_arm": neutral_arm,
            "harmful_arm_qualitative": _qualitative_harmful_arm(
                cache_dir, model_key, formatting_variant, fitted,
                harmful_ids, harmful_texts, harmful_labels,
            ),
            "combined_arms_auroc": combined_arm["auroc"],
            "combined_arms_polarity_note": combined_arm["polarity_note"],
            "probe_on_subtest_b": run_probe_on_subtest_b(
                cache_dir, model_key, formatting_variant, fitted, seed=seed,
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
        neutral_arm = subtest_b_results[method_name]["neutral_arm"]
        b_acc = neutral_arm["overall"]["accuracy"]
        b_n = neutral_arm["overall"]["n"]
        b_successes = neutral_arm["wilson"]["successes"]
        summary[method_name] = {
            "subtest_a_accuracy": a_acc,
            "subtest_a_ci": (fitted.test_result["ci_low"], fitted.test_result["ci_high"]),
            "subtest_b_neutral_arm_accuracy": b_acc,
            "subtest_b_neutral_arm_pairs_correct": f"{b_successes}/{b_n // 2} pairs",
            "subtest_b_neutral_arm_ci_bootstrap": (
                neutral_arm["overall"]["ci_low"],
                neutral_arm["overall"]["ci_high"],
            ),
            "subtest_b_neutral_arm_ci_wilson": (
                neutral_arm["wilson"]["wilson_low"],
                neutral_arm["wilson"]["wilson_high"],
            ),
            "subtest_b_combined_auroc": subtest_b_results[method_name].get("combined_arms_auroc"),
            "subtest_b_polarity_note": subtest_b_results[method_name].get("combined_arms_polarity_note"),
            "a_b_gap": a_acc - b_acc,
        }
    return summary
