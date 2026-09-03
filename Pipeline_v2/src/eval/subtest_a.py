"""Sub-test A (content-varying, tone-fixed): harmbench_filtered_250.csv vs
neutral_set_300.csv, on the frozen split from splits.py.

For each method, layer and pooling variant are selected by VALIDATION accuracy only
(spec requirement 2), then scored ONCE on test with a bootstrap CI (requirement 3), with
a length-correlation check run on the test scores before the result is treated as
trustworthy (the pooling section's "flag rather than report" instruction).

This module also returns each method's FROZEN configuration (which pooling/layer was
selected, and a ready-to-call predict/score function) — eval/subtest_b.py applies these
same frozen configurations to sub-test B content rather than re-selecting anything,
since sub-test B has no train/val split of its own; it exists purely to test how the
already-chosen configuration generalizes to tone variation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src import activation_store, splits
from src.config import POOLING_VARIANTS
from src.eval.bootstrap import bootstrap_accuracy_ci
from src.eval.length_correlation import length_correlation_report
from src.methods import content_pole, neutral_origin, random_direction, tone_pole


@dataclass
class FittedMethodResult:
    method_name: str
    pooling_variant: str
    layer: int
    predict_fn: Callable[[np.ndarray], np.ndarray]
    score_fn: Callable[[np.ndarray], np.ndarray]
    val_accuracy: float
    reference_point: np.ndarray  # the midpoint/origin this method scores relative to —
                                  # reused by the random-direction control so it isolates
                                  # the effect of a random DIRECTION, not a random point.
    test_result: dict | None = None
    note: str = ""


def _accuracy(pred: np.ndarray, true: list[str]) -> float:
    return float((np.asarray(pred) == np.asarray(true)).mean())


def _ids_and_labels(splits_df, split_name: str) -> tuple[list[str], list[str], list[str]]:
    """Returns (harmful_ids, neutral_ids, all_ids_in_harmful_then_neutral_order) for one split."""
    subset = splits_df[splits_df["split"] == split_name]
    harmful_ids = subset[subset["label"] == "harmful"]["PromptID"].tolist()
    neutral_ids = subset[subset["label"] == "neutral"]["PromptID"].tolist()
    return harmful_ids, neutral_ids, harmful_ids + neutral_ids


def run_subtest_a(
    cache_dir, model_key: str, formatting_variant: str, seed: int = 42,
    include_tone_pole: bool = True,
) -> dict[str, FittedMethodResult]:
    splits_df = splits.load_frozen_splits(seed=seed)
    text_by_id = {row.prompt_id: row.text for row in splits.load_combined_rows()}

    train_harmful_ids, train_neutral_ids, _ = _ids_and_labels(splits_df, "train")
    val_harmful_ids, val_neutral_ids, val_ids = _ids_and_labels(splits_df, "val")
    test_harmful_ids, test_neutral_ids, test_ids = _ids_and_labels(splits_df, "test")

    val_labels = ["harmful"] * len(val_harmful_ids) + ["neutral"] * len(val_neutral_ids)
    test_labels = ["harmful"] * len(test_harmful_ids) + ["neutral"] * len(test_neutral_ids)
    n_layers = activation_store.available_layers(cache_dir, model_key, formatting_variant)

    # candidate_builders[method_name] returns, for one (pooling_variant, layer), a dict
    # with predict_fn/score_fn plus enough info to compute val accuracy — kept as
    # closures over that (pooling_variant, layer)'s already-loaded activation matrices.
    candidates: dict[str, list[FittedMethodResult]] = {
        "content_pole": [], "neutral_origin_distance": [], "neutral_origin_direction": [],
    }
    if include_tone_pole:
        candidates["tone_pole"] = []

    for pooling_variant in POOLING_VARIANTS:
        for layer in range(n_layers):
            load = lambda ids: activation_store.load_layer_matrix(  # noqa: E731
                cache_dir, model_key, formatting_variant, pooling_variant, layer, ids
            )
            train_harmful = load(train_harmful_ids)
            train_neutral = load(train_neutral_ids)
            val_harmful = load(val_harmful_ids)
            val_matrix = load(val_ids)

            # Method 2 — content-pole
            pole2 = content_pole.fit_content_pole(train_harmful, train_neutral)
            val_pred2 = content_pole.predict(pole2, val_matrix)
            candidates["content_pole"].append(FittedMethodResult(
                method_name="content_pole", pooling_variant=pooling_variant, layer=layer,
                predict_fn=lambda a, p=pole2: content_pole.predict(p, a),
                score_fn=lambda a, p=pole2: content_pole.score(p, a),
                val_accuracy=_accuracy(val_pred2, val_labels),
                reference_point=pole2.midpoint,
            ))

            # Method 3a — neutral-origin, unsigned distance + validation-fit threshold
            origin = neutral_origin.fit_neutral_origin(train_neutral)
            val_distance = neutral_origin.score_by_origin_distance(origin, val_matrix)
            threshold3a = neutral_origin.fit_threshold(val_distance, val_labels)
            val_pred3a = neutral_origin.predict_with_threshold(val_distance, threshold3a)
            candidates["neutral_origin_distance"].append(FittedMethodResult(
                method_name="neutral_origin_distance", pooling_variant=pooling_variant, layer=layer,
                predict_fn=lambda a, o=origin, t=threshold3a: neutral_origin.predict_with_threshold(
                    neutral_origin.score_by_origin_distance(o, a), t),
                score_fn=lambda a, o=origin: neutral_origin.score_by_origin_distance(o, a),
                val_accuracy=_accuracy(val_pred3a, val_labels),
                reference_point=origin.origin,
                note="threshold fit on validation",
            ))

            # Method 3b — neutral-origin, direction fit on validation-split harmful mean,
            # threshold ALSO fit on validation (see neutral_origin.py module docstring:
            # `origin` is the neutral class's own train mean, not a true midpoint, so a
            # fixed threshold of 0 is not a valid assumption here the way it is for
            # Method 2's true-midpoint geometry — tests/test_eval_pipeline_synthetic.py
            # caught this when it was still hardcoded).
            # NOTE: this direction (and now threshold) are fit ON the same validation
            # split that then also selects (pooling_variant, layer) by accuracy under
            # them — not test-set leakage (the protocol's actual red line), but it does
            # mean this variant's validation accuracy is somewhat optimistic relative to
            # a truly independent selection set. Flagged here rather than hidden; test
            # accuracy is still reported honestly since test was never touched.
            direction3b = neutral_origin.fit_validation_direction(origin, val_harmful)
            val_scores3b = neutral_origin.score_by_validation_direction(origin, direction3b, val_matrix)
            threshold3b = neutral_origin.fit_threshold(val_scores3b, val_labels)
            val_pred3b = neutral_origin.predict_with_threshold(val_scores3b, threshold3b)
            candidates["neutral_origin_direction"].append(FittedMethodResult(
                method_name="neutral_origin_direction", pooling_variant=pooling_variant, layer=layer,
                predict_fn=lambda a, o=origin, d=direction3b, t=threshold3b: neutral_origin.predict_with_threshold(
                    neutral_origin.score_by_validation_direction(o, d, a), t),
                score_fn=lambda a, o=origin, d=direction3b: neutral_origin.score_by_validation_direction(o, d, a),
                val_accuracy=_accuracy(val_pred3b, val_labels),
                reference_point=origin.origin,
                note="direction AND threshold fit on validation split (see module docstring caveat)",
            ))

            # Method 1 — tone-pole, vector from the generation cache, scored on THESE
            # (sub-test A) prompt activations for a fair cross-method comparison.
            if include_tone_pole:
                try:
                    pole1 = tone_pole.fit_tone_pole(cache_dir, model_key, pooling_variant, layer)
                except activation_store.ActivationLookupError:
                    continue  # no generation cache yet for this model — skip, don't fabricate a result
                val_pred1 = tone_pole.predict(pole1, val_matrix)
                candidates["tone_pole"].append(FittedMethodResult(
                    method_name="tone_pole", pooling_variant=pooling_variant, layer=layer,
                    predict_fn=lambda a, p=pole1: tone_pole.predict(p, a),
                    score_fn=lambda a, p=pole1: tone_pole.score(p, a),
                    val_accuracy=_accuracy(val_pred1, val_labels),
                    reference_point=pole1.midpoint,
                ))

    results: dict[str, FittedMethodResult] = {}
    for method_name, method_candidates in candidates.items():
        if not method_candidates:
            continue
        best = max(method_candidates, key=lambda c: c.val_accuracy)

        test_matrix = activation_store.load_layer_matrix(
            cache_dir, model_key, formatting_variant, best.pooling_variant, best.layer, test_ids
        )
        test_pred = best.predict_fn(test_matrix)
        test_scores = best.score_fn(test_matrix)
        ci = bootstrap_accuracy_ci(np.asarray(test_labels), test_pred, seed=seed)
        length_report = length_correlation_report(
            test_scores, [text_by_id[i] for i in test_ids],
            np.array([1.0 if lbl == "harmful" else 0.0 for lbl in test_labels]),
        )
        best.test_result = {**ci, "length_correlation": length_report}
        results[method_name] = best

    return results


def evaluate_random_direction_control(
    cache_dir, model_key: str, formatting_variant: str, fitted: FittedMethodResult,
    seed: int = 42, n_directions: int = 20,
) -> dict:
    """Null control for `fitted` (spec requirement 4): re-scores the SAME test set,
    at the SAME (pooling_variant, layer, reference_point) `fitted` used, but replacing
    its learned direction with `n_directions` random unit vectors, and reports the
    resulting accuracy distribution. `fitted.test_result` only means something if it
    clearly exceeds this range.
    """
    splits_df = splits.load_frozen_splits(seed=seed)
    test_harmful_ids, test_neutral_ids, test_ids = _ids_and_labels(splits_df, "test")
    test_labels = np.array(["harmful"] * len(test_harmful_ids) + ["neutral"] * len(test_neutral_ids))

    test_matrix = activation_store.load_layer_matrix(
        cache_dir, model_key, formatting_variant, fitted.pooling_variant, fitted.layer, test_ids
    )
    hidden_dim = test_matrix.shape[1]

    accuracies = []
    for i in range(n_directions):
        direction = random_direction.random_unit_direction(hidden_dim, seed=seed + i)
        pred = random_direction.predict(fitted.reference_point, direction, test_matrix)
        accuracies.append(_accuracy(pred, test_labels))

    accuracies = np.array(accuracies)
    return {
        "method_controlled_for": fitted.method_name,
        "pooling_variant": fitted.pooling_variant,
        "layer": fitted.layer,
        "n_directions": n_directions,
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std(ddof=1)) if n_directions > 1 else 0.0,
        "min_accuracy": float(accuracies.min()),
        "max_accuracy": float(accuracies.max()),
        "fitted_method_test_accuracy": fitted.test_result["accuracy"] if fitted.test_result else None,
    }
