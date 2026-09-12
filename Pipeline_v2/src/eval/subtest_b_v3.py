"""Sub-test B v3 (style-transform / hard-natural-pairs): applies the FROZEN method
configurations selected by eval.subtest_a.run_subtest_a — no fitting or layer selection
is done here. Reports accuracy + bootstrap CI broken out by source and by mutation, so
the contribution of each design axis can be read independently.

See data/subtest_b_v3/subtest_b_v3_MANIFEST.md for full dataset documentation. Key
properties relevant to interpreting results here:
- Labels are "harmful" / "benign"; "benign" is mapped to "neutral" when comparing with
  predict_fn output (which was trained on "harmful" / "neutral").
- Type 1 items (7 mutations × HarmBench + neutral_set): same SOURCE CONTENT as sub-test
  A but a different surface form — activations must be extracted separately; these item
  ids are distinct from sub-test A ids.
- Type 2 items (XSTest + OR-Bench, mutation="none"): naturally-written hard pairs — the
  source texts are new content not seen in sub-test A.
- TF-IDF baseline on this set is ~69% overall (all mutation sub-groups < 80%), so
  performance near chance is informative and performance well above TF-IDF is meaningful.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import activation_store
from src.config import SUBTEST_B_V3_JSONL
from src.eval.bootstrap import bootstrap_accuracy_ci
from src.eval.length_correlation import length_correlation_report
from src.eval.subtest_a import FittedMethodResult


@dataclass
class V3Item:
    item_id: str    # f"{source}:{source_id}:{mutation}" — matches the extraction cache key
    text: str
    label_raw: str  # "harmful" | "benign" as written in the JSONL
    label_eval: str # "harmful" | "neutral" — benign mapped to neutral for predict_fn compat
    source: str
    mutation: str


def load_v3_items(jsonl_path: Path = SUBTEST_B_V3_JSONL) -> list[V3Item]:
    """Loads all items from subtest_b_v3.jsonl. The item_id f"{source}:{source_id}:{mutation}"
    matches the key used during activation extraction (same formula as build_subtest_b_v3.py).
    """
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label_raw = obj["label"]
            items.append(V3Item(
                item_id=f"{obj['source']}:{obj['source_id']}:{obj['mutation']}",
                text=obj["text"],
                label_raw=label_raw,
                label_eval="harmful" if label_raw == "harmful" else "neutral",
                source=obj["source"],
                mutation=obj["mutation"],
            ))
    return items


def _evaluate_subgroup(
    items: list[V3Item],
    matrix: np.ndarray,
    all_item_ids: list[str],
    id_to_row: dict[str, int],
    predict_fn,
    score_fn,
    seed: int,
) -> dict:
    """Shared helper: given a subset of items, slice the already-loaded matrix and
    return accuracy + bootstrap CI + length correlation."""
    if not items:
        return {"skipped": "empty subgroup"}
    indices = [id_to_row[it.item_id] for it in items]
    sub_matrix = matrix[indices]
    predictions = predict_fn(sub_matrix)
    scores = score_fn(sub_matrix)
    true_labels = np.array([it.label_eval for it in items])
    numeric_labels = np.array([1.0 if it.label_raw == "harmful" else 0.0 for it in items])

    result = bootstrap_accuracy_ci(true_labels, predictions, seed=seed)
    result["length_correlation"] = length_correlation_report(scores, [it.text for it in items], numeric_labels)

    # Per-label accuracy (harmful / benign) when both are present
    harmful_items = [it for it in items if it.label_raw == "harmful"]
    benign_items  = [it for it in items if it.label_raw == "benign"]
    if harmful_items:
        harm_idx = [id_to_row[it.item_id] for it in harmful_items]
        harm_pred = predict_fn(matrix[harm_idx])
        harm_true = np.array([it.label_eval for it in harmful_items])
        result["harmful_accuracy"] = float((harm_pred == harm_true).mean())
        result["n_harmful"] = len(harmful_items)
    if benign_items:
        ben_idx = [id_to_row[it.item_id] for it in benign_items]
        ben_pred = predict_fn(matrix[ben_idx])
        ben_true = np.array([it.label_eval for it in benign_items])
        result["benign_accuracy"] = float((ben_pred == ben_true).mean())
        result["n_benign"] = len(benign_items)

    return result


def run_subtest_b_v3(
    cache_dir,
    model_key: str,
    formatting_variant: str,
    fitted_methods: dict[str, FittedMethodResult],
    seed: int = 42,
    jsonl_path: Path = SUBTEST_B_V3_JSONL,
) -> dict:
    """Applies frozen method configs (from run_subtest_a) to the v3 item set and returns,
    per method:
      - overall: bootstrap_accuracy_ci + length_correlation
      - by_source: one entry per unique source (harmbench, neutral_set, xstest, or_bench)
      - by_mutation: one entry per unique mutation type (caesar, morse, atbash, ascii,
                     slang, misspellings, role_play, none)

    Raises activation_store.ActivationLookupError if the v3 activations have not been
    extracted yet — run the extraction notebook first with these item_ids included.
    No fitting or layer selection is performed here.
    """
    items = load_v3_items(jsonl_path)
    item_ids = [it.item_id for it in items]

    results: dict[str, dict] = {}
    for method_name, fitted in fitted_methods.items():
        matrix = activation_store.load_layer_matrix(
            cache_dir, model_key, formatting_variant,
            fitted.pooling_variant, fitted.layer, item_ids,
        )
        id_to_row = {iid: i for i, iid in enumerate(item_ids)}

        overall = _evaluate_subgroup(
            items, matrix, item_ids, id_to_row,
            fitted.predict_fn, fitted.score_fn, seed,
        )

        by_source: dict[str, dict] = {}
        for source in sorted({it.source for it in items}):
            sub = [it for it in items if it.source == source]
            by_source[source] = _evaluate_subgroup(
                sub, matrix, item_ids, id_to_row,
                fitted.predict_fn, fitted.score_fn, seed,
            )

        by_mutation: dict[str, dict] = {}
        for mutation in sorted({it.mutation for it in items}):
            sub = [it for it in items if it.mutation == mutation]
            by_mutation[mutation] = _evaluate_subgroup(
                sub, matrix, item_ids, id_to_row,
                fitted.predict_fn, fitted.score_fn, seed,
            )

        results[method_name] = {
            "pooling_variant": fitted.pooling_variant,
            "layer": fitted.layer,
            "overall": overall,
            "by_source": by_source,
            "by_mutation": by_mutation,
        }

    return results
