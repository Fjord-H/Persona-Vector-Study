"""
Fifth external review — threshold recalibration experiment.

Hypothesis: if AUROC is high but accuracy is low due to threshold-transfer failure,
refitting the decision threshold on a small labeled sample from sub-test B should
recover most of the lost accuracy.

Tested on two clearest threshold-transfer failures:
  1. llama-3.2-3b/raw probe    — AUROC 0.990, accuracy 56.2%
  2. qwen2.5-1.5b/chat content_pole — AUROC 0.910, accuracy 14.6%

Design:
  Combined arm = 48 neutral + 6 harmful = 54 sub-test B items.
  Calibration slice (seed=42): 6 randomly chosen neutral pairs (12 items) + all 6
  harmful items = 18 items. Harmful arm is too small to split further (N=3 pairs)
  so all 6 harmful items go into calibration; held-out is neutral-only.
  Held-out: remaining 18 neutral pairs (36 items). All held-out items are neutral —
  accuracy on them equals 1 - false_positive_rate, which is the failure mode.

  Threshold fitting: grid search over observed calibration scores (same logic as
  src/methods/neutral_origin.py::fit_threshold). No weights or directions are changed.

  Reference baselines reported per case:
    - original-threshold accuracy on held-out 36 items (expected low for qwen-base)
    - recalibrated-threshold accuracy on held-out 36 items
    - oracle threshold: threshold maximizing accuracy on the full 54-item combined arm
      (upper bound; requires labels for all B items — not usable in practice)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT    = Path(r"C:\School\Persona_Vector")
PIPELINE     = REPO_ROOT / "Pipeline_v2"
CACHE_KAGGLE = REPO_ROOT / "pv2_cache_kaggle" / "pv2_cache"

sys.path.insert(0, str(PIPELINE))
os.environ["PV2_CACHE_DIR"]   = str(CACHE_KAGGLE)
os.environ["PV2_DATA_DIR"]    = str(REPO_ROOT / "data")
os.environ["PV2_SPLITS_JSON"] = str(REPO_ROOT / "data" / "splits.json")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import activation_store, splits
from src.eval.bootstrap import wilson_interval
from src.eval.subtest_b import load_harmful_arm_clean_items, load_neutral_arm_items
from src.methods.content_pole import fit_content_pole, score as cp_score

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Load splits and sub-test B items
# ---------------------------------------------------------------------------
splits_df = splits.load_frozen_splits(seed=42)
harmful_df = pd.read_csv(REPO_ROOT / "data" / "harmbench_filtered_250.csv")
neutral_df = pd.read_csv(REPO_ROOT / "data" / "neutral_set_300.csv")
results_csv = REPO_ROOT / "Pipeline_v2" / "method_comparison_results.csv"
res_df = pd.read_csv(results_csv)

def get_ids_and_labels(split_name):
    rows = splits_df[splits_df["split"] == split_name]
    h_ids = list(rows[rows["label"] == "harmful"]["PromptID"])
    n_ids = list(rows[rows["label"] == "neutral"]["PromptID"])
    return h_ids + n_ids, ["harmful"] * len(h_ids) + ["neutral"] * len(n_ids)

train_ids, train_labels = get_ids_and_labels("train")

# Sub-test B arms
stb_n_ids, stb_n_texts, stb_n_labels = load_neutral_arm_items()   # 48 neutral items
stb_h_ids, stb_h_texts, stb_h_labels = load_harmful_arm_clean_items()  # 6 harmful items

# Pair indices for the neutral arm (24 pairs × 2 items each, interleaved calm/hostile)
# stb_n_ids[2*i] = pair i calm, stb_n_ids[2*i+1] = pair i hostile
n_neutral_pairs = len(stb_n_ids) // 2   # 24

# ---------------------------------------------------------------------------
# Build calibration / held-out split at the PAIR level (seed=42)
# ---------------------------------------------------------------------------
pair_indices = np.arange(n_neutral_pairs)  # 0..23
shuffled = RNG.permutation(pair_indices)
cal_pair_indices = sorted(shuffled[:6])    # 6 calibration pairs
held_pair_indices = sorted(shuffled[6:])   # 18 held-out pairs

# Item indices within stb_n_ids / stb_n_labels for each group
cal_item_indices  = [2*p for p in cal_pair_indices]  + [2*p+1 for p in cal_pair_indices]
held_item_indices = [2*p for p in held_pair_indices] + [2*p+1 for p in held_pair_indices]
cal_item_indices.sort()
held_item_indices.sort()

cal_n_ids     = [stb_n_ids[i]     for i in cal_item_indices]
cal_n_labels  = [stb_n_labels[i]  for i in cal_item_indices]   # all "neutral"
held_n_ids    = [stb_n_ids[i]     for i in held_item_indices]
held_n_labels = [stb_n_labels[i]  for i in held_item_indices]  # all "neutral"

# Full calibration slice: neutral cal + all 6 harmful (too few to split)
cal_ids    = cal_n_ids    + stb_h_ids
cal_labels = cal_n_labels + stb_h_labels   # 12 neutral + 6 harmful

# Combined arm for oracle threshold (all 54 items)
combined_ids    = stb_n_ids    + stb_h_ids
combined_labels = stb_n_labels + stb_h_labels

print(f"Calibration slice: {len(cal_n_ids)} neutral + {len(stb_h_ids)} harmful = {len(cal_ids)} items")
print(f"Held-out (neutral only): {len(held_n_ids)} items ({len(held_pair_indices)} pairs)")
print(f"Oracle combined arm: {len(combined_ids)} items")

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def fit_threshold_from_scores(scores, labels):
    """Grid search on observed scores for accuracy-maximising threshold.
    Equivalent to neutral_origin.fit_threshold; duplicated here to avoid import-
    order dependency. Returns (threshold, calibration_accuracy)."""
    is_harmful = np.asarray(labels) == "harmful"
    candidates = np.unique(scores)
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = ((scores > t) == is_harmful).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t), float(best_acc)

def held_out_accuracy(scores, labels, threshold):
    return float((np.asarray(scores > threshold) == (np.asarray(labels) == "harmful")).mean())

def wilson_str(k, n):
    wi = wilson_interval(k, n)
    return f"[{wi['wilson_low']:.1%}, {wi['wilson_high']:.1%}]"

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_recalibration(model_key, fmt, method):
    print(f"\n{'='*68}")
    print(f"  {model_key}/{fmt}  |  method={method}")
    print(f"{'='*68}")

    row = res_df[(res_df["model"] == model_key) &
                 (res_df["formatting"] == fmt) &
                 (res_df["method"] == "content_pole")]
    if row.empty:
        print("  No content_pole row in results CSV — skipping")
        return
    layer   = int(row.iloc[0]["layer"])
    pooling = row.iloc[0]["pooling_variant"]
    print(f"  layer={layer}  pooling={pooling}")

    def load(ids):
        return activation_store.load_layer_matrix(
            CACHE_KAGGLE, model_key, fmt, pooling, layer, ids
        )

    try:
        X_train    = load(train_ids)
        X_cal      = load(cal_ids)
        X_held     = load(held_n_ids)
        X_combined = load(combined_ids)
    except Exception as e:
        print(f"  Failed to load activations: {e}")
        return

    # ----------------------------------------------------------------
    # Get scores depending on method
    # ----------------------------------------------------------------
    if method == "content_pole":
        # Refit pole on sub-test A train split (same as pipeline)
        h_mask = np.array(train_labels) == "harmful"
        n_mask = ~h_mask
        pole = fit_content_pole(X_train[h_mask], X_train[n_mask])
        scores_cal      = cp_score(pole, X_cal)
        scores_held     = cp_score(pole, X_held)
        scores_combined = cp_score(pole, X_combined)
        original_threshold = 0.0   # midpoint by construction

    elif method == "probe":
        # Standardized probe (same protocol as review4_tfidf_auroc_probe_refit.py)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s   = scaler.transform(X_cal)
        X_held_s  = scaler.transform(X_held)
        X_combined_s = scaler.transform(X_combined)

        # C sweep on sub-test A val split (same as review4)
        val_ids_list, val_labels_list = get_ids_and_labels("val")
        X_val_s = scaler.transform(load(val_ids_list))
        C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_C, best_val_acc = None, -1.0
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
            clf.fit(X_train_s, train_labels)
            val_acc = (clf.predict(X_val_s) == np.array(val_labels_list)).mean()
            if val_acc > best_val_acc:
                best_val_acc, best_C = val_acc, C
        print(f"  Best C={best_C} (val acc {best_val_acc:.1%})")

        clf = LogisticRegression(C=best_C, max_iter=2000, random_state=42)
        clf.fit(X_train_s, train_labels)
        harm_idx = list(clf.classes_).index("harmful")
        scores_cal      = clf.predict_proba(X_cal_s)[:, harm_idx]
        scores_held     = clf.predict_proba(X_held_s)[:, harm_idx]
        scores_combined = clf.predict_proba(X_combined_s)[:, harm_idx]
        original_threshold = 0.5   # default logistic regression cutoff

    else:
        raise ValueError(f"Unknown method: {method}")

    # ----------------------------------------------------------------
    # Original threshold on held-out
    # ----------------------------------------------------------------
    orig_acc_held = held_out_accuracy(scores_held, held_n_labels, original_threshold)
    orig_correct = int(round(orig_acc_held * len(held_n_labels)))
    print(f"\n  Original threshold ({original_threshold})")
    print(f"    Held-out accuracy: {orig_acc_held:.1%}  ({orig_correct}/{len(held_n_labels)})")
    print(f"    Wilson CI: {wilson_str(orig_correct, len(held_n_labels))}")

    # ----------------------------------------------------------------
    # Recalibrated threshold (fit on calibration slice)
    # ----------------------------------------------------------------
    recal_threshold, recal_cal_acc = fit_threshold_from_scores(scores_cal, cal_labels)
    recal_acc_held = held_out_accuracy(scores_held, held_n_labels, recal_threshold)
    recal_correct = int(round(recal_acc_held * len(held_n_labels)))
    print(f"\n  Recalibrated threshold ({recal_threshold:.4f}, fit on {len(cal_ids)}-item cal slice)")
    print(f"    Cal slice accuracy: {recal_cal_acc:.1%}")
    print(f"    Held-out accuracy: {recal_acc_held:.1%}  ({recal_correct}/{len(held_n_labels)})")
    print(f"    Wilson CI: {wilson_str(recal_correct, len(held_n_labels))}")
    print(f"    Delta from original: {recal_acc_held - orig_acc_held:+.1%}")

    # ----------------------------------------------------------------
    # Oracle threshold (upper bound — fit on full combined arm)
    # ----------------------------------------------------------------
    oracle_threshold, _ = fit_threshold_from_scores(scores_combined, combined_labels)
    oracle_acc_held = held_out_accuracy(scores_held, held_n_labels, oracle_threshold)
    oracle_correct = int(round(oracle_acc_held * len(held_n_labels)))
    print(f"\n  Oracle threshold ({oracle_threshold:.4f}, fit on full {len(combined_ids)}-item combined arm — NOT usable in practice)")
    print(f"    Held-out accuracy: {oracle_acc_held:.1%}  ({oracle_correct}/{len(held_n_labels)})")

    # ----------------------------------------------------------------
    # AUROC on combined arm (threshold-free, for reference)
    # ----------------------------------------------------------------
    combined_int = [1 if l == "harmful" else 0 for l in combined_labels]
    auroc = roc_auc_score(combined_int, scores_combined)
    print(f"\n  Combined-arm AUROC (reference): {auroc:.3f}")

    # Summary line
    print(f"\n  SUMMARY: original {orig_acc_held:.1%} -> recalibrated {recal_acc_held:.1%}"
          f" -> oracle {oracle_acc_held:.1%}  (AUROC {auroc:.3f})")

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
run_recalibration("qwen2.5-1.5b", "chat",  "content_pole")
run_recalibration("llama-3.2-3b", "raw",   "probe")

print("\ndone")
