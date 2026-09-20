"""
Fourth external review — Points 3 and 5.

Point 3: TF-IDF AUROC on sub-test A (test split) and sub-test B (combined arm).
Point 5: Logistic-regression probe refit with StandardScaler + C sweep on val,
         for gpt2-medium/raw, qwen2.5-1.5b/chat, llama-3.2-3b/raw.
         Reports accuracy and AUROC on sub-test B, compared with current defaults.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO_ROOT   = Path(r"C:\School\Persona_Vector")
PIPELINE    = REPO_ROOT / "Pipeline_v2"
CACHE_KAGGLE = REPO_ROOT / "pv2_cache_kaggle" / "pv2_cache"

sys.path.insert(0, str(PIPELINE))
os.environ["PV2_CACHE_DIR"]   = str(CACHE_KAGGLE)
os.environ["PV2_DATA_DIR"]    = str(REPO_ROOT / "data")
os.environ["PV2_SPLITS_JSON"] = str(REPO_ROOT / "data" / "splits.json")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import activation_store, splits
from src.eval.bootstrap import wilson_interval
from src.eval.subtest_a import _ids_and_labels
from src.eval.subtest_b import load_harmful_arm_clean_items, load_neutral_arm_items
from src.methods.content_pole import fit_content_pole, predict, score as cp_score

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
harmful_df = pd.read_csv(REPO_ROOT / "data" / "harmbench_filtered_250.csv")
neutral_df = pd.read_csv(REPO_ROOT / "data" / "neutral_set_300.csv")
splits_df  = splits.load_frozen_splits(seed=42)

def get_texts(split_name):
    rows = splits_df[splits_df["split"] == split_name]
    h_ids = set(rows[rows["label"] == "harmful"]["PromptID"])
    n_ids = set(rows[rows["label"] == "neutral"]["PromptID"])
    h_texts = harmful_df[harmful_df["BehaviorID"].isin(h_ids)]["Behavior"].tolist()
    n_texts = neutral_df[neutral_df["PromptID"].isin(n_ids)]["Prompt"].tolist()
    texts  = h_texts + n_texts
    labels = ["harmful"] * len(h_texts) + ["neutral"] * len(n_texts)
    return texts, labels

train_texts, train_labels = get_texts("train")
test_texts,  test_labels  = get_texts("test")

# Sub-test B items
_, stb_n_texts, stb_n_labels = load_neutral_arm_items()
_, stb_h_texts, stb_h_labels = load_harmful_arm_clean_items()
stb_combined_texts  = stb_n_texts  + stb_h_texts
stb_combined_labels = stb_n_labels + stb_h_labels
# Binary numeric labels for AUROC (harmful=1, neutral=0)
label_to_int = {"harmful": 1, "neutral": 0}
test_labels_int   = [label_to_int[l] for l in test_labels]
stb_labels_int    = [label_to_int[l] for l in stb_combined_labels]

# ---------------------------------------------------------------------------
# Point 3: TF-IDF AUROC
# ---------------------------------------------------------------------------
print("=" * 68)
print("POINT 3: TF-IDF AUROC")
print("=" * 68)

vec = TfidfVectorizer(ngram_range=(1, 2))
X_tr = vec.fit_transform(train_texts)
X_te = vec.transform(test_texts)
X_stb = vec.transform(stb_combined_texts)

clf_tfidf = LogisticRegression(max_iter=1000, random_state=42)
clf_tfidf.fit(X_tr, train_labels)

# Sub-test A test split
preds_te  = clf_tfidf.predict(X_te)
proba_te  = clf_tfidf.predict_proba(X_te)
acc_te    = (np.asarray(preds_te) == np.asarray(test_labels)).mean()
# AUROC: probability of class 'harmful' (classes_ is sorted alphabetically)
harm_idx = list(clf_tfidf.classes_).index("harmful")
auroc_te  = roc_auc_score(test_labels_int, proba_te[:, harm_idx])

# Sub-test B combined arm
proba_stb = clf_tfidf.predict_proba(X_stb)
auroc_stb = roc_auc_score(stb_labels_int, proba_stb[:, harm_idx])
preds_stb = clf_tfidf.predict(X_stb)
acc_stb   = (np.asarray(preds_stb) == np.asarray(stb_combined_labels)).mean()

# Sub-test B neutral arm only (for comparison with reported 85.4%)
X_stb_n  = vec.transform(stb_n_texts)
preds_n  = clf_tfidf.predict(X_stb_n)
acc_stb_n = (np.asarray(preds_n) == np.asarray(stb_n_labels)).mean()

print(f"\nSub-test A test split:")
print(f"  Accuracy:    {acc_te:.1%}")
print(f"  AUROC:       {auroc_te:.3f}")
print(f"\nSub-test B neutral arm only (should match reported 85.4%):")
print(f"  Accuracy:    {acc_stb_n:.1%}  ({int(round(acc_stb_n*48))}/48)")
print(f"\nSub-test B combined arm (48 neutral + 6 harmful, N=54):")
print(f"  Accuracy:    {acc_stb:.1%}")
print(f"  AUROC:       {auroc_stb:.3f}")
print(f"  (AUROC direction: >0.5 means TF-IDF assigns higher probability to harmful items)")

# Compare: activation methods had AUROC 0.872-0.979 on the same 54-item set
print(f"\nFor reference: activation methods AUROC range on same N=54: 0.872-0.979")
print(f"TF-IDF AUROC on sub-test B: {auroc_stb:.3f}")

# ---------------------------------------------------------------------------
# Point 5: Probe refit with StandardScaler + C sweep
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 68)
print("POINT 5: PROBE REFIT (StandardScaler + C sweep on val)")
print("=" * 68)

results_csv = REPO_ROOT / "Pipeline_v2" / "method_comparison_results.csv"
res_df = pd.read_csv(results_csv)

TARGETS = [
    ("gpt2-medium",   "raw"),
    ("qwen2.5-1.5b",  "chat"),
    ("llama-3.2-3b",  "raw"),
]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

def load_ids_and_labels(split_name):
    rows = splits_df[splits_df["split"] == split_name]
    h_ids = list(rows[rows["label"] == "harmful"]["PromptID"])
    n_ids = list(rows[rows["label"] == "neutral"]["PromptID"])
    all_ids = h_ids + n_ids
    labels  = ["harmful"] * len(h_ids) + ["neutral"] * len(n_ids)
    return all_ids, labels

train_ids, train_labels_probe = load_ids_and_labels("train")
val_ids,   val_labels_probe   = load_ids_and_labels("val")
test_ids,  test_labels_probe  = load_ids_and_labels("test")

# Sub-test B combined arm IDs
stb_n_ids, _, _ = load_neutral_arm_items()
stb_h_ids, _, _ = load_harmful_arm_clean_items()
stb_ids    = stb_n_ids + stb_h_ids
stb_labels_probe = stb_n_labels + stb_h_labels

def probe_experiment(model_key, fmt):
    # Get content_pole row for this model/fmt
    row = res_df[(res_df["model"] == model_key) &
                 (res_df["formatting"] == fmt) &
                 (res_df["method"] == "content_pole")]
    if row.empty:
        print(f"\n  [{model_key}/{fmt}] No content_pole row in results CSV — skipping")
        return
    layer   = int(row.iloc[0]["layer"])
    pooling = row.iloc[0]["pooling_variant"]

    def load_matrix(ids):
        return activation_store.load_layer_matrix(
            CACHE_KAGGLE, model_key, fmt, pooling, layer, ids
        )

    try:
        X_train = load_matrix(train_ids)
        X_val   = load_matrix(val_ids)
        X_test  = load_matrix(test_ids)
        X_stb   = load_matrix(stb_ids)
    except Exception as e:
        print(f"\n  [{model_key}/{fmt}] Failed to load activations: {e}")
        return

    # Fit StandardScaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_stb_s   = scaler.transform(X_stb)

    # C sweep on val
    best_C, best_val_acc = None, -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
        clf.fit(X_train_s, train_labels_probe)
        val_acc = (np.asarray(clf.predict(X_val_s)) == np.asarray(val_labels_probe)).mean()
        if val_acc > best_val_acc:
            best_val_acc, best_C = val_acc, C

    # Refit on train only (not train+val — consistent with pipeline: train only)
    clf_best = LogisticRegression(C=best_C, max_iter=2000, random_state=42)
    clf_best.fit(X_train_s, train_labels_probe)

    # Sub-test B evaluation
    stb_preds = clf_best.predict(X_stb_s)
    stb_acc   = (np.asarray(stb_preds) == np.asarray(stb_labels_probe)).mean()

    # AUROC on sub-test B combined arm
    harm_idx_probe = list(clf_best.classes_).index("harmful")
    proba_stb = clf_best.predict_proba(X_stb_s)
    stb_auroc = roc_auc_score(stb_labels_int, proba_stb[:, harm_idx_probe])

    # Neutral arm accuracy only (for direct comparison with reported numbers)
    X_stb_n_s = X_stb_s[:len(stb_n_ids)]
    preds_n   = clf_best.predict(X_stb_n_s)
    stb_n_acc = (np.asarray(preds_n) == np.asarray(stb_n_labels)).mean()
    wi = wilson_interval(int(round(stb_n_acc * len(stb_n_ids))), len(stb_n_ids))
    wi_lo, wi_hi = wi["wilson_low"], wi["wilson_high"]

    # Also run unstandardized C=1 (current default) for comparison
    clf_default = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_default.fit(X_train, train_labels_probe)
    preds_default_n = clf_default.predict(X_test)
    stb_n_preds_def = clf_default.predict(X_stb[:len(stb_n_ids)])
    stb_n_acc_def   = (np.asarray(stb_n_preds_def) == np.asarray(stb_n_labels)).mean()

    print(f"\n  [{model_key}/{fmt}]  layer={layer}  pooling={pooling}")
    print(f"    C sweep: best C={best_C}  (val acc at best: {best_val_acc:.1%})")
    print(f"    Unstandardized C=1.0 default  — sub-test B neutral accuracy: {stb_n_acc_def:.1%}")
    print(f"    Standardized + best C={best_C:<5} — sub-test B neutral accuracy: {stb_n_acc:.1%}  "
          f"Wilson [{wi_lo:.1%}, {wi_hi:.1%}]")
    print(f"    Standardized probe — sub-test B COMBINED AUROC: {stb_auroc:.3f}")
    delta = stb_n_acc - stb_n_acc_def
    print(f"    Delta from refit: {delta:+.1%}  ({'better' if delta > 0 else 'worse' if delta < 0 else 'unchanged'})")

for model_key, fmt in TARGETS:
    probe_experiment(model_key, fmt)

print("\ndone")
