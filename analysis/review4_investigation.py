"""
Fourth external review — Points 1 and 2 investigation.

Point 1: Does sub-test B actually isolate tone, or does something else vary?
Point 2: Is TF-IDF's sub-test A performance partly a grammatical-mood confound?
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import re

REPO_ROOT = Path(r"C:\School\Persona_Vector")
PIPELINE  = REPO_ROOT / "Pipeline_v2"
sys.path.insert(0, str(PIPELINE))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import splits

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
harmful_df = pd.read_csv(REPO_ROOT / "data" / "harmbench_filtered_250.csv")
neutral_df = pd.read_csv(REPO_ROOT / "data" / "neutral_set_300.csv")
stb_df     = pd.read_csv(REPO_ROOT / "data" / "subtest_b_neutral_tone_pairs.csv")

harmful_texts = harmful_df["Behavior"].tolist()
neutral_texts = neutral_df["Prompt"].tolist()
stb_calm      = stb_df["calm_prompt"].tolist()
stb_hostile   = stb_df["hostile_prompt"].tolist()
stb_categories = stb_df["category"].tolist()

print("=" * 68)
print("POINT 1: WHAT ELSE VARIES BETWEEN SUB-TEST A AND SUB-TEST B?")
print("=" * 68)

# ---- 1a. Length distribution -----------------------------------------------
def length_stats(texts, label):
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    print(f"\n  {label} (n={len(texts)})")
    print(f"    char length:  mean={np.mean(chars):.0f}  median={np.median(chars):.0f}  "
          f"min={min(chars)}  max={max(chars)}")
    print(f"    word count:   mean={np.mean(words):.1f}  median={np.median(words):.0f}  "
          f"min={min(words)}  max={max(words)}")

print("\n-- 1a. Length distribution --")
length_stats(neutral_texts,  "Sub-test A neutral (train+val+test)")
length_stats(harmful_texts,  "Sub-test A harmful")
length_stats(stb_calm,       "Sub-test B calm")
length_stats(stb_hostile,    "Sub-test B hostile")

# ---- 1b. Opening-word / grammatical form ------------------------------------
IMPERATIVES = {"give", "create", "write", "explain", "describe", "list", "provide",
               "develop", "make", "generate", "draft", "produce", "compose", "design",
               "build", "outline", "suggest", "tell", "show", "identify"}
QUESTION_STARTERS = {"what", "how", "why", "when", "where", "who", "which", "can",
                     "could", "would", "is", "are", "do", "does"}

def form_analysis(texts, label):
    n = len(texts)
    imp   = sum(1 for t in texts if t.split()[0].lower().rstrip(".,?") in IMPERATIVES)
    quest = sum(1 for t in texts if t.split()[0].lower().rstrip(".,?") in QUESTION_STARTERS)
    howto = sum(1 for t in texts if re.search(r'\bhow to\b', t, re.I))
    step  = sum(1 for t in texts if re.search(r'\bstep[\s-]by[\s-]step\b|\bsteps?\b', t, re.I))
    instr = sum(1 for t in texts if re.search(r'\binstructions?\b|\bguide\b|\btutorial\b', t, re.I))
    idk   = sum(1 for t in texts
                if t.split()[0].lower().rstrip(".,?") not in IMPERATIVES
                and t.split()[0].lower().rstrip(".,?") not in QUESTION_STARTERS)
    print(f"\n  {label} (n={n})")
    print(f"    imperative start:   {imp:3d} / {n}  ({100*imp/n:.0f}%)")
    print(f"    question start:     {quest:3d} / {n}  ({100*quest/n:.0f}%)")
    print(f"    other start:        {idk:3d} / {n}  ({100*idk/n:.0f}%)")
    print(f"    contains 'how to':  {howto:3d} / {n}  ({100*howto/n:.0f}%)")
    print(f"    contains 'step*':   {step:3d} / {n}  ({100*step/n:.0f}%)")
    print(f"    contains instr/guide/tutorial: {instr:3d} / {n}  ({100*instr/n:.0f}%)")

print("\n-- 1b. Grammatical form / phrasing pattern --")
form_analysis(neutral_texts, "Sub-test A neutral")
form_analysis(harmful_texts, "Sub-test A harmful")
form_analysis(stb_calm,      "Sub-test B calm")
form_analysis(stb_hostile,   "Sub-test B hostile")

# ---- 1c. Topic coverage ----------------------------------------------------
print("\n-- 1c. Topic coverage --")
print("\n  Sub-test A neutral categories:")
for cat, cnt in neutral_df["Category"].value_counts().items():
    print(f"    {cat}: {cnt}")
print("\n  Sub-test B neutral categories (4 pairs each):")
for cat, cnt in stb_df["category"].value_counts().items():
    print(f"    {cat}: {cnt} pairs")

# Does sub-test A neutral cover the SAME 6 buckets as sub-test B?
stb_cats = set(stb_df["category"].unique())
sta_cats = set(neutral_df["Category"].unique())
print(f"\n  Sub-test B categories: {sorted(stb_cats)}")
print(f"  Sub-test A categories: {sorted(sta_cats)}")
shared = stb_cats & sta_cats
print(f"  Overlap: {sorted(shared)}")
print(f"  In B not A: {sorted(stb_cats - sta_cats)}")
print(f"  In A not B: {sorted(sta_cats - stb_cats)}")

# ---- 1d. Style signature — politeness / hedging markers -------------------
POLITE_MARKERS = [
    r"\bi'?d appreciate\b", r"\bcould you\b", r"\bwould you\b",
    r"\bplease\b", r"\bkindly\b", r"\bi was wondering\b",
]
PERSONAL_MARKERS = [r"\bmy\b", r"\bi\b", r"\bme\b", r"\bwe\b"]
FRUSTRATION_MARKERS = [r"\bsick of\b", r"\bkeep\b.*\bwrong\b", r"\bi.m sick\b",
                       r"\bevery (single )?time\b", r"\bnow\b$", r"\bactual(ly)?\b"]

def marker_analysis(texts, label, markers, marker_label):
    n = len(texts)
    hits = sum(1 for t in texts if any(re.search(m, t, re.I) for m in markers))
    print(f"    {label}: {hits}/{n} ({100*hits/n:.0f}%)")

print("\n-- 1d. Style markers (first-person, politeness, frustration) --")
for label_pair in [
    (neutral_texts, "Sub-test A neutral"),
    (stb_calm, "Sub-test B calm"),
    (stb_hostile, "Sub-test B hostile"),
]:
    texts, label = label_pair
    n = len(texts)
    polite = sum(1 for t in texts if any(re.search(m, t, re.I) for m in POLITE_MARKERS))
    personal = sum(1 for t in texts if any(re.search(m, t, re.I) for m in PERSONAL_MARKERS))
    print(f"\n  {label} (n={n})")
    print(f"    polite/hedging markers:  {polite}/{n} ({100*polite/n:.0f}%)")
    print(f"    personal pronouns (my/I/me/we): {personal}/{n} ({100*personal/n:.0f}%)")

# ---- 1e. Procedural tokens from TF-IDF model in each set ------------------
PROCEDURAL_TOKENS = ["how to", "step", "steps", "step-by-step", "write", "create",
                     "give", "instructions", "guide", "tutorial", "describe", "explain"]

print("\n-- 1e. TF-IDF 'harmful'-weighted procedural tokens in each set --")
print("  (These are the tokens the reviewer flagged as form-confound, not content)")
for texts, label in [
    (neutral_texts, "Sub-test A neutral"),
    (harmful_texts, "Sub-test A harmful"),
    (stb_calm,      "Sub-test B calm"),
    (stb_hostile,   "Sub-test B hostile"),
]:
    n = len(texts)
    counts = {}
    for tok in PROCEDURAL_TOKENS:
        hits = sum(1 for t in texts if re.search(r'\b' + re.escape(tok) + r'\b', t, re.I))
        counts[tok] = hits
    top = sorted(counts.items(), key=lambda x: -x[1])[:8]
    print(f"\n  {label} (n={n}):")
    for tok, cnt in top:
        print(f"    {tok:<25} {cnt:3d}/{n} ({100*cnt/n:.0f}%)")

print("\n\n" + "=" * 68)
print("POINT 2: TF-IDF GRAMMATICAL-MOOD CONFOUND INVESTIGATION")
print("=" * 68)

# ---- 2a. Framing-pattern breakdown in sub-test A --------------------------
print("\n-- 2a. Framing patterns in sub-test A harmful vs neutral --")

IMPERATIVE_PATTERNS = {
    "starts imperative verb": lambda t: t.split()[0].lower().rstrip(".,") in IMPERATIVES,
    "contains 'how to'":      lambda t: bool(re.search(r'\bhow to\b', t, re.I)),
    "contains 'step' variant": lambda t: bool(re.search(r'\bsteps?\b|\bstep-by-step\b', t, re.I)),
    "contains 'instructions'": lambda t: bool(re.search(r'\binstructions?\b', t, re.I)),
    "contains 'write'":       lambda t: bool(re.search(r'\bwrite\b', t, re.I)),
    "contains 'create'":      lambda t: bool(re.search(r'\bcreate\b', t, re.I)),
    "contains 'give'":        lambda t: bool(re.search(r'\bgive\b', t, re.I)),
    "contains 'guide'":       lambda t: bool(re.search(r'\bguide\b', t, re.I)),
    "starts 'what is'":       lambda t: bool(re.match(r'^what is\b', t, re.I)),
    "starts 'how do'":        lambda t: bool(re.match(r'^how do\b', t, re.I)),
}

for texts, label in [(harmful_texts, "Harmful"), (neutral_texts, "Neutral")]:
    n = len(texts)
    print(f"\n  {label} (n={n}):")
    for name, fn in IMPERATIVE_PATTERNS.items():
        cnt = sum(1 for t in texts if fn(t))
        print(f"    {name:<35} {cnt:3d}/{n} ({100*cnt/n:.0f}%)")

# ---- 2b. TF-IDF ablation — refit with procedural stopwords ----------------
print("\n-- 2b. TF-IDF ablation: add form/procedural tokens to stopword list --")

os.environ["PV2_DATA_DIR"]    = str(REPO_ROOT / "data")
os.environ["PV2_SPLITS_JSON"] = str(REPO_ROOT / "data" / "splits.json")

splits_df = splits.load_frozen_splits(seed=42)

# Build train/test text lists exactly as the pipeline does
def get_texts_labels(df, split_name):
    split_rows = splits_df[splits_df["split"] == split_name]
    harm_ids = set(split_rows[split_rows["label"] == "harmful"]["PromptID"])
    neut_ids = set(split_rows[split_rows["label"] == "neutral"]["PromptID"])

    harm_texts = harmful_df[harmful_df["BehaviorID"].isin(harm_ids)]["Behavior"].tolist()
    neut_texts = neutral_df[neutral_df["PromptID"].isin(neut_ids)]["Prompt"].tolist()
    texts  = harm_texts + neut_texts
    labels = ["harmful"] * len(harm_texts) + ["neutral"] * len(neut_texts)
    return texts, labels

train_texts, train_labels = get_texts_labels(splits_df, "train")
# Combine val into train for single-fit (mirroring pipeline which uses train split only)
val_texts, val_labels = get_texts_labels(splits_df, "val")
test_texts, test_labels = get_texts_labels(splits_df, "test")

all_train_texts  = train_texts + val_texts
all_train_labels = train_labels + val_labels

def fit_score_tfidf(extra_stopwords=None, label=""):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    sw = list(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        sw = list(set(sw) | set(extra_stopwords))
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=sw)
    X_tr = vec.fit_transform(all_train_texts)
    X_te = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, all_train_labels)
    preds = clf.predict(X_te)
    acc = (preds == np.asarray(test_labels)).mean()
    n_features = X_tr.shape[1]
    print(f"  {label:<45} accuracy={acc:.1%}  (n_features={n_features})")
    return acc, clf, vec

print()
baseline_acc, clf_base, vec_base = fit_score_tfidf(label="Baseline (sklearn default stopwords)")

FORM_TOKENS_MINIMAL = [
    "how", "to", "step", "steps", "write", "create", "give",
    "instructions", "guide", "tutorial", "plan", "make",
]
FORM_TOKENS_FULL = FORM_TOKENS_MINIMAL + [
    "provide", "develop", "describe", "explain", "list", "outline",
    "design", "build", "generate", "draft", "produce", "compose",
    "suggest", "how to",  # bigrams handled at ngram level
]

minimal_acc, _, _ = fit_score_tfidf(FORM_TOKENS_MINIMAL,
    label="+ minimal procedural stopwords (12 tokens)")
full_acc, _, _ = fit_score_tfidf(FORM_TOKENS_FULL,
    label="+ full form/procedural stopwords (22 tokens)")

print(f"\n  Baseline -> minimal ablation delta: {100*(minimal_acc - baseline_acc):+.1f} pp")
print(f"  Baseline -> full ablation delta:    {100*(full_acc - baseline_acc):+.1f} pp")

# ---- 2c. Cross-check: how well do purely content tokens separate classes? -
print("\n-- 2c. Top form vs. content tokens in baseline TF-IDF model --")
print("  (form = in FORM_TOKENS_MINIMAL; content = everything else)")
feature_names = vec_base.get_feature_names_out()
coefs = clf_base.coef_[0]  # positive = 'neutral', negative = 'harmful'
harmful_idx = np.where(np.asarray(clf_base.classes_) == "harmful")[0][0]
# Flip sign so positive = harmful
if clf_base.classes_[0] == "neutral":
    harm_coef = -coefs
else:
    harm_coef = coefs

form_set = set(FORM_TOKENS_FULL)
form_mask = np.array([any(tok in f.lower() for tok in form_set) for f in feature_names])
content_mask = ~form_mask

top_harm_form    = sorted(zip(feature_names[form_mask], harm_coef[form_mask]),
                          key=lambda x: -x[1])[:8]
top_harm_content = sorted(zip(feature_names[content_mask], harm_coef[content_mask]),
                          key=lambda x: -x[1])[:12]

print("\n  Top FORM-token features (harmful direction):")
for f, c in top_harm_form:
    print(f"    {f:<30} {c:+.4f}")
print("\n  Top CONTENT-token features (harmful direction):")
for f, c in top_harm_content:
    print(f"    {f:<30} {c:+.4f}")

# ---- 2d. TF-IDF on sub-test B with both stopword regimes ------------------
print("\n-- 2d. TF-IDF on sub-test B (baseline vs. ablated) --")

stb_all    = stb_calm + stb_hostile
stb_labels = ["neutral"] * len(stb_calm) + ["neutral"] * len(stb_hostile)

def stb_accuracy(vec, clf, label=""):
    X_stb = vec.transform(stb_all)
    preds = clf.predict(X_stb)
    correct = (preds == np.asarray(stb_labels)).mean()
    calm_correct    = (np.asarray(clf.predict(vec.transform(stb_calm))) == "neutral").mean()
    hostile_correct = (np.asarray(clf.predict(vec.transform(stb_hostile))) == "neutral").mean()
    print(f"  {label}")
    print(f"    overall: {correct:.1%}  calm: {calm_correct:.1%}  hostile: {hostile_correct:.1%}")

_, clf_bl,  vec_bl  = fit_score_tfidf(label="[ignored baseline refit]")
_, clf_min, vec_min = fit_score_tfidf(FORM_TOKENS_MINIMAL, label="[ignored minimal refit]")
_, clf_ful, vec_ful = fit_score_tfidf(FORM_TOKENS_FULL, label="[ignored full refit]")

print()
stb_accuracy(vec_bl,  clf_bl,  "Baseline stopwords (sub-test B):")
stb_accuracy(vec_min, clf_min, "Minimal ablation   (sub-test B):")
stb_accuracy(vec_ful, clf_ful, "Full ablation      (sub-test B):")

print("\ndone")
