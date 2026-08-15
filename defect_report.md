# Persona-Vector-Study — Defect Report

**Repo:** github.com/Fjord-H/Persona-Vector-Study (main, 48 commits)
**Date:** 2026-08-11
**Basis:** All findings verified against the **public repository**, not against local
working copies. Notebooks fetched from `raw.githubusercontent.com`; artifacts fetched
from `data/vectors/`. Where a local upload differs from the public file, the public file
is treated as authoritative and this is noted.

Severity: **S1** invalidates a headline claim · **S2** invalidates a supporting claim or
blocks reproduction · **S3** documentation or hygiene.

---

## S1-01 — The 92.5% result was measured on training data

**Location:** `notebooks/02_self_monitoring_experiment.ipynb`, code cells 5 and 12
(public file byte-identical to local copy).

**Evidence:** Cell 12 is headed `# Completely NEW queries for validation`. String
comparison against the training lists in cell 5:

| | overlap |
|---|---|
| validation safe | 19 / 20 verbatim in train |
| validation dangerous | 20 / 20 verbatim in train |
| total | **39 / 40 = 97.5%** |

Sole exception: `"What is photosynthesis?"` vs training item `"Explain photosynthesis"`
(near-duplicate).

**Invalidates:** The README's account of its own central failure. README §"Reality Check"
attributes the 92.5% → 44-61% collapse to *"overfitting to a tiny test set."* The
mechanism was train/test contamination, not test-set size. 92.5% is training accuracy
reported as validation accuracy.

**Verify:**
```python
import json, re
nb = json.load(open("notebooks/02_self_monitoring_experiment.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
tr = set(re.findall(r'"(.*?)"', re.search(r'safe_queries = \[(.*?)\]', src, re.S).group(1)))
val = re.findall(r'\("(.*?)",\s*"Safe"\)', src)
print(sum(q in tr for q in val), "/", len(val))
```

---

## S1-02 — Both canonical layers were selected from a sample of size 2

**Location:** `notebooks/02_self_monitoring_experiment.ipynb` cell 8;
`notebooks/02a_qwen_content_vectors.ipynb` cell 8. Both public files byte-identical to
local copies.

**Evidence (nb 02):**
```python
test_safe   = get_activation("What is 2+2?", layer)
test_danger = get_activation("How to make poison?", layer)
separation  = abs(safe_sim - danger_sim)
best_layer  = max(layer_results.keys(), key=lambda k: layer_results[k]['separation'])
```
One safe probe, one dangerous probe. `02a` uses the same two probes with a different
formula.

**Invalidates:** GPT-2 "Layer 23" and Qwen "Layer 1" as empirical findings. Notebook
`02a` cell 10 prints `KEY INSTRUCTION: Instruction tuning moves safety from Layer 23 ->
Layer 1!` — this is the documented origin of the project's RLHF claim, and it rests on
two n=2 measurements. Those indices were then hardcoded into `04-improved-vector-
training.ipynb` (`layer_idx=23`, `layer_idx=1`) and propagated into the published
vector artifacts.

---

## S1-03 — "Separation" denotes different quantities in different notebooks; the README table compares them directly

**Location:** four definitions across the repo.

| notebook | expression | units |
|---|---|---|
| 02 (cell 8) | `abs(safe_sim - danger_sim)` on 2 probes | ad hoc |
| 02a (cell 8) | `abs(s2s - s2d) + abs(d2d - d2s)` on 2 probes | ad hoc |
| 04 (cell 3) | `torch.cosine_similarity(safe_vec, danger_vec, dim=1)` | **cosine** |
| 06 (cell 3, public) | `1 - torch.cosine_similarity(safe_vec, danger_vec, dim=1)` | **1 − cosine** |

**Evidence from the published artifacts** (`data/vectors/`):

| artifact | stored `separation` | source notebook | cosine (common units) |
|---|---|---|---|
| `gpt2_vectors_200.pkl` | 0.99945045 | 04 | 0.99945 |
| `qwen_vectors_200.pkl` | 0.98974609 | 04 | 0.98975 |
| `llama_results.pkl` (layer 7) | 0.00048828 | 06 | 0.99951 |

**Invalidates:** The README results table (GPT-2 0.997 / Qwen 0.997 / Llama 0.0005) and
the accompanying claim *"Llama's representations are more entangled — safe and dangerous
content closer in activation space."* In common units all three models sit between 0.989
and 0.9995 cosine. **There is no cross-model separation difference.** The Llama
"entanglement" finding is a units mismatch between two notebooks.

Also void: `02a` metadata `'discovery': 'Layer 1 has 52x stronger signal than GPT-2'`,
which divides the `02a` formula by the `02` formula.

**Verify:** download the three artifacts, read the `separation` field, and compare
against the `separation =` line in the notebook that wrote each one.

---

## S1-04 — The README's separation figures do not match the repo's own artifacts

**Location:** README results table vs `data/vectors/*.pkl`.

| model | README | published artifact |
|---|---|---|
| GPT-2 | 0.997 | 0.99945 |
| Qwen | 0.997 | 0.98975 |
| Llama | 0.0005 | 0.00049 ✓ |

Only the Llama figure reproduces. The GPT-2 and Qwen figures are not derivable from any
published file.

---

## S1-05 — The published vectors are not the vectors behind the headline results

**Location:** `data/vectors/gpt2_vectors_200.pkl` (`'layer': 23`),
`data/vectors/qwen_vectors_200.pkl` (`'layer': 1`).

The README's headline table reports **GPT-2 Layer 0** and **Qwen Layer 27**. The
published artifacts are for layers 23 and 1 — the n=2-selected layers from S1-02. The
vectors corresponding to the reported results are not in the repository.

---

## S1-06 — Threshold is optimised against test labels, in all three evaluation notebooks

**Location:** public `04` cell 9, public `05` cells 3–4, public `06` cell 3.

**Evidence (public nb 05, cell 3):**
```python
thresholds = np.linspace(scores.min(), scores.max(), 100)
best_acc = 0
for t in thresholds:
    acc = (predictions == test_labels).mean()
    if acc > best_acc:
        best_acc = acc
return best_acc, best_thresh
```
Layer selection compounds it: `best_gpt2_acc = gpt2_df['accuracy'].max()` selects the
reported layer by maximum test accuracy.

**Measured cost:** small. Refitting the threshold on a validation half and scoring the
held-out half gives GPT-2 Layer 0 **63.79% ± 1.30** against the reported 64.22% — about
0.4 points. The README's claim that *"threshold optimization adds ~5-10% accuracy"* is
measuring against `threshold=0`, which is a calibration effect, not leakage. The defect
is real but its magnitude is minor; the layer-selection component matters more.

---

## S1-07 — No baseline; a bag-of-words classifier beats every result in the study

**Location:** absent from the repo. No lexical, random-direction, or length baseline
appears in any notebook.

**Measured** (TF-IDF 1-2 grams + logistic regression, on the repo's own
`train_test_split.pkl` and `dataset_2000.pkl`):

| method | training data | accuracy on the 1800 |
|---|---|---|
| TF-IDF + LogReg | the same 200 train examples | **66.39%** |
| TF-IDF + LogReg | 5-fold CV on the 1800 | **76.17% ± 2.07** |
| GPT-2 L0 activation vectors, honest val/test | same 200 | 63.79% ± 1.30 |
| Llama L14 activation vectors, honest val/test | same 200 | 64.74% ± 1.38 |

**Invalidates:** the README's summary claim that *"self-monitoring is viable for safety
applications."* Given identical training data, bag-of-words outperforms every
activation-based result reported. No activation method in the study clears the trivial
baseline.

---

## S2-01 — The RLHF claim is not licensed by the design

**Location:** README §"What This Reveals About RLHF"; model loading in `01a`, `04`, `06`.

The models actually loaded are `Qwen/Qwen2.5-1.5B-Instruct` and
`meta-llama/Llama-3.2-3B-Instruct`. The design is one non-instruction-tuned model
(GPT-2 Medium) against two instruction-tuned models, varying simultaneously in
architecture, parameter count, tokenizer, pretraining corpus and post-training recipe.
n=3 with every variable confounded.

The controlled experiment is a base/instruct pair of the same model family
(e.g. Llama-3.2-3B vs Llama-3.2-3B-Instruct).

Note: the README's own model list is correct ("Qwen 1.5B-Instruct", "Llama 3.2
3B-Instruct"); the results table and the surrounding prose treat GPT-2 as the base
condition without acknowledging the remaining confounds.

---

## S2-02 — Layer coverage is asymmetric at exactly the point of comparison

| model | layers swept | source |
|---|---|---|
| GPT-2 | 0–23 (all 24) | nb 05 |
| Qwen | `range(1, 29)` — **layer 0 never tested** | nb 02a cell 8 |
| Llama | `[0, 1, n//4, n//2, 3n//4, n-2, n-1]` = 7 of 28 | nb 06 cell 4 |

GPT-2's canonical answer is Layer 0. Layer 0 is the one layer never measured for Qwen.
The README's cross-model layer comparison and its "U-shaped curve" claim cannot be made
from this data.

---

## S2-03 — The reported Llama optimum is not stable

Re-scoring the saved per-example scores with threshold fitted on a validation half over
200 random splits:

| layer | README | honest val/test |
|---|---|---|
| 7 | 66.3% | 63.86% ± **6.45** |
| 14 | — | **64.74% ± 1.38** |

Layer 7's standard deviation is roughly five times every other layer's, and Layer 14
outperforms it. "Layer 7 (middle layer) — 66.3%" is not supported.

---

## S2-04 — Mean pooling injects a dominant length channel

Correlation between the classifier score and raw character length of the query, computed
from the stored per-example scores:

| layer | r(score, char_len) | r(score, label) |
|---|---|---|
| 0 | −0.690 | −0.353 |
| 6 | **−0.911** | −0.117 |
| 12 | −0.890 | −0.188 |
| 23 | −0.701 | −0.203 |

At layer 6, string length explains ~83% of score variance; the label explains ~1.4%.

This is **intrinsic to mean pooling** (averaging more tokens pulls a sequence toward the
corpus mean), not caused by padding. It does not inflate accuracy here — the dataset is
length-balanced (mean 79.4 vs 79.3 chars; r(label, length) = −0.001, p = 0.96; logistic
regression on length alone = 49.9%) — but it dominates the signal. Masked pooling will
not remove it; centring, standardisation, last-token pooling, or regressing length out
of the score are required.

---

## S2-05 — Padding contaminates reference vectors but not query vectors

**Location:** public `04` cell 3, public `06` cell 3.

Reference vectors are built with `tokenizer(batch, padding=True)` followed by
`.mean(dim=1)` with no attention mask, so pad/EOS embeddings enter the class means. The
test loop tokenizes one query at a time and is therefore unpadded. Class means and the
queries compared against them are computed under different pooling regimes. GPT-2 sets
`pad_token = eos_token`, so the contaminant is the EOS embedding.

---

## S2-06 — GPT-2 cosine similarity is dominated by outlier dimensions

From `gpt2_content_vectors.pkl`: dimension 268 alone holds 59.2% of the squared norm;
the top five dimensions hold 97.2%. Ablating them moves separation from 0.0055 to 0.1126
(20×). Cosine on raw mean-pooled GPT-2 states is measuring activation scale, not content.
Qwen is far less affected (top-1 = 16.1%, top-5 = 49.2%).

---

## S2-07 — Dataset composition does not match the stated research question

`data/vectors/dataset_2000.pkl`, `source_breakdown`:

| source | safe | dangerous |
|---|---|---|
| manual | 50 | 50 |
| RealToxicityPrompts | 475 | 475 |
| ToxiGen | 475 | 475 |

5% of the corpus is manually written capability requests ("How to make a bomb?"); 95% is
toxic-language material. The README describes the task as detecting "dangerous content"
and illustrates it exclusively with capability requests. Toxic language and harmful-
capability requests are different constructs. This also plausibly explains S1-07, since
toxic language is lexically obvious.

The dataset itself is clean: 1000 unique strings per class, zero duplicates, zero strings
labelled both ways.

---

## S2-08 — Key results are not reproducible from the repository

- `notebooks/03-extended-evaluation.ipynb` is listed in the README repository structure
  and its results are cited throughout, but returns **404**. Probed six filename
  variants; none exist. The notebook producing the project's signature honesty result
  (92.5% → 44-61%) is not published.
- `all_layers_proper.pkl` — the artifact holding per-example scores for every layer,
  backing the canonical layer analysis — is not published.
- README references `data/vectors/llama_results.pkl` (present, 629 bytes, summary only,
  no per-example scores).

---

## S3-01 — Three-way state divergence

The public notebook `06-llama-3-2-3b-evaluation.ipynb` computes
`safe_vec = safe_activations.mean(dim=0, keepdim=True)` with the comment
`# Simple unweighted mean (no softmax weighting!)`. The local working copy
`06-llama-3-2-3b-evaluation_real.ipynb` (not in the repo) uses
`torch.softmax(cosine_similarity(...) * 5)` weighting. The project handover document
lists the Llama weighting asymmetry as open issue #1.

Three sources, three states. Separately, the README already publishes the weighting
ablation (GPT-2 59.7 / 59.4 / 59.6; Qwen 71.9 / 72.3 / 72.4, spread < 0.5%), so the
handover document's #1 priority is already resolved in the README. Measured directly on
the vectors, weighted vs unweighted differ by cosine 0.99999554 — the scheme is
numerically a no-op.

## S3-02 — README internal inconsistencies

- GPT-2 Layer 0 is reported as 66.0% in §"Comprehensive Layer Analysis" and 64.2% in the
  results table.
- Repo structure describes notebook 02 as *"Testing tone vectors for detection
  (overfitted)"*; notebook 02 is the content-vector experiment.
- Reproducibility snippet calls `torch.cosine_similarity(safe_vec, danger_vec, dim=0)` on
  tensors of shape `[1, d]`. `dim=0` is the length-1 axis; should be `dim=1`.
- Quick-start uses `git clone https://github.com/yourusername/persona-vector-study`.
- Related Work cites a DeepLearning.AI summary article rather than primary literature.

## S3-03 — Duplicate artifacts under different names

`gpt2_tone_vectors.pkl` and `gpt2_safety_vectors.pkl` contain identical tensors at all
three layers, differing only in a metadata key.

## S3-04 — Undocumented dtype inconsistency

`gpt2_vectors_200.pkl` stores float32; `qwen_vectors_200.pkl` stores float16. Not
documented; half precision is material when the quantity of interest is a cosine near
0.99.

---

## Not defects — checked and clean

- `dataset_2000.pkl`: no duplicates, no label conflicts.
- `04` split logic (`random.sample` with `random.seed(42)`, then exclusion by
  membership) is correct and reproducible.
- No credential is exposed in the public repository. `notebooks/06-llama-3-2-3b-
  evaluation.ipynb` contains `token = "YOUR TOKEN"`; the variant containing a live token
  was never committed (404).
- Weighting ablation exists and its conclusion ("unweighted is as good") is correct.

---

## Corrections to earlier claims made during this audit

Listed so reviewers can weight the above appropriately.

| earlier claim | status |
|---|---|
| Prompt length correlates with label, driving accuracy | **Wrong.** Rejected: r = −0.001, p = 0.96; length-only classifier = 49.9%. |
| Threshold leakage costs 5–10 points | **Overstated.** Measured ≈ 0.4 points at n=1800. |
| Llama is the cleanly separated model | **Wrong, twice.** Correct reading in S1-03: all three are equally entangled. |
| A live token is exposed in the public repo | **Wrong.** Verified 404; the public notebook is sanitised. |

---

## Summary

Three headline claims are load-bearing in the README:

1. **Model accuracies of 64.2 / 74.4 / 66.3 at layers 0 / 27 / 7** — layers selected on
   test (S1-06) after an initial selection at n=2 (S1-02); the Llama optimum is unstable
   (S2-03); nothing clears a bag-of-words baseline trained on the same data (S1-07).
2. **Separation differs across models (0.997 / 0.997 / 0.0005)** — units mismatch
   (S1-03); figures not reproducible from the repo's own artifacts (S1-04).
3. **RLHF relocates where safety computation happens** — confounded design (S2-01);
   Qwen layer 0 never measured (S2-02); origin claim rests on n=2 (S1-02).

None of the three is supported by the experiments that produced it.

The dataset, the split logic, the weighting ablation, the tone-vs-content comparison as a
qualitative observation, and every stored per-example score remain usable. A methodology
and negative-results paper built on S1-01, S1-02, S1-03, S1-07 and S2-04 is supportable
from artifacts already in hand and requires no additional GPU time.
