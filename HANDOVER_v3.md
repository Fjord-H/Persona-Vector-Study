# Persona Vector Study — v3 Handover Brief
**For: next session / follow-up experiments**
**Written: 2026-09-24, end of the v2 review thread**

---

## Where v2 left off

v2 is complete and stable. Pipeline code: `Pipeline_v2/`. All results committed to
`origin/main`. Five rounds of external review were addressed across August–September
2026; no outstanding review comments. The repo is in a clean, reportable state.

**Read the README's v2 block first** — it is the authoritative summary. The key numbers
in brief:

- **Sub-test A** (content-varying, tone-fixed, N=550): activation methods 93–100%,
  TF-IDF 93.4%. Largely lexical. Activation geometry adds little on content-varying data.
- **Sub-test B** (phrasing-varied, N=24 pairs): qwen-instruct 95.8%, TF-IDF 85.4%. Gap
  is directional, not significant at N=24 (McNemar p=0.062–0.180, pair-level CIs
  overlap). AUROC comparable (0.920 vs. 0.872–0.979) — threshold-transfer explains the
  accuracy gap, not representations.
- **Threshold recalibration** (`analysis/review5_threshold_recalibration.py`, seed=42):
  18 labeled sub-test B items recovered accuracy from 13.9%→100% (qwen content_pole)
  and 58.3%→91.7% (llama probe). Results are striking but rest on one calibration split.
- **Sub-test B confound** (`analysis/review4_investigation.py`): sub-test B varies
  register and length alongside phrasing tone — 0% imperative in B-calm vs. 73% in A-neutral,
  2× length difference, 100% personal pronouns in B-hostile. Framing updated from
  "tone-invariance" to "register/distribution-shift robustness."

The defensible standing claim: under one A→B distribution shift, one instruct model's
default threshold transferred better than TF-IDF's. The mechanism (better
representations vs. better threshold calibration by luck) is not resolved.

---

## The three next-phase experiments

### Phase 1 — Multi-seed recalibration validation (highest priority, cheap)

**What and why:** The recalibration result (review5) used a single calibration/held-out
split (seed=42, 6 neutral pairs + 6 harmful in calibration, 18 neutral pairs held out).
The result is internally consistent but a N=18 calibration and N=36 held-out is sensitive
to which pairs happen to land where. One seed is not a robustness claim.

**What to do:** Extend `analysis/review5_threshold_recalibration.py` to loop over
N=50–100 random seeds for the calibration/held-out split. For each seed, record:
recalibrated accuracy on held-out, original accuracy on held-out, oracle accuracy on
full arm. Report mean ± std across seeds for each metric and for the delta
(recalibrated − original). Also test whether calibration slice size matters: run the
same sweep at 4, 6, 8, and 12 neutral-pair calibration sizes. Report the minimum slice
size that gives a stable recovery.

**Deliverable:** A results table (mean ± std across seeds, by slice size) added to the
README as an extension of the "Does threshold recalibration recover the gap?" section.
If recovery is stable across seeds, the claim strengthens substantially. If it's
variable, report that honestly — it would suggest the single-seed result was favorable
by chance and the threshold-calibration story needs more data, not less.

**Why cheap:** uses only cached activations already on disk (`pv2_cache_kaggle/`), no
GPU. Runtime is seconds. No new data collection needed.

---

### Phase 2 — Confound-free tone test (medium effort, high value)

**What and why:** Sub-test B was supposed to isolate tone (calm vs. hostile phrasing)
as the independent variable. It doesn't — the two phrasing variants differ in register,
length (~2×), grammatical form, and personal pronoun density as well as tone
(documented in `analysis/review4_investigation.py` and the README "What sub-test B
actually tests" section). The current sub-test B tests register/distribution-shift
robustness, which is an interesting thing to test, but it's not the tone test it was
designed to be.

**What to do:** Build a new tone-paraphrase set, call it sub-test B tone-controlled,
where calm and hostile variants are matched on:
- **Length** (within ±15%): generate/edit pairs so the phrasing variants are
  approximately the same character count.
- **Grammatical form**: both variants should use the same form (both imperative, or
  both question, not one of each).
- **Personal pronouns**: if the hostile variant uses "my/I/me," the calm variant should
  also reference the requester, not be impersonalized.
- **Topic and content**: same as current sub-test B (fixed underlying request, only
  tone varies).

N=50 pairs minimum to reach useful statistical power for McNemar. The current N=24
neutral pairs gives CIs too wide for significance; N=50 gives roughly ±14 pp Wilson CI
on a 95.8%-like result, which starts to be informative.

**Data authoring:** hand-authored or LLM-assisted paraphrase generation with explicit
constraints checked post-hoc by `analysis/review4_investigation.py`-style script. The
check script already computes: imperative rate, question rate, polite marker rate, mean
char length, personal pronoun rate — run it on the new pairs before committing them.
Flag any pair where calm and hostile differ by more than 15% on length or by more than
2 of the 5 register markers.

**Deliverable:** New CSV (`data/subtest_b_tone_controlled.csv`, same schema as
`data/subtest_b_neutral_tone_pairs.csv`), a build/validation script, a MANIFEST, and
updated README section replacing the current sub-test B result with the new one. If
the controlled result agrees with the current result (instruct model more robust), the
claim becomes better-grounded. If it disagrees, that's the more interesting finding.

---

### Phase 3 — Keyword-free harmful-content test (heaviest, requires GPU)

**What and why:** Sub-test A's lexical saturation (TF-IDF at 93.4%) means it doesn't
test whether activation methods capture semantic content that surface form misses — it
just tests whether they can separate two lexically-distinct corpora. Sub-test B v3 was
built specifically to fix this: 5,200 items, TF-IDF at 69.1% overall, all individual
mutation sub-groups below 80%.

**Sub-test B v3 exists and is ready:** `data/subtest_b_v3/subtest_b_v3.jsonl`,
fully documented in `data/subtest_b_v3/subtest_b_v3_MANIFEST.md`. The dataset covers
7 surface-form mutation types (caesar, morse, atbash, ascii, misspellings, slang,
role_play) applied to HarmBench/neutral_set_300 content, plus XSTest and OR-Bench
hard-paired items (harmful-looking benign vs. genuinely harmful). Build script is
`data/subtest_b_v3/build_subtest_b_v3.py`; the build is fully deterministic (seed=42).
Caveat: OR-Bench length-only accuracy is 57.6% (flagged threshold is 55%) — documented
in the MANIFEST, residual driven by terse phrasing in toxic items, not a blocker.

**What to do:** Run the v2 pipeline's activation extraction on the 5,200 v3 items.
The pipeline code in `Pipeline_v2/src/eval/` needs a new eval module
(`src/eval/subtest_b_v3.py`) that loads from the JSONL format, applies the frozen
method configurations already selected on sub-test A (same content_pole layer/pooling),
and evaluates per mutation type and per source. The extraction notebook
(`Pipeline_v2/notebooks/`) can be extended to include a v3 extraction cell.

**Compute:** 5,200 items × all layers × 2 pooling variants × models = roughly 3–4×
the current sub-test A extraction cost (~1 GPU-hour per model on T4). Run GPT-2 first
as a smoke test (cheapest, CPU-feasible). Cache layout is already supported: the v2
`activation_store.py` / `checkpoint.py` infrastructure accepts any prompt_id list.
Use new item_ids from the JSONL `source_id + mutation` field.

**Key question:** do activation methods score above TF-IDF's 69.1% on any mutation
sub-group? Cipher mutations (morse, ascii, atbash, caesar) are where activation methods
have the strongest theoretical advantage — TF-IDF sees garbage tokens; if the model
processes them semantically, its activations should still separate classes. That's the
test sub-test A never ran.

**Deliverable:** Accuracy table by mutation type for content_pole (and probe if time
allows), added to the README as a new "Sub-test B v3 (keyword-free test)" section.

---

## Repo layout — what's where

```
README.md                          — authoritative result summary (v2 complete)
HANDOVER.md                        — v1 audit handover (historical, keep as-is)
HANDOVER_v3.md                     — this file
defect_report.md                   — v1 defects, 12 items (historical)
portfolio_case_study.md            — write-up for job applications (keep synced with README)
v2_pipeline_SPEC.md                — original v2 design spec

Pipeline_v2/
  src/                             — pipeline source (complete, tested)
  method_comparison_results.csv    — all v2 results (do not edit by hand)
  notebooks/                       — extraction notebooks (Kaggle-ready)

analysis/
  audit.py                         — v1 audit reproduction code
  review4_investigation.py         — sub-test B register-shift analysis (Point 1/2)
  review4_tfidf_auroc_probe_refit.py — TF-IDF AUROC + probe refit (Point 3/5)
  review5_threshold_recalibration.py — threshold recalibration (Phase 1 starting point)

data/
  harmbench_filtered_250.csv       — harmful class (sub-test A, v3 Type 1 source)
  neutral_set_300.csv              — neutral class (sub-test A, v3 Type 1 source)
  subtest_b_neutral_tone_pairs.csv — current sub-test B neutral arm (N=24 pairs)
  subtest_b_harmful_tone_pairs.csv — current sub-test B harmful arm (N=9 pairs; 3 clean)
  subtest_b_MANIFEST.md            — sub-test B data provenance
  subtest_b_v3/
    subtest_b_v3.jsonl             — 5200-item keyword-free dataset (ready to use)
    subtest_b_v3_MANIFEST.md       — full build documentation + TF-IDF sanity results
    build_subtest_b_v3.py          — deterministic build script (seed=42)

pv2_cache_kaggle/pv2_cache/        — activation cache (all v2 models, ~1.3GB)
                                     — required for Phase 1; already on disk
```

---

## Process notes

- **Index.lock:** recurs intermittently; clear with `rm .git/index.lock`. `gc.auto=0`
  is set in the repo config (from 2026-09-21 session) to suppress background gc — this
  reduced but hasn't fully eliminated the issue, likely residual from crashes before
  that fix. Check that no Git GUI or editor has the repo open.

- **Activations are the bottleneck for Phase 3 only.** Phases 1 and 2 are CPU-only.
  Run Phase 1 first — it's one script extension, takes minutes, and directly strengthens
  the most interesting v2 finding. Phase 2 is data authoring (any environment). Phase 3
  needs a GPU session (Kaggle, T4, resume-safe via the existing checkpoint machinery).

- **Do not edit `method_comparison_results.csv` by hand.** It is the canonical record
  of all v2 pipeline output. New experiments (Phases 1–3) produce their own result files
  or add new sections to the README — they don't modify this file.

- **Keep README and portfolio_case_study.md in sync.** Every numerical claim added to
  the README v2 block should have a corresponding update in the portfolio doc's Results
  section. The commit history shows the pattern (five pairs of commits, one per review
  round).

- **Commit message convention in use:** verb phrase + colon + detail. Recent examples:
  "Address fourth external review: register-shift diagnosis, TF-IDF AUROC, probe refit"
  "Add threshold-recalibration experiment: does refitting on a small B sample recover the gap?"

- **Attribution line on commits:**
  `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
