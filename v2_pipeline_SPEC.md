# Persona Vector Study v2 — Pipeline Spec
**For: Claude Code, v2 extraction + method-comparison build**
**Written: 2026-09-01, following the Cowork planning thread that produced this spec**

---

## Context, in one paragraph

v1 of this project made three retracted claims (see `defect_report.md` and the corrected `README.md` at repo root — read those first, they are the authoritative record of what went wrong and why). v2 is not a patch on v1's binary safe/dangerous classifier. It is a redesigned method: instead of two labeled poles (safe vs. dangerous), v2 defines a single **neutral origin** vector from ordinary, harmless prompts, and measures new content by its distance and direction from that origin. This spec also requires building the two *older* methods (tone-pole, content-pole) as explicit comparison baselines in the same experiment, so the choice of method is settled by a measured result, not by assumption. Read the "Three methods" section below before writing any extraction code — it determines what gets cached and how.

## Data already sourced (do not re-source, do not modify)

Two CSVs exist at repo root under `data/`:

- **`data/harmbench_filtered_250.csv`** — 250 real harmful behaviors from HarmBench (Mazeika et al. 2024, MIT licensed), 100 copyright-violation rows already excluded. Columns: `Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID`. Six categories: illegal (62), misinformation_disinformation (51), cybercrime_intrusion (49), chemical_biological (41), harassment_bullying (25), harmful (22). Full provenance in `data/harmbench_filtered_250_MANIFEST.md`.
- **`data/neutral_set_300.csv`** — 300 hand-written ordinary/harmless prompts, six topic buckets (cooking_household, science_general_knowledge, technology_howto, hobbies_creative, everyday_logistics, health_wellness_basic), 50 each. Balanced between instruction-style and question-style phrasing to avoid a sentence-shape confound. Columns: `Prompt, Category, PromptID`. Full provenance in `data/neutral_set_300_MANIFEST.md`. (Note: an earlier 240-row version of this file exists as `neutral_set_240.csv` — superseded, do not use, kept only as historical record per this project's "nothing gets deleted" convention.)

**Near-duplicate check: done, clean.** A string-similarity scan (SequenceMatcher) between the 250 harmful and 300 neutral prompts found zero exact duplicates and only one borderline near-duplicate pair (ratio 0.75, a shared generic template phrase, not real content overlap). Content overlap is not a concern.

**Length-balance check: done, found a real problem, only partially fixed. This is the more important finding — read before writing extraction code.** Harmful prompts (`harmbench_filtered_250.csv`) average 96.4 characters; the neutral set was extended from 240 to 300 prompts specifically to close a length gap against this, but even after two rounds of adding longer prompts, neutral mean length is 66.4 characters, and **r(length, label) = 0.436** across the combined 550-prompt set. For comparison, v1's own dataset was safely length-balanced at r = −0.001. This is NOT solved by the data alone. Per the pooling section below, this makes length-robust pooling (centering, standardization, or last-token pooling, not raw mean pooling) a hard requirement for this project's data, not an optional ablation arm. Check r(score, length) directly on the extracted activations before trusting any method-comparison result — if raw mean-pooled scores correlate strongly with prompt length, that is v1's exact bug recurring in a new dataset, and the run should be flagged rather than reported.

## Three methods to implement and compare (this is the core of the build)

All three must run on the **same underlying content** where possible, so the comparison is fair. Each produces a "vector" and a scoring function that maps a new prompt's activation to a scalar.

**Method 1 — Tone-pole (expected to fail; included as the falsifying control).**
Vector = mean(activations on "helpful/harmless" system-prompt-conditioned generations) − mean(activations on "harmful/dishonest" system-prompt-conditioned generations), following the article's method (Chen et al., Anthropic persona vectors, arXiv:2507.21509): contrasting system prompts, model generates responses, activations pulled from generated responses not the prompt. This method needs its own small prompt set (system-prompt pairs + a batch of neutral questions to provoke responses under each condition) — this is a different input than the two CSVs above. Keep this set small (the original article used 40 questions); it does not need HarmBench-scale data since it is the known-failing control.

**Method 2 — Content-pole (v1's approach, done honestly this time).**
Vector = mean(activations on `harmbench_filtered_250.csv` prompts) − mean(activations on `neutral_set_300.csv` prompts). Classification/scoring = cosine similarity to each pole, or position relative to the midpoint between the two class means. This is structurally the v1 method; the difference from v1 is procedural: proper train/val/test split, no threshold or layer selection on test data, ever.

**Method 3 — Neutral-origin (primary candidate, Design 1).**
Origin = mean(activations on `neutral_set_300.csv` prompts) only. No harmful pole is used to construct the vector. Score for any new prompt = distance/direction of its activation from the origin (cosine distance from origin, or projection onto the origin-to-harmful-mean direction computed separately for validation, not for vector construction). The `harmbench_filtered_250.csv` set is used only to *validate* that harmful prompts land consistently off-origin, never to build the vector itself. This distinction matters — do not let Method 3's implementation quietly become Method 2 with extra steps.

## The evaluation test (Part A + Part B settled in one experiment)

Two sub-tests, run against all three methods' scores:

- **Sub-test A (content-varying, tone-fixed):** neutral-toned prompts, alternating harmful/neutral content (this is what `harmbench_filtered_250.csv` vs `neutral_set_300.csv` already gives you directly).
- **Sub-test B (tone-varying, content-fixed):** same underlying request, varying calm vs. hostile/urgent phrasing. **Partially built as of 2026-09-03 — see `data/subtest_b_MANIFEST.md` for full status and impact analysis.** Two files exist under `data/`:
  - `data/subtest_b_neutral_tone_pairs.csv` — **complete**: 24 calm/hostile pairs on neutral requests, 4 per category across all 6 neutral buckets, all pairs verified clean by inspection.
  - `data/subtest_b_harmful_tone_pairs.csv` — **limited**: 9 sourced pairs from CHATS-Lab Persuasive-Jailbreaker-Data; only 3 pass quality inspection as genuine calm/hostile contrasts (pairs 4, 5, 9). The remaining 6 fail because the persuasion technique labels do not reliably predict actual tonal contrast in the text. See the manifest for the full pair-by-pair verdict.
  - **Statistical consequence:** the neutral arm (N=24) is reportable with bootstrap CIs. The harmful arm (N=3) is not — treat those 3 pairs as a qualitative supplement only, clearly labeled as N=3. The A/B gap analysis can run meaningfully on the neutral arm. Whether harmful-content detectors are tone-sensitive is left as an open question for future work with a larger sourced set.
  - **Future improvement paths** are documented in the manifest: PAIR jailbreaks, human annotation pass on the 9 existing sourced pairs, or LLM-generated paraphrases with human review. Do not silently build around this limitation — flag it in any reported results.

**Win condition:** for each method, report accuracy (bootstrap CI, honest val/test split, per protocol below) separately on sub-test A and sub-test B, plus the gap between them. A method accurate on A but collapsing on B is tone-sensitive, that gap is the actual evidence. Best method = smallest A/B gap with competitive absolute accuracy on A. This determines which method becomes "primary" for the rest of the paper (Part C, portability testing) — do not presuppose Method 3 wins; let the numbers decide.

## Non-negotiable protocol requirements (from the v1 corrected README, carried forward)

1. **Frozen splits, created once.** Train/val/test, seeded, stratified. Group by any shared template/pattern before splitting to avoid near-duplicate leakage across splits.
2. **Selection on validation only, reported once on test.** No threshold or layer choice may see test labels at any stage, including exploratory ones. This was the single largest source of v1's invalidated results (n=2 probe layer selection, threshold optimized against test labels).
3. **Bootstrap confidence intervals on every reported number.** A layer or method "winning" by less than its own CI width is noise — this killed v1's Llama Layer 7 result (63.86% ± 6.45 under honest re-evaluation).
4. **Baselines under every claim.** Include a TF-IDF + logistic regression baseline (the same one that beat every v1 activation result, 66.39% single-fit / 76.17% ± 2.07 five-fold CV) run on the same splits, and ideally a random-direction control for the vector methods themselves.
5. **Base-vs-instruct pair of the same model family**, not cross-family comparison, for any claim about instruction tuning's effect. Use Qwen2.5-1.5B vs Qwen2.5-1.5B-Instruct, or the Llama-3.2-3B equivalent. This replaces v1's confounded GPT-2 (base) vs. Qwen/Llama (instruct, different families) design.
6. **Cache raw per-example activations once, reuse everywhere.** All layers, both pooling variants (masked-mean AND last-token), both formatting variants (raw AND chat-templated for instruct models), float32 only. This is the single extraction pass; everything downstream (method comparison, portability test) runs on the cache with zero further GPU time.
7. **Layer 0 must be tested for every model.** v1's cross-model comparison broke because Qwen's layer sweep skipped layer 0, which was GPT-2's reported optimum. Do not repeat this asymmetry.
8. **Manifest per extraction run:** model revision SHA, transformers version, tokenizer, pooling variant, formatting variant, layer-indexing convention (document that index 0 = embedding output), split file hash, dataset hash (both CSVs above), extraction script git commit.

## Pooling: the length-channel problem (measured, confirmed present, must be addressed)

v1 found mean pooling has a dominant length channel: at GPT-2 layer 6, r(score, char_length) = −0.911 while r(score, label) = −0.117 — length explained ~83% of score variance. This did not inflate v1's accuracy (the v1 dataset happened to be length-balanced, r(label, length) = −0.001) but it drowned the signal.

**This project's data has already been checked, and the confound is present: r(length, label) = 0.436 between `harmbench_filtered_250.csv` and `neutral_set_300.csv` (see the manifests for the full length-check writeup).** This is not a hypothetical risk to watch for, it is a measured property of the two source datasets. Two rounds of manually extending the neutral set closed part of the original gap (43 chars down to 30 chars mean difference) but did not solve it.

Given this, raw mean-pooling must not be used as the sole or default pooling method for this project's data. Centering, standardization, and last-token pooling must be implemented as real, run arms, not optional ablations. Concretely: after extraction, compute r(score, char_length) for each pooling variant on a held-out slice before trusting any downstream method-comparison number. If a variant shows |r(score, length)| approaching v1's −0.911, treat that variant's results as unreliable for reporting, the same way v1's original mean-pooled results were unreliable, and prefer whichever variant shows the weakest length correlation alongside the strongest label correlation.

## Compute constraints (Colab / Kaggle free-tier GPUs)

No persistent guaranteed session, no guaranteed same GPU class twice, sessions can be evicted mid-run. The extraction pass (step 6 above) is the only GPU-heavy phase and must be:

- **Checkpointed per model.** If Qwen finishes and Llama gets evicted mid-run, do not lose Qwen's cache or need to restart Qwen.
- **Resume-safe.** Re-running the script after an eviction should skip already-completed (model, layer, pooling variant, formatting variant) combinations rather than recomputing them.
- **Start with the unbatched-equivalence unit test** before running real extraction: item 0 in a mixed-length batch must score identically whether run alone or batched. This is flagged in the handover as the single highest-probability-of-a-new-bug spot, given the length-channel finding above.

Everything after extraction (method comparison, bootstrap CIs, portability test) is pure numpy/sklearn on the cached activations — no further GPU time needed, can run anywhere.

## Two open method questions (log the comparison, do not resolve by assumption)

- **Cosine-to-class-mean vs. logistic regression probe.** Run a probe as a second arm alongside the mean-diff/cosine approach for whichever method wins the A/B test above. Comparing them is itself a finding worth reporting, not a decision to make silently.

## Deliverables expected back from this build

1. Extraction script(s) meeting the checkpointing/resume requirements above, with the unbatched-equivalence unit test passing before any real run.
2. Cached activations (all layers, both pooling variants, both formatting variants, float32) for GPT-2 Medium, Qwen2.5-1.5B/-Instruct, Llama-3.2-3B/-Instruct, plus the base-vs-instruct pair emphasized in requirement 5.
3. Manifest file(s) per extraction run per requirement 8.
4. Method comparison results: accuracy + bootstrap CI for all three methods on both sub-tests A and B, plus the TF-IDF baseline, reported in a form ready to drop into the paper's Part A/B section.
5. A short note on the length-balance check between the two CSVs and what pooling approach was used as a result.
6. Flag back to Fjord (not silently build around) anything in this spec that turns out to be infeasible, ambiguous, or requiring a data decision — this is a multi-week build against a 2027 deadline, catching a wrong assumption early is cheap; catching it after a full extraction pass on free-tier GPU hours is not.
