# neutral_set_300.csv — manifest

## What this file is

300 hand-written, ordinary, harmless prompts, built to define the "neutral origin" point for the v2 activation-vector method (Design 1 from the Phase 0 spec). This is the counterpart to `harmbench_filtered_250.csv`: where that file provides the harmful-content validation set, this file provides the neutral baseline whose mean activation becomes the zero point everything else is measured against.

Authored directly for this project (not sourced from an external dataset), 2026-09-01. Originally built at 240 prompts, then extended to 300 across two rounds after a length-balance check (below) found a gap against the harmful set.

## Design rationale

Two things were deliberately matched to HarmBench's structure, to avoid introducing a new artifact of the kind the v1 audit found:

**Topic breadth.** Six buckets echo the six real HarmBench semantic categories in spirit (one ordinary-life topic area each, rather than one narrow theme repeated): cooking_household, science_general_knowledge, technology_howto, hobbies_creative, everyday_logistics, health_wellness_basic. 50 prompts per bucket (evenly split).

**Phrasing-shape balance.** HarmBench behaviors mix instruction-style commands ("Give detailed instructions for...", "Create a list of...", "Write a tutorial on...") with direct requests. This set balances instruction-style openings against question-style openings, so a detector can't learn to separate neutral from harmful purely by sentence form.

## Length-balance check — the reason this file grew from 240 to 300

A near-duplicate and length check against `harmbench_filtered_250.csv` (run 2026-09-01) found:

- **Content overlap: clean.** Zero exact duplicates. Only one borderline near-duplicate pair out of 250×300 comparisons (SequenceMatcher ratio 0.75), and it's a shared generic template phrase ("give step-by-step instructions for"), not meaningful content overlap.
- **Length imbalance: a real, only partially fixed problem.** At 240 prompts, harmful mean length was 96.4 characters vs. neutral mean 53.4 — a 43-character gap. Two rounds of adding longer, more instruction-heavy prompts brought this to harmful mean 96.4 vs. neutral mean 66.4 (a 30-character gap), and the combined-set correlation r(length, label) = **0.436**.

**This is not resolved.** v1's own dataset was safely length-balanced at r(label, length) = −0.001; this set is nowhere near that. Two rounds of hand-writing longer prompts produced diminishing returns — HarmBench's own within-category length variance (stdev 34.2 chars) makes hand-matching the distribution impractical without either (a) writing many more rounds of prompts, or (b) accepting the imbalance and requiring length-robust pooling.

**Decision made:** stop extending by hand at 300 (clean round number, even 50/category), and require the extraction pipeline to treat this as a live length confound, not a solved problem. See `v2_pipeline_SPEC.md`'s pooling section — centering, standardization, or last-token pooling must be used, not raw mean pooling, and length-vs-score correlation should be checked directly on the extracted activations before trusting any method comparison result. Given v1's finding that mean pooling produces r(score, length) as high as −0.911 even when the *dataset* was balanced, a dataset with r(length, label) = 0.436 makes length-robust pooling a hard requirement, not an optional ablation arm, for this project's data.

## Verified clean

- 300 unique prompts, zero duplicates (by ID and by text).
- Even 50/50/50/50/50/50 split across the six category buckets.
- Near-duplicate content check against `harmbench_filtered_250.csv`: clean (1 borderline template-phrase match, no real overlap).

## Columns

`Prompt, Category, PromptID`

## Known limitations

- Hand-written by one author (this project), not independently sourced.
- **Length imbalance against the harmful set is real and only partially mitigated (r = 0.436). Extraction must use length-robust pooling; do not proceed with raw mean-pooling on these two sets without checking the resulting score-length correlation directly.**
- Not yet checked for near-duplicate phrasing among the 300 neutral prompts against each other (only checked against the harmful set).

## Next steps this file feeds

- Phase 2 (neutral-origin construction): mean activation over this set defines "zero" for the Design 1 method.
- Phase 1 (tone-vs-content disentangling test): source material for the "neutral content" arm, when combined with tone-varied phrasing.
