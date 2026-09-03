# neutral_set_240.csv — manifest (SUPERSEDED)

**This file and its 240-row CSV are superseded by `neutral_set_300.csv` and
`neutral_set_300_MANIFEST.md`.** A length-balance check against
`harmbench_filtered_250.csv` found the original 240-prompt set was too short
relative to the harmful set (a 43-character mean gap), risking the same
length-channel confound documented in the v1 audit. The set was extended to
300 prompts to partially close that gap. Use `neutral_set_300.csv` going
forward; this file is kept only as a historical record, per this project's
"nothing gets deleted, only corrected" convention.

---

## What this file is (original, now superseded)

240 hand-written, ordinary, harmless prompts, built to define the "neutral origin" point for the v2 activation-vector method (Design 1 from the Phase 0 spec). This is the counterpart to `harmbench_filtered_250.csv`: where that file provides the harmful-content validation set, this file provides the neutral baseline whose mean activation becomes the zero point everything else is measured against.

Authored directly for this project (not sourced from an external dataset), 2026-09-01.

## Design rationale

Two things were deliberately matched to HarmBench's structure, to avoid introducing a new artifact of the kind the v1 audit found:

**Topic breadth.** Six buckets echo the six real HarmBench semantic categories in spirit (one ordinary-life topic area each, rather than one narrow theme repeated): cooking_household, science_general_knowledge, technology_howto, hobbies_creative, everyday_logistics, health_wellness_basic. 40 prompts per bucket.

**Phrasing-shape balance.** HarmBench behaviors mix instruction-style commands ("Give detailed instructions for...", "Create a list of...", "Write a tutorial on...") and, less often, direct requests. This set deliberately balances instruction-style openings (117 of 240: Give / Write / Create) against question-style openings (123 of 240: What / How / Explain), so a detector can't learn to separate neutral from harmful purely by sentence form. This directly addresses the tone-vs-content confound documented in Part A of this project — a neutral set that was all questions while HarmBench skews toward instructions would reintroduce exactly that failure mode.

## Verified clean

- 240 unique prompts, zero duplicates.
- Even 40/40/40/40/40/40 split across the six category buckets.
- Near-even split between instruction-style (117) and question-style (123) phrasing.

## Columns

`Prompt, Category, PromptID`

## Known limitations

- Hand-written by one author (this project), not independently sourced — some topic-selection bias is possible despite the deliberate spread across six buckets.
- Not yet checked against HarmBench's `Behavior` column for accidental near-duplicate phrasing across the two files; worth a leakage check before use in Phase 1's disentangling test specifically, since that test compares behavior directly against this set.
- Size (240) is close to but not exactly matched to the harmful set (250) — fine for defining an origin point, but worth keeping in mind if a strictly balanced comparison is needed later.

## Next steps this file feeds

- Phase 2 (neutral-origin construction): mean activation over this set defines "zero" for the Design 1 method.
- Phase 1 (tone-vs-content disentangling test): source material for the "neutral content" arm, when combined with tone-varied phrasing.
