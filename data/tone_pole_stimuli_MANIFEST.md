# Tone-pole stimulus set — manifest

## What this is

Method 1 (tone-pole, the expected-to-fail falsifying control — `v2_pipeline_SPEC.md`
"Three methods", Method 1) needs its own small stimulus set, separate from
`harmbench_filtered_250.csv` and `neutral_set_300.csv`. Following the article this
method reproduces (Chen et al., *Anthropic persona vectors*, arXiv:2507.21509), the
tone-pole vector is built from **generated responses** under contrasting system
prompts, not from the prompts themselves. Two files:

- **`tone_pole_system_prompts.csv`** — one helpful/harmless system prompt, one
  harmful/dishonest system prompt. Columns: `Condition, SystemPrompt`.
- **`tone_pole_questions_40.csv`** — 40 neutral provocation questions used to elicit a
  response under each system prompt. Columns: `QuestionID, Category, Question`.

Both hand-authored for this project, 2026-09-03.

## Design rationale

**Why 40 questions, not HarmBench-scale.** The spec is explicit that this method
"does not need HarmBench-scale data since it is the known-failing control" and that the
original article used 40 questions — matched here deliberately rather than scaled up,
since inflating this set would not change what the method is being used to demonstrate
(that persona-conditioned tone, not content, drives the resulting vector).

**Why the questions are neutral.** The provocation questions are deliberately ordinary
(cooking, general science, tech how-to, hobbies, everyday logistics, basic wellness —
the same six topic buckets as `neutral_set_300.csv`, for consistency, but freshly
written, not duplicated from that file). None of the 40 questions request harmful
content. The point of Method 1 is that the *system prompt* alone should shift the
model's response style/activations even on completely benign questions — that shift is
what the tone-pole vector captures, and why it is expected to fail at detecting harmful
*content* in sub-test A: it was never built from content in the first place.

**System-prompt wording.** Both system prompts are trait-conditioning instructions in
the same style as Chen et al.'s persona-vector elicitation prompts — they describe a
*persona* (careful and honest vs. careless and dishonest), not a request for specific
dangerous operational content. Combined with the neutral questions above, no prompt in
this set asks the model to produce genuinely dangerous material; worst case under the
harmful_dishonest condition is a deliberately careless or dismissive answer to a
cooking/tech/wellness question. This keeps the control uncontaminated by an unrelated
safety-refusal confound (a model refusing to answer under the "harmful" persona would
itself be a confound, not signal).

## How this differs from `harmbench_filtered_250.csv` / `neutral_set_300.csv`

Those two files are prompts whose own activations are extracted directly (Methods 2 and
3, sub-test A). This set is never scored directly — it is a generation harness: for each
of the 40 questions, the pipeline generates one response under `helpful_harmless` and one
under `harmful_dishonest`, and pools activations over the **generated response tokens**,
per the spec ("activations pulled from generated responses not the prompt"). 40 questions
x 2 conditions = 80 generations per model, all on models with a chat template (persona
system-prompt conditioning is an instruct-model technique — see the pipeline's model
registry / `formatting.py` for which models this applies to).

## Known limitations

- Hand-written by one author (this project), same convention as `neutral_set_300.csv`
  and the sub-test B pairs — not independently sourced or reviewed.
- Only one system-prompt pair per condition (not multiple phrasings of "helpful" or
  "harmful"), matching the article's minimal-pair approach but meaning the resulting
  vector reflects one specific pair of instructions, not an average over many.
- Topic buckets echo `neutral_set_300.csv`'s six categories for internal consistency;
  this is a deliberate choice, not an oversight, since Method 1's questions are meant to
  be as unremarkable as possible.

## Next steps this file feeds

Extraction pipeline (`Pipeline_v2/`), Method 1 implementation
(`Pipeline_v2/src/methods/tone_pole.py`) and its generation-based extraction path
(`Pipeline_v2/src/extraction.py::extract_generation_activations`).
