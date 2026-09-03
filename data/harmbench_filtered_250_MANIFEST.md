# harmbench_filtered_250.csv — manifest

## Source

Fetched from the official HarmBench GitHub repository:
https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv

Fetch date: 2026-09-01

## Citation

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., Hendrycks, D. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249.

## License

MIT License. Free to use, modify, and redistribute with attribution. See the HarmBench repository for the full license text.

## What this file is

The original HarmBench text-behaviors dataset contains 350 rows across seven `SemanticCategory` values: illegal, chemical_biological, misinformation_disinformation, cybercrime_intrusion, harassment_bullying, harmful, and copyright.

This file excludes the 100 `copyright` rows (song-lyric reproduction requests, e.g. "Generate the lyrics of X by Y"). That category was dropped for two reasons specific to this project:

1. **Off-topic.** Copyright violation is a different kind of harm from the dangerous-content detection this project studies. It doesn't test whether a model can recognize genuinely harmful intent or capability.
2. **Template-repetition risk.** Nearly all 100 copyright rows follow the identical sentence pattern with only the song and artist swapped. Including them risked teaching a detector to key off of surface phrasing ("generate the lyrics of...") rather than actual harmful content, which is exactly the kind of artifact this project's v1 audit was built to catch.

The remaining 250 rows are kept as-is, with all original columns intact (`Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID`). No behavior text was modified.

## Category breakdown (250 rows)

| SemanticCategory | Count |
|---|---|
| illegal | 62 |
| misinformation_disinformation | 51 |
| cybercrime_intrusion | 49 |
| chemical_biological | 41 |
| harassment_bullying | 25 |
| harmful | 22 |
| **Total** | **250** |

FunctionalCategory split: 200 standard (bare behavior prompts) + 50 contextual (behavior prompts paired with a scenario/context string the model must build on). No multimodal rows are present in this text-only source file.

## Known thin spots

`harassment_bullying` (25) and `harmful` (22) are the smallest categories. If deeper coverage in either is needed, this is where hand-written or hand-sourced additions would matter most — this file alone may be thin there for some downstream splits.

## Next steps this file feeds

- Phase 1 (tone-vs-content disentangling test set): source material for the "harmful content, varying tone" arm.
- Phase 2 (neutral-origin validation set): the harmful-example set used to confirm the origin vector's direction is meaningful.
