# data/vectors — file manifest

Files are kept flat (not in subfolders) because notebooks 01/01a/02/02a load and save
them by bare filename relative to `data/vectors/`. Moving files into subfolders would
break those paths. This manifest exists instead, to make provenance explicit.

**All v1 accuracy/separation figures in this directory are retracted.**
See `../../defect_report.md`.

## Canonical run (200 train / 1800 test, notebooks 04–06)

| file | model | contents |
|---|---|---|
| `gpt2_vectors_200.pkl` | GPT-2 | safe/danger mean vectors, layer 23 (see defect S1-05: this is not the layer the headline result reports) |
| `qwen_vectors_200.pkl` | Qwen | safe/danger mean vectors, layer 1 (same caveat) |
| `gpt2_vectors_weighted.pkl` | GPT-2 | confidence-weighted variant — near-identical to unweighted, see defect S3-01 |
| `qwen_vectors_weighted.pkl` | Qwen | confidence-weighted variant |
| `train_test_split.pkl` | — | the 200/1800 split used across notebooks 04–06 |
| `dataset_2000.pkl` | — | full 2000-example dataset with source breakdown (defect S2-07) |
| `gpt2_results_200vec.pkl`, `qwen_results_200vec.pkl` | — | per-example scores, 200-example scale |
| `all_layers_proper.pkl` | GPT-2 | **per-example scores at every layer, all 1800 test items.** This is what `defect_report.md`'s honest val/test recomputation (S1-06, S2-03) is derived from. Published 2026-08 alongside the correction; was previously local-only. |
| `llama_results.pkl` | Llama | summary only (7 layers, no per-example scores) |
| `layer_analysis_results.pkl` | — | layer sweep summary |
| `category_results.pkl`, `ensemble_results.pkl` | — | failed-experiment results (per-category vectors, multi-layer ensemble) |

## Pre-canonical / superseded — kept for the historical record, not for citation

| file | why superseded |
|---|---|
| `gpt2_content_vectors.pkl`, `qwen_content_vectors.pkl` | 50-example era, predates the 200/1800 split. `metadata.accuracy` fields (92.5 / 100.0) are the contaminated figures — see defect S1-01. |
| `gpt2_tone_vectors.pkl` | notebook 01 tone-based baseline (superseded by content vectors) |
| `qwen_tone_vectors.pkl` | notebook 01a tone-based baseline |
| `gpt2_safety_vectors.pkl` | **identical tensors to `gpt2_tone_vectors.pkl`** under a different name — defect S3-03. Kept rather than deleted; do not treat as an independent artifact. |

## Not in this directory / not reproducible

`all_layers_comprehensive.pkl`, referenced during the audit, contains internally
impossible values (exact 0.000 accuracy at several GPT-2 layers — a direction-error
artifact) and is not published here.
