"""Model registry and shared paths for the v2 extraction pipeline.

Layer-indexing convention (per spec requirement 8): `output_hidden_states=True` returns
a tuple of length n_layers + 1, where index 0 is the embedding output (before any
transformer block) and index n_layers is the final block's output. This convention is
used everywhere in this pipeline and is recorded in every run manifest so it is never
ambiguous downstream. "Layer 0" always means the embedding output, never the first
transformer block.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo layout: Pipeline_v2/ is a sibling of data/ at the repo root.
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
DATA_DIR = REPO_ROOT / "data"

# Overridable so Colab/Kaggle can point this at a mounted Drive folder or
# /kaggle/working without touching code — see notebooks/01_extract_activations.ipynb.
CACHE_DIR = Path(os.environ.get("PV2_CACHE_DIR", PIPELINE_ROOT / "cache"))
MANIFEST_DIR = Path(os.environ.get("PV2_MANIFEST_DIR", PIPELINE_ROOT / "manifests"))
# Pipeline-generated (not hand-sourced) artifacts live under Pipeline_v2/, not data/ —
# data/ is reserved for the hand-curated/sourced CSVs and their manifests per this
# project's existing convention; splits.csv is derived output, not source data.
SPLITS_DIR = Path(os.environ.get("PV2_SPLITS_DIR", PIPELINE_ROOT / "splits"))
FROZEN_SPLITS_PATH = SPLITS_DIR / "splits_frozen.csv"

HARMBENCH_CSV = DATA_DIR / "harmbench_filtered_250.csv"
NEUTRAL_CSV = DATA_DIR / "neutral_set_300.csv"
TONE_POLE_SYSTEM_PROMPTS_CSV = DATA_DIR / "tone_pole_system_prompts.csv"
TONE_POLE_QUESTIONS_CSV = DATA_DIR / "tone_pole_questions_40.csv"
SUBTEST_B_NEUTRAL_CSV = DATA_DIR / "subtest_b_neutral_tone_pairs.csv"
SUBTEST_B_HARMFUL_CSV = DATA_DIR / "subtest_b_harmful_tone_pairs.csv"
# Rows in subtest_b_harmful_tone_pairs.csv verified clean by manual inspection
# (see data/subtest_b_MANIFEST.md) — 1-indexed row numbers as listed in the manifest's
# pair-by-pair verdict table. N=3; not independently reportable with a bootstrap CI,
# qualitative supplement only. Row indices below are 0-indexed positions in the CSV
# (row 4 -> index 3, row 5 -> index 4, row 9 -> index 8).
SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES = (3, 4, 8)


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short identifier used throughout the pipeline and cache filenames
    hf_id: str           # HuggingFace model id
    family: str            # groups a base/instruct pair for requirement 5 comparisons
    is_instruct: bool        # has (or is expected to have) a chat template
    pairs_with: str | None     # key of this model's base/instruct counterpart, or None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt2-medium": ModelSpec(
        key="gpt2-medium", hf_id="gpt2-medium", family="gpt2",
        is_instruct=False, pairs_with=None,
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b", hf_id="Qwen/Qwen2.5-1.5B", family="qwen2.5",
        is_instruct=False, pairs_with="qwen2.5-1.5b-instruct",
    ),
    "qwen2.5-1.5b-instruct": ModelSpec(
        key="qwen2.5-1.5b-instruct", hf_id="Qwen/Qwen2.5-1.5B-Instruct", family="qwen2.5",
        is_instruct=True, pairs_with="qwen2.5-1.5b",
    ),
    "llama-3.2-3b": ModelSpec(
        key="llama-3.2-3b", hf_id="meta-llama/Llama-3.2-3B", family="llama-3.2",
        is_instruct=False, pairs_with="llama-3.2-3b-instruct",
    ),
    "llama-3.2-3b-instruct": ModelSpec(
        key="llama-3.2-3b-instruct", hf_id="meta-llama/Llama-3.2-3B-Instruct", family="llama-3.2",
        is_instruct=True, pairs_with="llama-3.2-3b",
    ),
}

# Formatting variants, per model: raw always runs; chat only runs when the loaded
# tokenizer actually reports a chat_template (checked at runtime in formatting.py,
# not assumed from is_instruct here, since that's a claim about the pipeline's intent
# and the tokenizer's actual chat_template attribute is the ground truth).
FORMATTING_VARIANTS = ("raw", "chat")
POOLING_VARIANTS = ("masked_mean", "last_token")

DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42
