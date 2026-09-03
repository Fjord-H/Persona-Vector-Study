"""Method 1 — tone-pole (expected-to-fail falsifying control).

Vector = mean(activations on harmful_dishonest-conditioned generations) -
mean(activations on helpful_harmless-conditioned generations), built entirely from
data/tone_pole_system_prompts.csv + data/tone_pole_questions_40.csv (never from
harmbench_filtered_250.csv or neutral_set_300.csv — see
data/tone_pole_stimuli_MANIFEST.md). Scoring reuses the same midpoint-projection
geometry as Method 2 (mean_diff.py) so all three methods are compared with the same
scoring mechanic — the difference between methods is entirely in what each vector was
built from, not in how a vector turns into a score.

Per spec, this method is applied to score the SAME prompt-activation content sub-tests
A and B already use (harmbench/neutral CSVs, sub-test B pairs) — only the vector's
construction is different (generated-response activations, not prompt activations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import activation_store, checkpoint
from src.config import TONE_POLE_QUESTIONS_CSV, TONE_POLE_SYSTEM_PROMPTS_CSV
from src.formatting import format_chat
from src.methods.mean_diff import MeanDiffPole, fit_mean_diff_pole, predict_from_pole, project

HELPFUL_CONDITION = "helpful_harmless"
HARMFUL_CONDITION = "harmful_dishonest"


def load_stimulus_texts(tokenizer) -> tuple[list[str], list[str]]:
    """Returns (item_ids, formatted_prompt_texts) for all 40 questions x 2 conditions
    (80 items total), chat-formatted with the matching system prompt via
    formatting.format_chat. Requires a tokenizer with a chat_template — Method 1 only
    runs on instruct models, per the build plan (persona system-prompt conditioning is
    an instruct-model technique).
    """
    system_prompts = pd.read_csv(TONE_POLE_SYSTEM_PROMPTS_CSV).set_index("Condition")["SystemPrompt"]
    questions = pd.read_csv(TONE_POLE_QUESTIONS_CSV)

    item_ids, texts = [], []
    for condition in (HELPFUL_CONDITION, HARMFUL_CONDITION):
        system_prompt = system_prompts.loc[condition]
        for _, row in questions.iterrows():
            item_ids.append(f"{row['QuestionID']}__{condition}")
            texts.append(format_chat(tokenizer, row["Question"], system_prompt=system_prompt))
    return item_ids, texts


def fit_tone_pole(cache_dir, model_key: str, pooling_variant: str, layer: int) -> MeanDiffPole:
    """Reads the "generation" cache (written by
    checkpoint.run_generation_extraction_with_checkpointing) and fits a MeanDiffPole
    with positive=harmful_dishonest, negative=helpful_harmless, at one (pooling_variant,
    layer) combination.
    """
    all_ids, mm_all, lt_all = checkpoint.load_all_shards(cache_dir, model_key, "generation")
    if not all_ids:
        raise activation_store.ActivationLookupError(
            f"no tone-pole generation cache found for model={model_key!r} under {cache_dir}"
        )
    arr = mm_all if pooling_variant == "masked_mean" else lt_all

    helpful_rows = [i for i, iid in enumerate(all_ids) if iid.endswith(f"__{HELPFUL_CONDITION}")]
    harmful_rows = [i for i, iid in enumerate(all_ids) if iid.endswith(f"__{HARMFUL_CONDITION}")]
    if not helpful_rows or not harmful_rows:
        raise activation_store.ActivationLookupError(
            f"tone-pole cache for model={model_key!r} is missing one condition "
            f"(helpful rows: {len(helpful_rows)}, harmful rows: {len(harmful_rows)})"
        )

    helpful_matrix = arr[helpful_rows, layer, :]
    harmful_matrix = arr[harmful_rows, layer, :]
    return fit_mean_diff_pole(positive_matrix=harmful_matrix, negative_matrix=helpful_matrix)


def score(pole: MeanDiffPole, activations: np.ndarray) -> np.ndarray:
    """Applied to PROMPT activations (sub-test A/B content), NOT more generations —
    this is the scoring step that makes Method 1 comparable to Methods 2/3 on the same
    evaluation content, per spec ("All three must run on the same underlying content
    where possible")."""
    return project(activations, pole.midpoint, pole.direction)


def predict(pole: MeanDiffPole, activations: np.ndarray) -> np.ndarray:
    return predict_from_pole(pole, activations)
