"""Orchestrates one model's full extraction pass: sub-test A + sub-test B prompt
activations (both formatting variants where available), plus Method 1's generation
activations for instruct models. This is the actual logic behind
notebooks/01_extract_activations.ipynb — kept here as plain, testable code rather than
inline in notebook JSON, per this project's "verify against the artifact" practice.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.checkpoint import run_extraction_with_checkpointing, run_generation_extraction_with_checkpointing
from src.config import (
    CACHE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    MODEL_REGISTRY,
    POOLING_VARIANTS,
    SUBTEST_B_HARMFUL_CSV,
    SUBTEST_B_NEUTRAL_CSV,
    ModelSpec,
)
from src.extraction import DEFAULT_MAX_NEW_TOKENS
from src.formatting import available_formatting_variants, format_chat, format_raw, has_chat_template
from src.manifest import build_run_manifest, write_run_manifest
from src.methods.tone_pole import load_stimulus_texts
from src.model_io import load_model_and_tokenizer
from src.splits import load_combined_rows


def _subtest_b_neutral_items() -> list[tuple[str, str]]:
    df = pd.read_csv(SUBTEST_B_NEUTRAL_CSV)
    items = []
    for _, row in df.iterrows():
        items.append((f"{row['PairID']}__calm", row["calm_prompt"]))
        items.append((f"{row['PairID']}__hostile", row["hostile_prompt"]))
    return items


def _subtest_b_harmful_items() -> list[tuple[str, str]]:
    """All 9 sourced rows are extracted (cheap, and keeps the cache complete/inspectable)
    even though only the 3 verified-clean rows (config.SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES)
    are ever used as a reportable result — see data/subtest_b_MANIFEST.md and
    eval/subtest_b.py."""
    df = pd.read_csv(SUBTEST_B_HARMFUL_CSV)
    items = []
    for idx, row in df.iterrows():
        items.append((f"STB_H_{idx:03d}__calm", row["calm_prompt"]))
        items.append((f"STB_H_{idx:03d}__hostile", row["hostile_prompt"]))
    return items


def build_subtest_ab_items(tokenizer, formatting_variant: str) -> tuple[list[str], list[str]]:
    """Returns (item_ids, formatted_texts) for sub-test A (550 harmbench + neutral
    prompts) plus both sub-test B arms (48 + 18 = 66 items), formatted per
    `formatting_variant`. All of this content shares one namespace (same model_key,
    formatting_variant) in the cache since eval/subtest_a.py and eval/subtest_b.py both
    read from it via activation_store.load_layer_matrix keyed by item_id.
    """
    if formatting_variant == "raw":
        format_fn = format_raw
    elif formatting_variant == "chat":
        format_fn = lambda text: format_chat(tokenizer, text)  # noqa: E731
    else:
        raise ValueError(f"unknown formatting_variant {formatting_variant!r}")

    raw_items: list[tuple[str, str]] = [(row.prompt_id, row.text) for row in load_combined_rows()]
    raw_items += _subtest_b_neutral_items()
    raw_items += _subtest_b_harmful_items()

    ids = [item_id for item_id, _ in raw_items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id across sub-test A/B content — cannot extract safely")

    item_ids = [item_id for item_id, _ in raw_items]
    texts = [format_fn(text) for _, text in raw_items]
    return item_ids, texts


def extract_all_for_model(
    model_key: str, cache_dir: Path | None = None, batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    progress_callback=None, device: str | None = None,
) -> list[Path]:
    """Runs every extraction this model needs (prompt activations for every available
    formatting variant, plus Method 1's generation activations if the tokenizer has a
    chat template) and writes one manifest per run. Resume-safe throughout — safe to
    interrupt and re-call.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model_key {model_key!r}; see src.config.MODEL_REGISTRY")
    spec: ModelSpec = MODEL_REGISTRY[model_key]
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR

    model, tokenizer, resolved_device = load_model_and_tokenizer(spec, device=device)
    manifest_paths: list[Path] = []

    for formatting_variant in available_formatting_variants(tokenizer):
        item_ids, texts = build_subtest_ab_items(tokenizer, formatting_variant)
        run_extraction_with_checkpointing(
            model, tokenizer, resolved_device, cache_dir, model_key, formatting_variant,
            item_ids, texts, batch_size=batch_size, max_length=max_length,
            progress_callback=progress_callback,
        )
        run_manifest = build_run_manifest(
            spec, tokenizer, formatting_variant, POOLING_VARIANTS,
            n_prompts_extracted=len(item_ids), batch_size=batch_size, max_length=max_length,
        )
        manifest_paths.append(write_run_manifest(run_manifest))

    if has_chat_template(tokenizer):
        item_ids, texts = load_stimulus_texts(tokenizer)
        run_generation_extraction_with_checkpointing(
            model, tokenizer, resolved_device, cache_dir, model_key, item_ids, texts,
            max_new_tokens=max_new_tokens, progress_callback=progress_callback,
        )
        run_manifest = build_run_manifest(
            spec, tokenizer, "generation", POOLING_VARIANTS, n_prompts_extracted=len(item_ids),
            extra={"purpose": "method_1_tone_pole", "max_new_tokens": max_new_tokens},
        )
        manifest_paths.append(write_run_manifest(run_manifest))

    return manifest_paths
