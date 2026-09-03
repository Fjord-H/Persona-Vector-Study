"""Simulated-eviction test: interrupt run_extraction_with_checkpointing partway
through, then re-run it, and confirm (a) already-written shards are never
recomputed and (b) the final merged cache is numerically identical to an
uninterrupted from-scratch run — the resume-safety property spec'd for Colab/Kaggle
free-tier eviction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import checkpoint, extraction  # noqa: E402

PROMPT_IDS = [f"p{i}" for i in range(10)]
TEXTS = [
    "Hello.",
    "What is the capital of France?",
    "How do I boil an egg?",
    "Explain photosynthesis briefly.",
    "Thanks for your help today.",
    "What time zone is Tokyo in?",
    "Describe a good morning routine.",
    "Why is the sky blue?",
    "Give me a short packing checklist.",
    "Goodbye for now.",
]
BATCH_SIZE = 3
MAX_LENGTH = 32


def test_resume_skips_completed_shards_and_matches_uninterrupted_run(
    tmp_path, monkeypatch, gpt2_model_and_tokenizer
):
    model, tokenizer, device = gpt2_model_and_tokenizer

    call_count = {"n": 0}
    real_extract = extraction.extract_prompt_activations

    def crash_after_first_batch(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated Colab/Kaggle eviction mid-run")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(checkpoint.extraction, "extract_prompt_activations", crash_after_first_batch)

    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="simulated Colab/Kaggle eviction"):
        checkpoint.run_extraction_with_checkpointing(
            model, tokenizer, device, cache_dir, "gpt2", "raw",
            PROMPT_IDS, TEXTS, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        )

    done_after_crash = checkpoint.completed_prompt_ids(cache_dir, "gpt2", "raw")
    assert done_after_crash == set(PROMPT_IDS[:BATCH_SIZE]), (
        "expected exactly the first batch to be checkpointed before the simulated crash"
    )

    monkeypatch.undo()  # restore the real extraction function for the resume call

    resume_calls = {"n": 0}
    original_after_undo = extraction.extract_prompt_activations

    def counting_wrapper(*args, **kwargs):
        resume_calls["n"] += 1
        return original_after_undo(*args, **kwargs)

    monkeypatch.setattr(checkpoint.extraction, "extract_prompt_activations", counting_wrapper)

    checkpoint.run_extraction_with_checkpointing(
        model, tokenizer, device, cache_dir, "gpt2", "raw",
        PROMPT_IDS, TEXTS, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    # 10 items, batch size 3, first batch (3 items) already done -> 3 more batches
    # (3, 3, 1) needed on resume, never re-touching the first.
    assert resume_calls["n"] == 3

    final_ids = checkpoint.completed_prompt_ids(cache_dir, "gpt2", "raw")
    assert final_ids == set(PROMPT_IDS)

    monkeypatch.undo()

    # Cross-check against a clean, uninterrupted run in a separate cache dir: the
    # merged, id-sorted arrays must match exactly (same model, same inputs, float32).
    fresh_cache_dir = tmp_path / "cache_fresh"
    checkpoint.run_extraction_with_checkpointing(
        model, tokenizer, device, fresh_cache_dir, "gpt2", "raw",
        PROMPT_IDS, TEXTS, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    resumed_ids, resumed_mm, resumed_lt = checkpoint.load_all_shards(cache_dir, "gpt2", "raw")
    fresh_ids, fresh_mm, fresh_lt = checkpoint.load_all_shards(fresh_cache_dir, "gpt2", "raw")

    resumed_order = np.argsort(resumed_ids)
    fresh_order = np.argsort(fresh_ids)

    assert [resumed_ids[i] for i in resumed_order] == [fresh_ids[i] for i in fresh_order]
    np.testing.assert_allclose(resumed_mm[resumed_order], fresh_mm[fresh_order], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(resumed_lt[resumed_order], fresh_lt[fresh_order], atol=1e-4, rtol=1e-4)


def test_second_call_with_everything_already_done_extracts_nothing(
    tmp_path, monkeypatch, gpt2_model_and_tokenizer
):
    model, tokenizer, device = gpt2_model_and_tokenizer
    cache_dir = tmp_path / "cache"

    checkpoint.run_extraction_with_checkpointing(
        model, tokenizer, device, cache_dir, "gpt2", "raw",
        PROMPT_IDS, TEXTS, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    calls = {"n": 0}

    def fail_if_called(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("extract_prompt_activations should not be called when everything is cached")

    monkeypatch.setattr(checkpoint.extraction, "extract_prompt_activations", fail_if_called)

    checkpoint.run_extraction_with_checkpointing(
        model, tokenizer, device, cache_dir, "gpt2", "raw",
        PROMPT_IDS, TEXTS, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )

    assert calls["n"] == 0
