"""THE unit test the spec requires before any real extraction run:

    "item 0 in a mixed-length batch must score identically whether run alone or
    batched" (v2_pipeline_SPEC.md, Compute constraints).

This exercises the real pipeline path (model_io.load_model_and_tokenizer +
extraction.extract_prompt_activations), not a mock, against a small real pretrained
model ("gpt2" — 124M, the smallest available, downloaded once and cached by
huggingface_hub). It must pass before extraction.py is trusted on Qwen/Llama or on
Colab/Kaggle GPU quota. If no network/model cache is available, the test is skipped
with a clear reason rather than silently reporting green — an environment where this
test cannot run is an environment where a real extraction run should not be trusted
either.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import extraction  # noqa: E402

# Deliberately mixed lengths, with the shortest and longest NOT adjacent, so padding
# amount varies non-monotonically across the batch.
MIXED_LENGTH_PROMPTS = [
    "Hello.",
    "What is a simple way to make rice not stick to the pot when cooking dinner tonight?",
    "How do I tune a guitar?",
    "Explain, in careful and complete detail, why the sky appears blue during the "
    "middle of a clear day, and how that differs at sunrise and sunset.",
    "Thanks!",
]


def test_tokenizer_is_right_padded(gpt2_model_and_tokenizer):
    _model, tokenizer, _device = gpt2_model_and_tokenizer
    assert tokenizer.padding_side == "right"


def test_batched_matches_unbatched_for_every_item(gpt2_model_and_tokenizer):
    model, tokenizer, device = gpt2_model_and_tokenizer

    batched = extraction.extract_prompt_activations(model, tokenizer, MIXED_LENGTH_PROMPTS, device=device)

    # Sanity on shape: gpt2 has 12 transformer blocks -> 13 hidden_states (index 0 =
    # embedding output, per config.py's documented convention).
    assert batched.n_layers_plus_1 == 13
    assert batched.masked_mean.shape == (len(MIXED_LENGTH_PROMPTS), 13, batched.hidden_dim)

    for i, prompt in enumerate(MIXED_LENGTH_PROMPTS):
        unbatched = extraction.extract_prompt_activations(model, tokenizer, [prompt], device=device)

        mm_diff = (batched.masked_mean[i] - unbatched.masked_mean[0]).abs().max().item()
        lt_diff = (batched.last_token[i] - unbatched.last_token[0]).abs().max().item()

        assert torch.allclose(batched.masked_mean[i], unbatched.masked_mean[0], atol=1e-4, rtol=1e-4), (
            f"masked_mean mismatch at batch index {i} ({prompt!r}): max abs diff {mm_diff}"
        )
        assert torch.allclose(batched.last_token[i], unbatched.last_token[0], atol=1e-4, rtol=1e-4), (
            f"last_token mismatch at batch index {i} ({prompt!r}): max abs diff {lt_diff}"
        )


def test_batch_order_does_not_change_item_0(gpt2_model_and_tokenizer):
    """The spec names item 0 specifically. Run it first, then re-run with the same
    item moved to the middle and to the end of the batch, and confirm its pooled
    vectors don't change — i.e. equivalence holds regardless of where in the batch
    (and thus how much padding) an item sits.
    """
    model, tokenizer, device = gpt2_model_and_tokenizer
    item0 = MIXED_LENGTH_PROMPTS[0]

    orderings = [
        [item0] + MIXED_LENGTH_PROMPTS[1:],
        MIXED_LENGTH_PROMPTS[1:3] + [item0] + MIXED_LENGTH_PROMPTS[3:],
        MIXED_LENGTH_PROMPTS[1:] + [item0],
    ]
    results = []
    for ordering in orderings:
        idx = ordering.index(item0)
        out = extraction.extract_prompt_activations(model, tokenizer, ordering, device=device)
        results.append((out.masked_mean[idx], out.last_token[idx]))

    reference_mm, reference_lt = results[0]
    for mm, lt in results[1:]:
        assert torch.allclose(mm, reference_mm, atol=1e-4, rtol=1e-4)
        assert torch.allclose(lt, reference_lt, atol=1e-4, rtol=1e-4)
