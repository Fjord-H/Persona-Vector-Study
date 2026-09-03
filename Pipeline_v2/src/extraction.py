"""Core forward-pass / pooling logic shared by every extraction path.

Two extraction modes, matching the spec's two content sources:

- `extract_prompt_activations`: activations of the *prompt itself* (Methods 2/3,
  sub-tests A and B). Batched, checkpoint-friendly.
- `extract_generation_activations`: activations of a *generated response* under a
  system prompt (Method 1 only). Unbatched by design — see its docstring.

Both always compute BOTH pooling variants (masked_mean, last_token) and ALL layers
from a single forward pass, per the plan's checkpoint-granularity decision: one forward
pass is the expensive, resumable unit; layers and pooling variants are free byproducts
of it and are never separately checkpointed or recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src import pooling

DEFAULT_MAX_NEW_TOKENS = 64


@dataclass
class BatchActivations:
    """All-layer, both-pooling-variant activations for one batch.

    masked_mean / last_token: float32 tensors [batch, n_layers_plus_1, hidden].
    """
    masked_mean: torch.Tensor
    last_token: torch.Tensor
    n_layers_plus_1: int
    hidden_dim: int


def _forward_hidden_states(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Runs the model and returns hidden_states as one stacked tensor
    [batch, n_layers_plus_1, seq_len, hidden], float32.

    Layer-indexing convention (config.py docstring): index 0 = embedding output.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # outputs.hidden_states is a tuple of length n_layers+1, each [batch, seq_len, hidden].
    return torch.stack(outputs.hidden_states, dim=1).to(torch.float32)


def _pool_all_layers(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                      pooling_fn) -> torch.Tensor:
    """hidden_states: [batch, n_layers_plus_1, seq_len, hidden]. Returns
    [batch, n_layers_plus_1, hidden], applying pooling_fn independently per layer.
    """
    batch, n_layers_plus_1, _seq_len, _hidden = hidden_states.shape
    pooled_per_layer = [pooling_fn(hidden_states[:, layer], attention_mask) for layer in range(n_layers_plus_1)]
    return torch.stack(pooled_per_layer, dim=1)


def extract_prompt_activations(model, tokenizer, texts: list[str], device: str,
                                max_length: int = 128) -> BatchActivations:
    """Tokenizes `texts` as one right-padded batch and returns both pooling variants
    for every layer. This is the function whose item-0 output must match an unbatched
    (single-item) call exactly — see tests/test_pooling_batch_equivalence.py.
    """
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    hidden_states = _forward_hidden_states(model, input_ids, attention_mask)
    masked_mean_pooled = _pool_all_layers(hidden_states, attention_mask, pooling.masked_mean)
    last_token_pooled = _pool_all_layers(hidden_states, attention_mask, pooling.last_token)

    return BatchActivations(
        masked_mean=masked_mean_pooled.cpu(),
        last_token=last_token_pooled.cpu(),
        n_layers_plus_1=hidden_states.shape[1],
        hidden_dim=hidden_states.shape[3],
    )


def extract_generation_activations(model, tokenizer, prompt_text: str, device: str,
                                    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                                    max_prompt_length: int = 256) -> BatchActivations:
    """Method 1 only: generates one response to `prompt_text` (already chat-formatted
    with the persona system prompt by the caller, via formatting.format_chat) and pools
    activations over the GENERATED response tokens only, never the prompt tokens.

    Unbatched by design. Batched generation needs left-padding (so all sequences end at
    the same position for `generate()` to continue correctly), and left-padding is
    exactly the footgun model_io.py forces right-padding to avoid for prompt-only
    extraction. Method 1's stimulus set is only 40 questions x 2 conditions x a handful
    of instruct models — small enough that unbatched generation is fast enough, and it
    keeps this path correctness-first rather than reintroducing the padding-side risk
    for a method that is expected to fail anyway (it is the falsifying control, not the
    primary result).
    """
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_len = generated.shape[1]
    if full_len <= prompt_len:
        raise ValueError(
            "extract_generation_activations: model produced zero new tokens "
            f"(prompt_len={prompt_len}, full_len={full_len}); cannot pool an empty response."
        )

    full_attention_mask = torch.ones_like(generated)
    hidden_states = _forward_hidden_states(model, generated, full_attention_mask)

    # Response-only mask: zero out the prompt span so pooling.py's masked_mean/last_token
    # only ever see the generated tokens, per spec ("activations pulled from generated
    # responses not the prompt").
    response_mask = full_attention_mask.clone()
    response_mask[:, :prompt_len] = 0

    masked_mean_pooled = _pool_all_layers(hidden_states, response_mask, pooling.masked_mean)
    last_token_pooled = _pool_all_layers(hidden_states, response_mask, pooling.last_token)

    return BatchActivations(
        masked_mean=masked_mean_pooled.cpu(),
        last_token=last_token_pooled.cpu(),
        n_layers_plus_1=hidden_states.shape[1],
        hidden_dim=hidden_states.shape[3],
    )
