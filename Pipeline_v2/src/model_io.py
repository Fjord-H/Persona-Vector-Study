"""Model/tokenizer loading, following the incantations that already work in v1
(survey of notebooks/01*, 01a*, 06* — see the build plan) with two deliberate changes:

1. Right-padding is forced on every tokenizer, always. With right-padding, a real
   token's absolute position (and therefore its hidden state) is identical whether it
   sits alone in a batch of 1 or alongside longer, right-padded sequences in a bigger
   batch — HF's default position-id computation (arange(seq_len), no attention-mask
   adjustment) only guarantees this for right-padding. Left-padding shifts position ids
   for real tokens unless position_ids are explicitly recomputed from the attention
   mask (as `generate()` does internally but a raw `forward()` call does not), which is
   exactly the kind of silent batching bug the spec's unbatched-equivalence unit test
   exists to catch. Forcing right-padding here sidesteps the footgun instead of
   papering over it. See tests/test_pooling_batch_equivalence.py.
2. The cache is float32-only (spec requirement 6); model weights may load in fp16/bf16
   on CUDA to fit free-tier GPU memory, but callers must cast hidden states to float32
   immediately after the forward pass (pooling.py already does this internally).
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ModelSpec


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def load_tokenizer(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=True, token=_hf_token()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(spec: ModelSpec, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=compute_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        token=_hf_token(),
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, device


def load_model_and_tokenizer(spec: ModelSpec, device: str | None = None):
    tokenizer = load_tokenizer(spec)
    model, resolved_device = load_model(spec, device=device)
    return model, tokenizer, resolved_device
