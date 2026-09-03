import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelSpec  # noqa: E402
from src.model_io import load_model_and_tokenizer  # noqa: E402

GPT2_SPEC = ModelSpec(key="gpt2", hf_id="gpt2", family="gpt2", is_instruct=False, pairs_with=None)


@pytest.fixture(scope="session")
def gpt2_model_and_tokenizer():
    try:
        model, tokenizer, device = load_model_and_tokenizer(GPT2_SPEC, device="cpu")
    except Exception as exc:  # network unavailable, no cache, etc.
        pytest.skip(f"could not load gpt2 for this test: {exc!r}")
    return model, tokenizer, device
