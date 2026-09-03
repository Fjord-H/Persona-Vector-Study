"""Raw vs. chat-templated text formatting (spec requirement 6: "both formatting
variants (raw AND chat-templated for instruct models)").

The chat-templated variant is only meaningful for a tokenizer that actually has a
chat_template — that is the ground truth checked here, not the model registry's
`is_instruct` flag, which only records intent. A model without a chat_template simply
has no chat-formatted extraction run for it; this is recorded explicitly in the run
manifest (manifest.py) as a documented skip, not silently omitted.
"""
from __future__ import annotations


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def format_raw(prompt: str) -> str:
    return prompt


def format_chat(tokenizer, prompt: str, system_prompt: str | None = None) -> str:
    """Render `prompt` (and optional `system_prompt`) through the tokenizer's chat
    template, with the generation prompt appended (i.e. ready for the assistant turn
    to begin), returned as plain text (not tokenized).
    """
    if not has_chat_template(tokenizer):
        raise ValueError(
            "format_chat called on a tokenizer with no chat_template; "
            "callers must check has_chat_template() first and skip this variant."
        )
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def available_formatting_variants(tokenizer) -> tuple[str, ...]:
    return ("raw", "chat") if has_chat_template(tokenizer) else ("raw",)
