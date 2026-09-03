"""Resume-safe shard cache for extracted activations.

Checkpoint granularity is (model, formatting_variant, shard-of-prompt-ids) — NOT
(model, layer, pooling_variant, formatting_variant) as a literal reading of the spec's
resume requirement might suggest. A single forward pass with output_hidden_states=True
already yields every layer at once, and both pooling variants are computed from the
same hidden states for free (extraction.py). Checkpointing at layer/pooling
granularity would mean re-running identical forward passes for no reason; this reads
the spec's intent (never repeat a forward pass already done) rather than its letter.
Flagged in the build plan for Fjord to overrule if a literal per-layer checkpoint was
actually wanted.

Each shard is one .npz file: `prompt_ids` (fixed-width unicode array of str — NOT an
object array; object arrays would need `allow_pickle=True` to load, and pickle is
avoided deliberately since these files may later live on a shared Drive/Kaggle-output
folder), `masked_mean` and
`last_token` (float32 [n, n_layers_plus_1, hidden]). Shards are named by a content
hash of their sorted prompt-id set, written atomically (temp file + rename) so an
eviction mid-write can never leave a corrupt shard for the resume logic to trip over.
Resume works by reading which prompt_ids already exist across all shards for a given
(model, formatting_variant) and skipping them — the shards themselves are the source of
truth, no separate index file to fall out of sync.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from src import extraction


def _variant_dir(cache_dir: Path, model_key: str, formatting_variant: str) -> Path:
    return Path(cache_dir) / model_key / formatting_variant


def _shard_name(prompt_ids: list[str]) -> str:
    digest = hashlib.sha256("\x00".join(sorted(prompt_ids)).encode("utf-8")).hexdigest()[:16]
    return f"shard_{digest}.npz"


def completed_prompt_ids(cache_dir: Path, model_key: str, formatting_variant: str) -> set[str]:
    variant_dir = _variant_dir(cache_dir, model_key, formatting_variant)
    if not variant_dir.exists():
        return set()
    done: set[str] = set()
    for shard_path in variant_dir.glob("shard_*.npz"):
        try:
            with np.load(shard_path) as data:
                done.update(str(pid) for pid in data["prompt_ids"])
        except (zipfile.BadZipFile, EOFError):
            # A shard that fails to load as a valid archive is treated as
            # not-completed rather than raising — this is the one failure mode the
            # atomic temp-file+rename write strategy cannot fully rule out (e.g. disk
            # corruption). Any other exception (KeyError, a pickling error, etc.)
            # indicates a real bug in how the shard was written and is left to
            # propagate rather than being silently treated as "not done" — that
            # swallowing is exactly what hid the object-array/allow_pickle bug this
            # function used to have.
            continue
    return done


def write_shard(cache_dir: Path, model_key: str, formatting_variant: str,
                 prompt_ids: list[str], batch: extraction.BatchActivations) -> Path:
    variant_dir = _variant_dir(cache_dir, model_key, formatting_variant)
    variant_dir.mkdir(parents=True, exist_ok=True)

    final_path = variant_dir / _shard_name(prompt_ids)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    # np.savez silently appends ".npz" to any path that doesn't already end in ".npz"
    # (so a plain `np.savez(tmp_path, ...)` would write to "*.npz.tmp.npz", not
    # tmp_path, and the rename below would then raise FileNotFoundError). Passing an
    # open file handle instead of a path bypasses that auto-suffixing entirely.
    with open(tmp_path, "wb") as fh:
        np.savez(
            fh,
            prompt_ids=np.array(prompt_ids),  # plain str list -> fixed-width unicode dtype, no pickle
            masked_mean=batch.masked_mean.numpy().astype(np.float32),
            last_token=batch.last_token.numpy().astype(np.float32),
        )
    tmp_path.replace(final_path)  # atomic on POSIX and Windows (same filesystem)
    return final_path


def load_all_shards(cache_dir: Path, model_key: str, formatting_variant: str):
    """Returns (prompt_ids: list[str], masked_mean: np.ndarray, last_token: np.ndarray)
    concatenated across every shard for this (model, formatting_variant), in shard-file
    order (not the original extraction order — callers that need a specific prompt
    should index by prompt_id, not position).
    """
    variant_dir = _variant_dir(cache_dir, model_key, formatting_variant)
    prompt_ids: list[str] = []
    mm_chunks, lt_chunks = [], []
    for shard_path in sorted(variant_dir.glob("shard_*.npz")):
        with np.load(shard_path) as data:
            prompt_ids.extend(str(pid) for pid in data["prompt_ids"])
            mm_chunks.append(data["masked_mean"])
            lt_chunks.append(data["last_token"])
    if not mm_chunks:
        return [], np.empty((0,)), np.empty((0,))
    return prompt_ids, np.concatenate(mm_chunks, axis=0), np.concatenate(lt_chunks, axis=0)


def run_extraction_with_checkpointing(
    model, tokenizer, device: str, cache_dir: Path, model_key: str, formatting_variant: str,
    prompt_ids: list[str], texts: list[str], batch_size: int, max_length: int,
    progress_callback=None,
) -> None:
    """Resume-safe driver: filters out prompt_ids already cached for this
    (model_key, formatting_variant), then extracts and flushes the remainder one
    shard per batch. Safe to interrupt at any point and re-run — already-written
    shards are never touched or recomputed.
    """
    if len(prompt_ids) != len(texts):
        raise ValueError(f"prompt_ids ({len(prompt_ids)}) and texts ({len(texts)}) length mismatch")

    already_done = completed_prompt_ids(cache_dir, model_key, formatting_variant)
    remaining = [(pid, text) for pid, text in zip(prompt_ids, texts) if pid not in already_done]

    if progress_callback:
        progress_callback(total=len(prompt_ids), already_done=len(already_done), remaining=len(remaining))

    for start in range(0, len(remaining), batch_size):
        batch_items = remaining[start:start + batch_size]
        batch_ids = [pid for pid, _ in batch_items]
        batch_texts = [text for _, text in batch_items]

        batch_activations = extraction.extract_prompt_activations(
            model, tokenizer, batch_texts, device=device, max_length=max_length
        )
        write_shard(cache_dir, model_key, formatting_variant, batch_ids, batch_activations)

        if progress_callback:
            progress_callback(
                total=len(prompt_ids),
                already_done=len(already_done) + start + len(batch_items),
                remaining=len(remaining) - start - len(batch_items),
            )


def run_generation_extraction_with_checkpointing(
    model, tokenizer, device: str, cache_dir: Path, model_key: str,
    item_ids: list[str], prompt_texts: list[str], max_new_tokens: int,
    progress_callback=None,
) -> None:
    """Method 1 only. One shard per completed generation (item_ids are typically
    f"{question_id}__{condition}") — extraction.extract_generation_activations is
    unbatched by design (see its docstring), so there is no batch to flush together;
    writing a shard immediately after each generation still gives per-item resume
    safety, which is what matters at this cache's scale (~80 generations per model).
    formatting_variant is fixed to "generation" — this cache is never read by the
    prompt-activation methods (Methods 2/3), only by methods/tone_pole.py.
    """
    formatting_variant = "generation"
    if len(item_ids) != len(prompt_texts):
        raise ValueError(f"item_ids ({len(item_ids)}) and prompt_texts ({len(prompt_texts)}) length mismatch")

    already_done = completed_prompt_ids(cache_dir, model_key, formatting_variant)
    remaining = [(iid, text) for iid, text in zip(item_ids, prompt_texts) if iid not in already_done]

    if progress_callback:
        progress_callback(total=len(item_ids), already_done=len(already_done), remaining=len(remaining))

    for i, (item_id, prompt_text) in enumerate(remaining):
        batch_activations = extraction.extract_generation_activations(
            model, tokenizer, prompt_text, device=device, max_new_tokens=max_new_tokens
        )
        write_shard(cache_dir, model_key, formatting_variant, [item_id], batch_activations)

        if progress_callback:
            progress_callback(
                total=len(item_ids), already_done=len(already_done) + i + 1,
                remaining=len(remaining) - i - 1,
            )
