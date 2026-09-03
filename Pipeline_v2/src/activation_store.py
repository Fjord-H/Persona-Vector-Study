"""Reads back the checkpoint.py shard cache and aligns it to a requested prompt_id
order — checkpoint.load_all_shards returns activations in shard-file order, which is
not meaningful to callers that need e.g. "the train-split rows, in split-file order."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import checkpoint


class ActivationLookupError(ValueError):
    pass


def load_layer_matrix(
    cache_dir: Path, model_key: str, formatting_variant: str, pooling_variant: str,
    layer: int, prompt_ids: list[str],
) -> np.ndarray:
    """Returns a [len(prompt_ids), hidden] float32 matrix, one row per prompt_id, in
    the exact order given. Raises ActivationLookupError listing any prompt_ids that
    aren't in the cache, rather than silently truncating or reordering.
    """
    if pooling_variant not in ("masked_mean", "last_token"):
        raise ValueError(f"unknown pooling_variant {pooling_variant!r}")

    all_ids, mm_all, lt_all = checkpoint.load_all_shards(cache_dir, model_key, formatting_variant)
    if not all_ids:
        raise ActivationLookupError(
            f"no cached activations found for model={model_key!r} formatting={formatting_variant!r} "
            f"under {cache_dir}"
        )
    arr = mm_all if pooling_variant == "masked_mean" else lt_all

    id_to_row = {pid: i for i, pid in enumerate(all_ids)}
    missing = [pid for pid in prompt_ids if pid not in id_to_row]
    if missing:
        raise ActivationLookupError(
            f"{len(missing)} prompt_id(s) not found in cache for model={model_key!r} "
            f"formatting={formatting_variant!r}: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    n_layers_plus_1 = arr.shape[1]
    if not (0 <= layer < n_layers_plus_1):
        raise ValueError(f"layer {layer} out of range [0, {n_layers_plus_1}) for this cache")

    rows = [id_to_row[pid] for pid in prompt_ids]
    return arr[rows, layer, :].astype(np.float32)


def available_layers(cache_dir: Path, model_key: str, formatting_variant: str) -> int:
    """Number of entries along the layer axis (n_layers + 1, per the config.py
    layer-indexing convention) present in the cache for this (model, formatting)."""
    all_ids, mm_all, _lt_all = checkpoint.load_all_shards(cache_dir, model_key, formatting_variant)
    if not all_ids:
        raise ActivationLookupError(
            f"no cached activations found for model={model_key!r} formatting={formatting_variant!r} "
            f"under {cache_dir}"
        )
    return mm_all.shape[1]
