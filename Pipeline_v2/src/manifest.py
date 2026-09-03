"""Per-extraction-run manifest, per spec requirement 8: "model revision SHA,
transformers version, tokenizer, pooling variant, formatting variant, layer-indexing
convention (document that index 0 = embedding output), split file hash, dataset hash
(both CSVs above), extraction script git commit."

One manifest is written per (model, formatting_variant) extraction run — matching the
checkpoint granularity in checkpoint.py — plus a combined summary manifest that lists
every run for a full pipeline pass. Manifests are plain JSON so they're diffable and
readable without any of this project's own code.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import model_info

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    FROZEN_SPLITS_PATH,
    HARMBENCH_CSV,
    MANIFEST_DIR,
    ModelSpec,
    NEUTRAL_CSV,
)

LAYER_INDEXING_CONVENTION = (
    "output_hidden_states=True returns a tuple of length n_layers + 1. Index 0 is the "
    "embedding output (before any transformer block). Index n_layers is the final "
    "transformer block's output. This convention is used for every model in this "
    "pipeline without exception (v1's Qwen layer sweep skipped index 0 -- see "
    "defect_report.md S2-02 -- that asymmetry is deliberately not repeated here)."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return None


def _model_revision_sha(spec: ModelSpec) -> str | None:
    try:
        return model_info(spec.hf_id).sha
    except Exception:
        # Offline, rate-limited, or gated-without-access — recorded as unknown rather
        # than failing the whole extraction run over a metadata lookup.
        return None


def _dataset_hashes() -> dict:
    hashes = {"harmbench_filtered_250.csv": _sha256_file(HARMBENCH_CSV),
              "neutral_set_300.csv": _sha256_file(NEUTRAL_CSV)}
    return hashes


def _split_file_hash() -> str | None:
    if FROZEN_SPLITS_PATH.exists():
        return _sha256_file(FROZEN_SPLITS_PATH)
    return None


def build_run_manifest(
    spec: ModelSpec, tokenizer, formatting_variant: str, pooling_variants: tuple[str, ...],
    n_prompts_extracted: int, batch_size: int = DEFAULT_BATCH_SIZE, max_length: int = DEFAULT_MAX_LENGTH,
    extra: dict | None = None,
) -> dict:
    manifest = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "key": spec.key,
            "hf_id": spec.hf_id,
            "family": spec.family,
            "is_instruct": spec.is_instruct,
            "revision_sha": _model_revision_sha(spec),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "pad_token": tokenizer.pad_token,
            "padding_side": tokenizer.padding_side,
            "has_chat_template": getattr(tokenizer, "chat_template", None) is not None,
        },
        "formatting_variant": formatting_variant,
        "pooling_variants": list(pooling_variants),
        "layer_indexing_convention": LAYER_INDEXING_CONVENTION,
        "extraction_params": {"batch_size": batch_size, "max_length": max_length},
        "n_prompts_extracted": n_prompts_extracted,
        "dataset_sha256": _dataset_hashes(),
        "split_file_sha256": _split_file_hash(),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "git": {
            "commit": _git_commit(),
            "dirty_working_tree": _git_dirty(),
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_run_manifest(manifest: dict, manifest_dir: Path = MANIFEST_DIR) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    model_key = manifest["model"]["key"]
    formatting_variant = manifest["formatting_variant"]
    timestamp = manifest["written_at_utc"].replace(":", "-")
    path = manifest_dir / f"{model_key}__{formatting_variant}__{timestamp}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
