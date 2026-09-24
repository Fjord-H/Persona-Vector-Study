"""
Experiment 1 prerequisite: extract activations for the v1 2000-item corpus
(200 training items + 1800 test items, data/vectors/dataset_2000.pkl) through
v2's clean extraction pipeline (proper masked_mean/last_token pooling, correct
layer indexing -- NOT v1's original unmasked mean-pooling, which
defect_report.md S2-04/S2-05 already flagged as length-channel-dominated and
padding-contaminated).

WHY THIS EXISTS
----------------
Verified directly (see HANDOVER_v3.md's "Data inventory" section, 2026-09-25):
the existing v2 activation cache (pv2_cache_kaggle/, pv2_cache_nb2/) contains
ZERO items from this corpus -- it only has sub-test A/B/v3 content, checked
against every cached prompt_id. v1's own artifacts (data/vectors/*.pkl) only
hold pre-collapsed per-item SCORES (a single cosine-similarity scalar), never
raw activation vectors, and only for at most 2 of 5 models (GPT-2: all 24
layers; Qwen-Instruct: layer 1 only; Qwen-base and both Llama variants: none
at all). A fresh LR probe needs the full feature vector, which does not exist
anywhere in v1's artifacts for any model. Experiment 1 (label-preserving
partitions of this 2000-item pool, threshold-transfer + recalibration across
content_pole/probe/TF-IDF x every model) cannot run at full cross-model scope
without this extraction.

WHERE TO RUN THIS
------------------
Does NOT run in the Cowork device-bridge VM this project's other analysis
scripts use -- that VM has no GPU, ~3.8GB RAM (has already OOM'd once in this
project just loading existing caches), and no HF auth for the gated Llama
checkpoints. Run this instead:
  (a) on Kaggle, same setup as the existing Phase 3 workflow (HANDOVER_v3.md):
      GPU T4, HF_TOKEN as a Kaggle secret exported to the environment, or
  (b) via a local Claude Code session on a machine with a real GPU (or enough
      CPU/RAM to be tolerable -- these are small models) and your own HF
      token exported as the HF_TOKEN environment variable.
Requires: torch, transformers, huggingface_hub (same deps as the rest of
Pipeline_v2 -- see notebooks/01_extract_activations.ipynb for the exact
install cell used on Kaggle).

Compute scale, for planning: 2000 items x 1 formatting variant (raw) for the
3 base-ish models (gpt2-medium, qwen2.5-1.5b, llama-3.2-3b -- "ish" because
formatting.available_formatting_variants checks the tokenizer's actual
chat_template attribute, not the registry's is_instruct flag, exactly like
the rest of this pipeline already does) and 2000 items x 2 variants (raw +
chat) for the 2 instruct models = ~14,000 forward passes total across all 5
models, batched at DEFAULT_BATCH_SIZE=16. This is smaller than the existing
Phase 3 plan's 5,200-item x more-mutations extraction (~1 GPU-hour/model on a
T4), so each model here should take noticeably less than that per model.

OUTPUT / HOW TO MERGE BACK
----------------------------
Writes into the SAME cache format and directory layout as the rest of v2
(checkpoint.py's shard_*.npz files under <cache_dir>/<model_key>/<raw|chat>/),
so every existing v2 loader (load_matrix_lowmem in
analysis/review5_threshold_recalibration.py, activation_store.load_layer_matrix)
reads it with ZERO code changes. Point --cache-dir (or the PV2_CACHE_DIR env
var, same convention as the rest of this pipeline) at a fresh local directory
while extracting, then copy that directory's <model_key>/<raw|chat>/shard_*.npz
files into pv2_cache_kaggle/pv2_cache/<model_key>/<raw|chat>/ (or wherever
Experiment 1's analysis script points PV2_CACHE_DIR) -- plain file copies, no
merge logic needed, since shard filenames are content-hashed and therefore
automatically distinct from every existing sub-test A/B/v3 shard.

Item IDs use the "V1POOL:<safe|dangerous>:<4-digit index>" scheme, indexed
into dataset_2000.pkl's safe_queries/dangerous_queries lists in their
ORIGINAL list order (NOT the train/test split order, and verified to have
zero duplicate strings within or across the two 1000-item lists) -- this is
deliberate: the same 2000 items are extracted exactly once here, and
train_test_split.pkl's 200/1800 train/test membership is looked up by exact
text match at ANALYSIS time (Experiment 1's own script does this), so this
extraction script does not need to know or care about the split. This also
means "V1POOL:safe:0000" is NOT the same item as any "V1_TRAIN_SAFE_000" id
used in the earlier Task 1 data-integrity audit scripts -- those were assigned
by position within the already-split 200-item train subset; these are
assigned by position within the full 2000-item pool before any split.

USAGE
-----
    cd Pipeline_v2 && python3 ../analysis/experiment1_extract_v1_pool.py <model_key> [--cache-dir DIR]

    model_key is one of: gpt2-medium, qwen2.5-1.5b, qwen2.5-1.5b-instruct,
    llama-3.2-3b, llama-3.2-3b-instruct (Pipeline_v2/src/config.py's
    MODEL_REGISTRY). Run one model per invocation/session -- same
    recommendation as the existing Phase 3 plan ("Run GPT-2 first as a smoke
    test," since it's CPU-feasible and needs no HF auth).

    Resume-safe: safe to interrupt and re-run, exactly like run_extraction.py's
    extract_all_for_model (same checkpoint.py machinery underneath -- it skips
    any V1POOL item_id already present in a shard for that model/variant).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "Pipeline_v2"
sys.path.insert(0, str(PIPELINE))

from src.checkpoint import run_extraction_with_checkpointing  # noqa: E402
from src.config import CACHE_DIR, DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, MODEL_REGISTRY, POOLING_VARIANTS  # noqa: E402
from src.formatting import available_formatting_variants, format_chat, format_raw  # noqa: E402
from src.manifest import build_run_manifest, write_run_manifest  # noqa: E402
from src.model_io import load_model_and_tokenizer  # noqa: E402

DATASET_2000_PKL = REPO_ROOT / "data" / "vectors" / "dataset_2000.pkl"


def load_v1_pool_raw_items() -> list[tuple[str, str, str]]:
    """Returns (item_id, text, class_label) for all 2000 items, class_label in
    {"safe", "dangerous"} matching dataset_2000.pkl's own field names."""
    with open(DATASET_2000_PKL, "rb") as f:
        d = pickle.load(f)
    items: list[tuple[str, str, str]] = []
    for i, text in enumerate(d["safe_queries"]):
        items.append((f"V1POOL:safe:{i:04d}", text, "safe"))
    for i, text in enumerate(d["dangerous_queries"]):
        items.append((f"V1POOL:dangerous:{i:04d}", text, "dangerous"))
    return items


def build_v1_pool_items(tokenizer, formatting_variant: str) -> tuple[list[str], list[str]]:
    if formatting_variant == "raw":
        format_fn = format_raw
    elif formatting_variant == "chat":
        format_fn = lambda text: format_chat(tokenizer, text)  # noqa: E731
    else:
        raise ValueError(f"unknown formatting_variant {formatting_variant!r}")

    raw_items = load_v1_pool_raw_items()
    item_ids = [item_id for item_id, _, _ in raw_items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(
            "duplicate V1POOL item_id -- should be impossible given verified-unique "
            "source lists; check dataset_2000.pkl for a changed safe_queries/"
            "dangerous_queries list before re-running"
        )
    texts = [format_fn(text) for _, text, _ in raw_items]
    return item_ids, texts


def extract_v1_pool_for_model(model_key: str, cache_dir: Path, batch_size: int, max_length: int) -> None:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model_key {model_key!r}; see Pipeline_v2/src/config.py MODEL_REGISTRY")
    spec = MODEL_REGISTRY[model_key]

    print(f"Loading {spec.hf_id} ...")
    model, tokenizer, device = load_model_and_tokenizer(spec)
    print(f"Loaded on device={device}")

    for formatting_variant in available_formatting_variants(tokenizer):
        item_ids, texts = build_v1_pool_items(tokenizer, formatting_variant)
        print(
            f"\n{model_key}/{formatting_variant}: extracting {len(item_ids)} V1POOL items "
            f"(the same 2000 underlying items; the chat variant renders them through "
            f"the chat template rather than being a different set)"
        )

        def progress(total, already_done, remaining):
            print(f"  total={total} already_done={already_done} remaining={remaining}")

        run_extraction_with_checkpointing(
            model, tokenizer, device, cache_dir, model_key, formatting_variant,
            item_ids, texts, batch_size=batch_size, max_length=max_length,
            progress_callback=progress,
        )
        run_manifest = build_run_manifest(
            spec, tokenizer, formatting_variant, POOLING_VARIANTS,
            n_prompts_extracted=len(item_ids), batch_size=batch_size, max_length=max_length,
            extra={
                "purpose": "experiment_1_v1_pool_extraction",
                "source": "data/vectors/dataset_2000.pkl (v1 corpus, NOT harmbench_filtered_250.csv/neutral_set_300.csv "
                           "-- the dataset_sha256 field above is inherited from build_run_manifest's fixed hashing of "
                           "those two sub-test-A source files and is not meaningful for this run)",
                "item_id_scheme": "V1POOL:<safe|dangerous>:<4-digit index into dataset_2000.pkl's own list order>",
            },
        )
        manifest_path = write_run_manifest(run_manifest)
        print(f"  done, manifest written to {manifest_path}")

    print(f"\n{model_key}: all available formatting variants extracted for the V1POOL corpus.")
    print(f"Cache written under: {cache_dir}/{model_key}/")
    print("Copy that model's shard_*.npz files into pv2_cache_kaggle/pv2_cache/<model_key>/<variant>/ "
          "(or wherever the analysis scripts point PV2_CACHE_DIR) to make them usable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract v2-clean activations for the v1 2000-item corpus (Experiment 1 prerequisite).",
    )
    parser.add_argument("model_key", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--cache-dir", default=None,
        help="Output cache directory. Defaults to Pipeline_v2/src/config.py's CACHE_DIR "
             "(itself overridable via the PV2_CACHE_DIR env var, same convention as the rest of this pipeline).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    print(f"Cache dir: {cache_dir}")
    extract_v1_pool_for_model(args.model_key, cache_dir, args.batch_size, args.max_length)
