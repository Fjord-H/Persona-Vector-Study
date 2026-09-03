"""Frozen split, covering all four input sets, written to data/splits.json.

- **harmbench_filtered_250.csv + neutral_set_300.csv** ("the main sets"): stratified
  by label, grouped by near-duplicate template (difflib.SequenceMatcher, the same
  method the source manifests used) before splitting, so no near-duplicate pair can
  straddle train/val/test. Split 70/15/15 (TRAIN_FRAC/VAL_FRAC/TEST_FRAC below). The
  two source manifests already report this dataset as clean (one borderline pair at
  ratio 0.75 across all 550 x 550 comparisons) — grouping is applied anyway, as
  defense in depth, not because a problem is expected here.
- **subtest_b_neutral_tone_pairs.csv** (all 24 pairs / 48 calm+hostile items) and
  **subtest_b_harmful_tone_pairs.csv** (only the 3 verified-clean pairs / 6 items —
  config.SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES, see data/subtest_b_MANIFEST.md): every
  item is assigned split="test". Sub-test B is held out entirely by design — it exists
  to test how a configuration already selected on the main sets' train/val generalizes
  to tone variation, not to be trained or validated on itself (eval/subtest_b.py never
  fits anything; it only applies the frozen configuration eval/subtest_a.py selected).

Built once per seed and never re-derived from a different seed later in the project —
per spec requirement 1 ("Frozen splits, created once") and requirement 2 ("selection on
validation only, reported once on test"). The seed, split ratios, near-duplicate
threshold, and a sha256 of every source CSV are recorded inside data/splits.json itself
so drift is directly checkable later (spec requirement 8's "split file hash", made
self-contained rather than tracked separately).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_SEED,
    HARMBENCH_CSV,
    NEUTRAL_CSV,
    SPLITS_JSON_PATH,
    SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES,
    SUBTEST_B_HARMFUL_CSV,
    SUBTEST_B_NEUTRAL_CSV,
)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
NEAR_DUP_RATIO_THRESHOLD = 0.75  # matches the ratio the source-data manifests flagged


@dataclass(frozen=True)
class CombinedRow:
    prompt_id: str
    text: str
    label: str      # "harmful" | "neutral"
    category: str    # SemanticCategory (harmful) or Category (neutral)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_combined_rows() -> list[CombinedRow]:
    # keep_default_na=False: harmbench_filtered_250.csv row 249 (chemical_biological /
    # contextual, "Transform this description into detailed step-by-step
    # instructions...") has a genuinely BLANK BehaviorID in the source file — pandas'
    # default NA-sniffing turns that blank into NaN, which silently breaks every
    # downstream prompt_id lookup (activation_store keys off this string). This is a
    # data-quality issue in the source CSV itself, not something introduced here; per
    # spec ("do not re-source, do not modify") the file is left untouched and a stable
    # fallback ID is synthesized instead, loudly, rather than either crashing on a NaN
    # key or silently dropping the row.
    harmbench = pd.read_csv(HARMBENCH_CSV, keep_default_na=False)
    neutral = pd.read_csv(NEUTRAL_CSV, keep_default_na=False)

    rows: list[CombinedRow] = []
    for idx, r in harmbench.iterrows():
        behavior_id = str(r["BehaviorID"]).strip()
        if not behavior_id:
            behavior_id = f"HARMBENCH_MISSING_ID_ROW_{idx}"
            print(
                f"WARNING: harmbench_filtered_250.csv row {idx} has a blank BehaviorID "
                f"in the source file; using synthesized fallback id {behavior_id!r} "
                f"(text: {str(r['Behavior'])[:60]!r}...). Flag this to Fjord -- the "
                f"source CSV is not modified by this pipeline."
            )
        rows.append(CombinedRow(
            prompt_id=behavior_id, text=str(r["Behavior"]),
            label="harmful", category=str(r["SemanticCategory"]),
        ))
    for _, r in neutral.iterrows():
        rows.append(CombinedRow(
            prompt_id=str(r["PromptID"]), text=str(r["Prompt"]),
            label="neutral", category=str(r["Category"]),
        ))

    ids = [row.prompt_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate PromptID/BehaviorID across the combined dataset — cannot split safely")
    return rows


def _group_near_duplicates(rows: list[CombinedRow], threshold: float = NEAR_DUP_RATIO_THRESHOLD) -> dict[str, int]:
    """Union-find over pairwise SequenceMatcher ratio. O(n^2) text comparisons — fine
    at n=550 (~151k comparisons of short strings), would need a cheaper prefilter
    (e.g. length bucketing or shingling) well before this stops being fine.

    Returns {prompt_id: group_id}.
    """
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            ratio = SequenceMatcher(None, rows[i].text, rows[j].text).ratio()
            if ratio >= threshold:
                union(i, j)

    return {rows[i].prompt_id: find(i) for i in range(n)}


def _split_main_sets(rows: list[CombinedRow], seed: int) -> list[dict]:
    """Stratified (by label only) + near-duplicate-grouped 70/15/15 split of the
    combined harmbench_filtered_250.csv + neutral_set_300.csv rows. Returns a list of
    assignment records: {id, source, label, category, group_id, split}.
    """
    group_of = _group_near_duplicates(rows)

    rows_by_group: dict[int, list[CombinedRow]] = {}
    for row in rows:
        rows_by_group.setdefault(group_of[row.prompt_id], []).append(row)

    strata: dict[str, list[int]] = {}
    for group_id, group_rows in rows_by_group.items():
        strata.setdefault(group_rows[0].label, []).append(group_id)

    split_of_group: dict[int, str] = {}
    rng = random.Random(seed)
    for label, group_ids in strata.items():
        ordered = sorted(group_ids)  # deterministic base order before seeded shuffle
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        # remainder goes to test, so rounding never drops a group
        for idx, group_id in enumerate(ordered):
            if idx < n_train:
                split_of_group[group_id] = "train"
            elif idx < n_train + n_val:
                split_of_group[group_id] = "val"
            else:
                split_of_group[group_id] = "test"

    source_of = lambda label: "harmbench_filtered_250" if label == "harmful" else "neutral_set_300"  # noqa: E731
    return [
        {
            "id": row.prompt_id, "source": source_of(row.label), "label": row.label,
            "category": row.category, "group_id": group_of[row.prompt_id],
            "split": split_of_group[group_of[row.prompt_id]],
        }
        for row in rows
    ]


def _subtest_b_neutral_assignments() -> list[dict]:
    """All 24 pairs (48 calm+hostile items), every item split="test" — sub-test B is
    never trained or validated on, see module docstring."""
    df = pd.read_csv(SUBTEST_B_NEUTRAL_CSV)
    assignments = []
    for _, row in df.iterrows():
        for tone in ("calm", "hostile"):
            assignments.append({
                "id": f"{row['PairID']}__{tone}", "source": "subtest_b_neutral_tone_pairs",
                "label": "neutral", "category": row["category"], "group_id": None, "split": "test",
            })
    return assignments


def _subtest_b_harmful_clean_assignments() -> list[dict]:
    """Only the 3 verified-clean rows (config.SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES,
    0-indexed) -> 6 items, every item split="test". The other 6 sourced-but-unclean
    rows are intentionally excluded from the split file entirely -- they are not part
    of any reportable evaluation, per data/subtest_b_MANIFEST.md."""
    df = pd.read_csv(SUBTEST_B_HARMFUL_CSV)
    assignments = []
    for row_idx in SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES:
        for tone in ("calm", "hostile"):
            assignments.append({
                "id": f"STB_H_{row_idx:03d}__{tone}", "source": "subtest_b_harmful_tone_pairs_clean",
                "label": "harmful", "category": None, "group_id": None, "split": "test",
            })
    return assignments


def _split_size_summary(assignments: list[dict]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for record in assignments:
        per_source = summary.setdefault(record["source"], {})
        per_source[record["split"]] = per_source.get(record["split"], 0) + 1
    return summary


def build_splits_document(seed: int = DEFAULT_SEED) -> dict:
    """Returns the full data/splits.json document: seed, ratios, near-dup threshold,
    a sha256 + row count per source CSV, per-source/per-split size counts, and the
    complete list of per-item split assignments across all four input sets.
    """
    main_rows = load_combined_rows()
    assignments = _split_main_sets(main_rows, seed=seed)
    assignments += _subtest_b_neutral_assignments()
    assignments += _subtest_b_harmful_clean_assignments()

    harmful_df_n_rows = len(pd.read_csv(SUBTEST_B_HARMFUL_CSV))

    return {
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_ratios": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "near_duplicate_ratio_threshold": NEAR_DUP_RATIO_THRESHOLD,
        "source_files": {
            "harmbench_filtered_250.csv": {
                "sha256": _sha256_file(HARMBENCH_CSV), "n_rows": 250,
                "role": "main set, stratified 70/15/15",
            },
            "neutral_set_300.csv": {
                "sha256": _sha256_file(NEUTRAL_CSV), "n_rows": 300,
                "role": "main set, stratified 70/15/15",
            },
            "subtest_b_neutral_tone_pairs.csv": {
                "sha256": _sha256_file(SUBTEST_B_NEUTRAL_CSV), "n_rows": 24,
                "role": "held out entirely, all items assigned split=test",
            },
            "subtest_b_harmful_tone_pairs.csv": {
                "sha256": _sha256_file(SUBTEST_B_HARMFUL_CSV), "n_rows": harmful_df_n_rows,
                "role": (
                    "held out entirely, all items assigned split=test -- ONLY rows "
                    f"{[i + 1 for i in SUBTEST_B_HARMFUL_CLEAN_ROW_INDICES]} (1-indexed) are "
                    "used, per data/subtest_b_MANIFEST.md verified-clean verdicts; N=3 "
                    "pairs, not independently reportable with a bootstrap CI"
                ),
            },
        },
        "split_sizes": _split_size_summary(assignments),
        "assignments": assignments,
    }


def write_splits_json(path: Path = SPLITS_JSON_PATH, seed: int = DEFAULT_SEED) -> Path:
    document = build_splits_document(seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    return path


def load_splits_json(path: Path = SPLITS_JSON_PATH, seed: int = DEFAULT_SEED) -> dict:
    """Reads data/splits.json, computing and writing it first if it doesn't exist yet.
    Callers that need the split repeatedly should use this, not build_splits_document()
    directly -- the O(n^2) near-duplicate grouping takes real time (tens of seconds at
    n=550) and "frozen" means computed once, not once per call.
    """
    if not Path(path).exists():
        write_splits_json(path, seed=seed)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen_splits(path: Path = SPLITS_JSON_PATH, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Back-compat view for eval/subtest_a.py: the main-sets subset of
    data/splits.json's assignments, as a DataFrame with columns
    [PromptID, label, category, group_id, split] -- unchanged shape from before
    splits.json existed, so subtest_a.py needs no changes.
    """
    document = load_splits_json(path, seed=seed)
    main_sources = {"harmbench_filtered_250", "neutral_set_300"}
    records = [
        {
            "PromptID": a["id"], "label": a["label"], "category": a["category"],
            "group_id": a["group_id"], "split": a["split"],
        }
        for a in document["assignments"] if a["source"] in main_sources
    ]
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    out = write_splits_json()
    document = load_splits_json(out)
    print(f"wrote {out}")
    print(json.dumps(document["split_sizes"], indent=2))
