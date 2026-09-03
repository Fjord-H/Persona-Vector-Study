"""Frozen train/val/test split over the combined harmbench_filtered_250.csv +
neutral_set_300.csv set (sub-test A's content). Built once, seeded, and never
re-derived from a different seed later in the project — per spec requirement 1
("Frozen splits, created once") and requirement 2 ("selection on validation only,
reported once on test").

Near-duplicate prompts are grouped (via difflib.SequenceMatcher, the same method the
two source manifests used to check cross-set overlap) and always kept together in one
split, so no near-duplicate pair can straddle train/val/test. The two source manifests
already report this dataset as clean (one borderline pair at ratio 0.75 across all
550 x 550 comparisons) — grouping is applied anyway, as defense in depth per the
build plan, not because a problem is expected here.

Sub-test B (subtest_b_neutral_tone_pairs.csv, subtest_b_harmful_tone_pairs.csv) and the
tone-pole stimulus set (tone_pole_questions_40.csv) are NOT split by this module. They
are held out entirely: sub-test B is scored once using whatever threshold/layer/pooling
choice was selected on THIS module's validation split, never used to select anything
itself; the tone-pole set is small enough (40 questions) that all of it goes into
building Method 1's vector, with no train/test distinction inside Method 1's own
construction (matching the spec's minimal-pair design for the falsifying control).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_SEED, FROZEN_SPLITS_PATH, HARMBENCH_CSV, NEUTRAL_CSV

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
NEAR_DUP_RATIO_THRESHOLD = 0.75  # matches the ratio the source-data manifests flagged


@dataclass(frozen=True)
class CombinedRow:
    prompt_id: str
    text: str
    label: str      # "harmful" | "neutral"
    category: str    # SemanticCategory (harmful) or Category (neutral)


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


def create_frozen_splits(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Returns a DataFrame with columns [PromptID, label, category, group_id, split],
    one row per prompt in the combined 550-prompt set. Stratified by (label, category)
    at the GROUP level (an entire near-duplicate cluster is assigned to one split),
    with group order shuffled deterministically per stratum before slicing into
    train/val/test at TRAIN_FRAC/VAL_FRAC/TEST_FRAC.
    """
    rows = load_combined_rows()
    group_of = _group_near_duplicates(rows)

    # One representative row per group (label/category are group-uniform in practice
    # since near-dup groups are essentially always within-category; if a group somehow
    # spans two categories the first-seen row's category is used as the stratum key —
    # correctness of the split assignment doesn't depend on this choice, only its
    # readability does).
    rows_by_group: dict[int, list[CombinedRow]] = {}
    for row in rows:
        rows_by_group.setdefault(group_of[row.prompt_id], []).append(row)

    strata: dict[tuple[str, str], list[int]] = {}
    for group_id, group_rows in rows_by_group.items():
        key = (group_rows[0].label, group_rows[0].category)
        strata.setdefault(key, []).append(group_id)

    assignment: dict[int, str] = {}
    rng = random.Random(seed)
    for key, group_ids in strata.items():
        ordered = sorted(group_ids)  # deterministic base order before seeded shuffle
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        # remainder goes to test, so rounding never drops a group
        for idx, group_id in enumerate(ordered):
            if idx < n_train:
                assignment[group_id] = "train"
            elif idx < n_train + n_val:
                assignment[group_id] = "val"
            else:
                assignment[group_id] = "test"

    records = [
        {
            "PromptID": row.prompt_id,
            "label": row.label,
            "category": row.category,
            "group_id": group_of[row.prompt_id],
            "split": assignment[group_of[row.prompt_id]],
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def write_frozen_splits(output_path: Path, seed: int = DEFAULT_SEED) -> Path:
    df = create_frozen_splits(seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_frozen_splits(path: Path = FROZEN_SPLITS_PATH, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Reads the frozen split file written by write_frozen_splits(), computing and
    writing it first if it doesn't exist yet. Callers that need the split repeatedly
    (every eval/subtest_a.py run) should use this, not create_frozen_splits() directly
    -- the O(n^2) near-duplicate grouping in create_frozen_splits() takes real time
    (tens of seconds at n=550) and "frozen" means computed once, not once per call.
    """
    if not Path(path).exists():
        write_frozen_splits(path, seed=seed)
    return pd.read_csv(path, dtype={"PromptID": str}, keep_default_na=False)


if __name__ == "__main__":
    out = write_frozen_splits(FROZEN_SPLITS_PATH)
    df = pd.read_csv(out)
    print(f"wrote {out}")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
