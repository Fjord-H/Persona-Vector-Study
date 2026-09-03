"""Invariants for the frozen train/val/test split: no PromptID leakage across splits,
near-duplicate groups never straddle a split boundary, and per-label proportions land
close to the target 60/20/20 (grouping distorts this somewhat, so the check is a loose
band, not an exact match).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import splits  # noqa: E402

# Computed once per test session (module scope) — the O(n^2) near-duplicate grouping
# over 550 prompts takes real time; every test in this file reuses the same result.
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def frozen_splits_df():
    return splits.create_frozen_splits(seed=42)


def test_every_prompt_id_appears_exactly_once(frozen_splits_df):
    ids = frozen_splits_df["PromptID"]
    assert ids.is_unique
    assert len(ids) == 550


def test_only_expected_split_labels(frozen_splits_df):
    assert set(frozen_splits_df["split"].unique()) <= {"train", "val", "test"}
    assert set(frozen_splits_df["split"].unique()) == {"train", "val", "test"}


def test_near_duplicate_groups_never_span_a_split(frozen_splits_df):
    for group_id, group in frozen_splits_df.groupby("group_id"):
        assert group["split"].nunique() == 1, (
            f"group {group_id} spans multiple splits: {group['split'].unique()}"
        )


def test_split_proportions_roughly_60_20_20_per_label(frozen_splits_df):
    for label in ("harmful", "neutral"):
        subset = frozen_splits_df[frozen_splits_df["label"] == label]
        n = len(subset)
        fractions = subset["split"].value_counts(normalize=True)
        # Loose bands: grouping (near-dup clusters kept intact) can shift the exact
        # split sizes; this only guards against a badly broken stratification, not
        # exact 60/20/20 to the row.
        assert 0.5 <= fractions.get("train", 0) <= 0.7, f"{label} train fraction {fractions.get('train', 0)}"
        assert 0.1 <= fractions.get("val", 0) <= 0.3, f"{label} val fraction {fractions.get('val', 0)}"
        assert 0.1 <= fractions.get("test", 0) <= 0.3, f"{label} test fraction {fractions.get('test', 0)}"


def test_deterministic_across_reruns():
    first = splits.create_frozen_splits(seed=42)
    second = splits.create_frozen_splits(seed=42)
    assert first.equals(second)


def test_different_seed_gives_a_different_split():
    default = splits.create_frozen_splits(seed=42)
    other = splits.create_frozen_splits(seed=7)
    merged = default.merge(other, on="PromptID", suffixes=("_seed42", "_seed7"))
    disagreement_rate = (merged["split_seed42"] != merged["split_seed7"]).mean()
    assert disagreement_rate > 0.05
