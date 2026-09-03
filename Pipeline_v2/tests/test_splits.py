"""Invariants for data/splits.json: no id leakage across splits, near-duplicate groups
never straddle a split boundary within the main sets, per-label proportions land close
to the target 70/15/15, every source CSV's hash is recorded, and every sub-test B item
is held out (split="test") rather than trained or validated on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import splits  # noqa: E402

import pytest  # noqa: E402

# Computed once per test session (module scope) — the O(n^2) near-duplicate grouping
# over 550 prompts takes real time; every test in this file reuses the same result.


@pytest.fixture(scope="module")
def document():
    return splits.build_splits_document(seed=42)


@pytest.fixture(scope="module")
def main_set_records(document):
    main_sources = {"harmbench_filtered_250", "neutral_set_300"}
    return [a for a in document["assignments"] if a["source"] in main_sources]


def test_every_id_appears_exactly_once(document):
    ids = [a["id"] for a in document["assignments"]]
    assert len(ids) == len(set(ids))


def test_main_set_size_is_550(main_set_records):
    assert len(main_set_records) == 550


def test_subtest_b_sizes(document):
    neutral = [a for a in document["assignments"] if a["source"] == "subtest_b_neutral_tone_pairs"]
    harmful = [a for a in document["assignments"] if a["source"] == "subtest_b_harmful_tone_pairs_clean"]
    assert len(neutral) == 48  # 24 pairs x (calm, hostile)
    assert len(harmful) == 6   # 3 verified-clean pairs x (calm, hostile)


def test_only_expected_split_labels(document):
    splits_seen = {a["split"] for a in document["assignments"]}
    assert splits_seen == {"train", "val", "test"}


def test_subtest_b_is_entirely_held_out(document):
    for source in ("subtest_b_neutral_tone_pairs", "subtest_b_harmful_tone_pairs_clean"):
        records = [a for a in document["assignments"] if a["source"] == source]
        assert all(a["split"] == "test" for a in records), f"{source} has a non-test item"


def test_near_duplicate_groups_never_span_a_split(main_set_records):
    groups: dict[int, set[str]] = {}
    for record in main_set_records:
        groups.setdefault(record["group_id"], set()).add(record["split"])
    for group_id, splits_seen in groups.items():
        assert len(splits_seen) == 1, f"group {group_id} spans multiple splits: {splits_seen}"


def test_split_proportions_roughly_70_15_15_per_label(main_set_records):
    for label in ("harmful", "neutral"):
        subset = [a for a in main_set_records if a["label"] == label]
        n = len(subset)
        counts = {"train": 0, "val": 0, "test": 0}
        for a in subset:
            counts[a["split"]] += 1
        fractions = {k: v / n for k, v in counts.items()}
        # Loose bands: grouping (near-dup clusters kept intact) can shift the exact
        # split sizes; this only guards against a badly broken stratification.
        assert 0.60 <= fractions["train"] <= 0.80, f"{label} train fraction {fractions['train']}"
        assert 0.05 <= fractions["val"] <= 0.25, f"{label} val fraction {fractions['val']}"
        assert 0.05 <= fractions["test"] <= 0.25, f"{label} test fraction {fractions['test']}"


def test_source_file_hashes_present_for_all_four_sets(document):
    expected = {
        "harmbench_filtered_250.csv", "neutral_set_300.csv",
        "subtest_b_neutral_tone_pairs.csv", "subtest_b_harmful_tone_pairs.csv",
    }
    assert set(document["source_files"].keys()) == expected
    for entry in document["source_files"].values():
        assert len(entry["sha256"]) == 64  # sha256 hex digest length


def test_seed_and_ratios_recorded(document):
    assert document["seed"] == 42
    assert document["split_ratios"] == {"train": 0.70, "val": 0.15, "test": 0.15}


def test_split_sizes_summary_matches_assignments(document):
    for source, per_split in document["split_sizes"].items():
        expected_total = sum(per_split.values())
        actual_total = len([a for a in document["assignments"] if a["source"] == source])
        assert expected_total == actual_total


def test_deterministic_across_reruns():
    first = splits.build_splits_document(seed=42)
    second = splits.build_splits_document(seed=42)
    first_assignments = {a["id"]: a["split"] for a in first["assignments"]}
    second_assignments = {a["id"]: a["split"] for a in second["assignments"]}
    assert first_assignments == second_assignments


def test_different_seed_gives_a_different_main_set_split():
    default = {a["id"]: a["split"] for a in splits.build_splits_document(seed=42)["assignments"]}
    other = {a["id"]: a["split"] for a in splits.build_splits_document(seed=7)["assignments"]}
    disagreements = sum(1 for k in default if default[k] != other[k])
    assert disagreements / len(default) > 0.05


def test_write_and_load_round_trip(tmp_path):
    out_path = tmp_path / "splits.json"
    written = splits.write_splits_json(out_path, seed=42)
    assert written == out_path
    loaded = splits.load_splits_json(out_path, seed=42)
    assert loaded["seed"] == 42
    assert len(loaded["assignments"]) == 550 + 48 + 6


def test_load_frozen_splits_backcompat_view(tmp_path):
    out_path = tmp_path / "splits.json"
    splits.write_splits_json(out_path, seed=42)
    df = splits.load_frozen_splits(out_path, seed=42)
    assert set(df.columns) == {"PromptID", "label", "category", "group_id", "split"}
    assert len(df) == 550
