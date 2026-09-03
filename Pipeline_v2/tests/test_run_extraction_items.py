"""Pure data-plumbing checks for run_extraction.build_subtest_ab_items — no model
needed for the "raw" formatting variant (format_raw ignores the tokenizer entirely),
so this stays fast and catches ID-construction bugs (duplicates, wrong counts) before
they'd otherwise only surface deep inside a real extraction run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import run_extraction  # noqa: E402

EXPECTED_SUBTEST_A_COUNT = 550  # 250 harmbench + 300 neutral
EXPECTED_SUBTEST_B_NEUTRAL_COUNT = 48  # 24 pairs x (calm, hostile)
EXPECTED_SUBTEST_B_HARMFUL_COUNT = 18  # 9 sourced pairs x (calm, hostile)


def test_raw_items_have_expected_count_and_no_duplicates():
    item_ids, texts = run_extraction.build_subtest_ab_items(tokenizer=None, formatting_variant="raw")

    expected_total = (
        EXPECTED_SUBTEST_A_COUNT + EXPECTED_SUBTEST_B_NEUTRAL_COUNT + EXPECTED_SUBTEST_B_HARMFUL_COUNT
    )
    assert len(item_ids) == len(texts) == expected_total
    assert len(set(item_ids)) == len(item_ids), "duplicate item_id in the combined extraction list"


def test_raw_formatting_is_the_identity():
    item_ids, texts = run_extraction.build_subtest_ab_items(tokenizer=None, formatting_variant="raw")
    from src.splits import load_combined_rows

    text_by_id = {row.prompt_id: row.text for row in load_combined_rows()}
    for item_id, text in zip(item_ids, texts):
        if item_id in text_by_id:
            assert text == text_by_id[item_id]


def test_subtest_b_item_ids_use_the_documented_scheme():
    item_ids, _texts = run_extraction.build_subtest_ab_items(tokenizer=None, formatting_variant="raw")
    id_set = set(item_ids)

    assert "STB_N_001__calm" in id_set and "STB_N_001__hostile" in id_set
    assert "STB_H_000__calm" in id_set and "STB_H_000__hostile" in id_set


def test_unknown_formatting_variant_rejected():
    import pytest

    with pytest.raises(ValueError):
        run_extraction.build_subtest_ab_items(tokenizer=None, formatting_variant="not_a_real_variant")
