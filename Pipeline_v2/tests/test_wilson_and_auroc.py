"""Unit tests for the new analysis additions from the external review:
  - wilson_interval (Task 3)
  - AUROC + polarity check in _evaluate_arm / run_subtest_b (Task 1)
  - probe_on_subtest_b (Task 2)

These tests use the SAME synthetic-cache fixture pattern as
test_eval_pipeline_synthetic.py so no real model or GPU is needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import checkpoint, config, splits  # noqa: E402
from src.eval import subtest_a, subtest_b  # noqa: E402
from src.eval.bootstrap import wilson_interval  # noqa: E402
from src.extraction import BatchActivations  # noqa: E402

# ── Wilson interval ────────────────────────────────────────────────────────────

def test_wilson_interval_known_values():
    # 50% p-hat at n=100: centre near 0.5, CI roughly [0.40, 0.60]
    r = wilson_interval(50, 100)
    assert r["accuracy"] == pytest.approx(0.50, abs=1e-9)
    assert r["wilson_low"] < 0.50 < r["wilson_high"]
    assert 0.40 <= r["wilson_low"] <= 0.42
    assert 0.58 <= r["wilson_high"] <= 0.60


def test_wilson_interval_high_accuracy():
    # 23/24 pairs correct — the reported sub-test B headline result
    r = wilson_interval(23, 24)
    assert r["accuracy"] == pytest.approx(23 / 24, rel=1e-6)
    # Should be a reasonably wide CI given N=24, not artificially narrow
    assert r["wilson_high"] - r["wilson_low"] > 0.10
    assert r["wilson_low"] > 0.70   # well above chance
    assert r["wilson_high"] <= 1.0  # never exceeds 1


def test_wilson_interval_zero_successes():
    r = wilson_interval(0, 10)
    assert r["accuracy"] == 0.0
    assert r["wilson_low"] == 0.0
    assert r["wilson_high"] > 0.0  # not zero — Wilson stays off the boundary


def test_wilson_interval_all_successes():
    r = wilson_interval(10, 10)
    assert r["accuracy"] == 1.0
    assert r["wilson_high"] == 1.0
    assert r["wilson_low"] < 1.0


def test_wilson_interval_validates_inputs():
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(5, 0)
    with pytest.raises(NotImplementedError):
        wilson_interval(5, 10, confidence=0.90)


# ── Shared synthetic-cache fixture (mirrors test_eval_pipeline_synthetic.py) ──

MODEL_KEY = "synthetic-auroc"
FORMATTING_VARIANT = "raw"
N_LAYERS_PLUS_1 = 4
SIGNAL_LAYER = 2
HIDDEN_DIM = 32
SIGNAL_STRENGTH = 4.0
NOISE_STD = 1.0


def _make_activation_array(labels, positive_label, direction, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    arr = rng.normal(0, NOISE_STD, size=(n, N_LAYERS_PLUS_1, HIDDEN_DIM)).astype(np.float32)
    sign = np.array([1.0 if lbl == positive_label else -1.0 for lbl in labels])
    arr[:, SIGNAL_LAYER, :] += sign[:, None] * SIGNAL_STRENGTH * direction[None, :]
    return arr


def _write_cache(cache_dir, model_key, formatting_variant, item_ids, arr):
    tensor = torch.from_numpy(arr)
    batch = BatchActivations(masked_mean=tensor, last_token=tensor,
                              n_layers_plus_1=N_LAYERS_PLUS_1, hidden_dim=HIDDEN_DIM)
    checkpoint.write_shard(cache_dir, model_key, formatting_variant, item_ids, batch)


@pytest.fixture(scope="module")
def content_direction_auroc():
    rng = np.random.default_rng(42)
    d = rng.normal(size=HIDDEN_DIM)
    return d / np.linalg.norm(d)


@pytest.fixture(scope="module")
def normal_cache(tmp_path_factory, content_direction_auroc):
    """Cache where content_pole has the correct polarity (harmful scores high)."""
    cache_dir = tmp_path_factory.mktemp("normal_cache")
    rows = splits.load_combined_rows()
    ids_a = [r.prompt_id for r in rows]
    labels_a = [r.label for r in rows]
    arr_a = _make_activation_array(labels_a, "harmful", content_direction_auroc, seed=10)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, ids_a, arr_a)

    neutral_df = pd.read_csv(config.SUBTEST_B_NEUTRAL_CSV)
    b_neutral_ids = [f"{pid}__calm" for pid in neutral_df["PairID"]] + \
                    [f"{pid}__hostile" for pid in neutral_df["PairID"]]
    b_neutral_labels = ["neutral"] * (2 * len(neutral_df))
    arr_bn = _make_activation_array(b_neutral_labels, "harmful", content_direction_auroc, seed=30)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_neutral_ids, arr_bn)

    harmful_df = pd.read_csv(config.SUBTEST_B_HARMFUL_CSV)
    b_harmful_ids = [f"STB_H_{i:03d}__calm" for i in range(len(harmful_df))] + \
                    [f"STB_H_{i:03d}__hostile" for i in range(len(harmful_df))]
    b_harmful_labels = ["harmful"] * (2 * len(harmful_df))
    arr_bh = _make_activation_array(b_harmful_labels, "harmful", content_direction_auroc, seed=40)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_harmful_ids, arr_bh)

    return cache_dir


@pytest.fixture(scope="module")
def inverted_subtest_b_cache(tmp_path_factory, content_direction_auroc):
    """Cache where sub-test A is normal (content_pole fits correctly on sub-test A),
    but sub-test B items land on the OPPOSITE side of the threshold — neutral items
    score as if harmful, harmful items score as if neutral. This models the real
    inversion case: a method fit on sub-test A but applied out-of-distribution
    produces backwards predictions on sub-test B.

    AUROC on the combined arm should be < 0.5 (harmful scores lower than neutral).
    """
    inverted = -content_direction_auroc
    cache_dir = tmp_path_factory.mktemp("inverted_stb_cache")

    # Sub-test A: normal — harmful in +D, neutral in -D; pole fits correctly
    rows = splits.load_combined_rows()
    ids_a = [r.prompt_id for r in rows]
    labels_a = [r.label for r in rows]
    arr_a = _make_activation_array(labels_a, "harmful", content_direction_auroc, seed=10)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, ids_a, arr_a)

    # Sub-test B neutral: inverted → neutral items score HIGH (predicted "harmful")
    neutral_df = pd.read_csv(config.SUBTEST_B_NEUTRAL_CSV)
    b_neutral_ids = [f"{pid}__calm" for pid in neutral_df["PairID"]] + \
                    [f"{pid}__hostile" for pid in neutral_df["PairID"]]
    b_neutral_labels = ["neutral"] * (2 * len(neutral_df))
    # positive_label="harmful" with inverted direction → neutral items get +SIGNAL_STRENGTH * inverted
    # = -SIGNAL_STRENGTH * content_direction → score < 0 → wait, that's still "neutral"
    # We need neutral items to score POSITIVE (above threshold). Use "neutral" as positive_label.
    arr_bn = _make_activation_array(b_neutral_labels, "neutral", content_direction_auroc, seed=30)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_neutral_ids, arr_bn)

    # Sub-test B harmful: inverted → harmful items score LOW (predicted "neutral")
    harmful_df = pd.read_csv(config.SUBTEST_B_HARMFUL_CSV)
    b_harmful_ids = [f"STB_H_{i:03d}__calm" for i in range(len(harmful_df))] + \
                    [f"STB_H_{i:03d}__hostile" for i in range(len(harmful_df))]
    b_harmful_labels = ["harmful"] * (2 * len(harmful_df))
    # positive_label="neutral" → harmful items get -SIGNAL_STRENGTH * content_direction → score < 0
    arr_bh = _make_activation_array(b_harmful_labels, "neutral", content_direction_auroc, seed=40)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_harmful_ids, arr_bh)

    return cache_dir


# ── AUROC and polarity tests ───────────────────────────────────────────────────

def test_auroc_correct_polarity_normal_cache(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)

    cp = b_results["content_pole"]
    # Combined arms should have AUROC > 0.5 (correct polarity)
    assert cp["combined_arms_auroc"] is not None
    assert cp["combined_arms_auroc"] > 0.5, (
        f"Expected AUROC > 0.5 for correct-polarity cache, got {cp['combined_arms_auroc']}"
    )
    assert "correct" in cp["combined_arms_polarity_note"]


def test_auroc_inverted_polarity_detected(inverted_subtest_b_cache):
    """When sub-test B items land on the wrong side of the threshold (out-of-distribution
    inversion), the combined-arm AUROC should be < 0.5: harmful items score LOWER than
    neutral items under the pole fitted on sub-test A."""
    a_results = subtest_a.run_subtest_a(inverted_subtest_b_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(inverted_subtest_b_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)

    cp = b_results["content_pole"]
    assert cp["combined_arms_auroc"] is not None
    assert cp["combined_arms_auroc"] < 0.5, (
        f"Expected AUROC < 0.5 for inverted sub-test B cache, got {cp['combined_arms_auroc']}"
    )
    assert "inverted" in cp["combined_arms_polarity_note"]


def test_neutral_arm_auroc_is_none(normal_cache):
    # The neutral arm alone (all one class) must return auroc=None
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    neutral_arm = b_results["content_pole"]["neutral_arm"]
    assert neutral_arm["auroc"] is None, (
        "neutral_arm auroc must be None (single class — AUROC undefined)"
    )
    assert "single-class" in neutral_arm["polarity_note"]


def test_wilson_ci_present_in_neutral_arm(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    neutral_arm = b_results["content_pole"]["neutral_arm"]
    assert "wilson" in neutral_arm
    assert "wilson_low" in neutral_arm["wilson"]
    assert "wilson_high" in neutral_arm["wilson"]
    assert neutral_arm["wilson"]["wilson_low"] <= neutral_arm["wilson"]["wilson_high"]


def test_fraction_predicted_harmful_present(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    neutral_arm = b_results["content_pole"]["neutral_arm"]
    assert "fraction_predicted_harmful" in neutral_arm
    assert 0.0 <= neutral_arm["fraction_predicted_harmful"] <= 1.0


# ── Probe on sub-test B (Task 2) ───────────────────────────────────────────────

def test_probe_on_subtest_b_present_in_results(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    assert "probe_on_subtest_b" in b_results["content_pole"]
    probe_r = b_results["content_pole"]["probe_on_subtest_b"]
    assert 0.0 <= probe_r["accuracy"] <= 1.0
    assert probe_r["n_test"] == 48  # 24 pairs x 2 items
    assert "wilson" in probe_r


def test_probe_on_subtest_b_high_accuracy_with_strong_signal(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    probe_r = b_results["content_pole"]["probe_on_subtest_b"]
    # With a strong synthetic signal the probe should generalise well to sub-test B
    assert probe_r["accuracy"] > 0.8, (
        f"Probe accuracy {probe_r['accuracy']} unexpectedly low on a strong synthetic signal"
    )


def test_summarize_a_b_gap_includes_wilson_and_probe(normal_cache):
    a_results = subtest_a.run_subtest_a(normal_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(normal_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    gap = subtest_b.summarize_a_b_gap(a_results, b_results)
    cp = gap["content_pole"]
    assert "subtest_b_neutral_arm_ci_wilson" in cp
    assert "subtest_b_neutral_arm_pairs_correct" in cp
    assert "subtest_b_combined_auroc" in cp
    # Pairs-correct string should include a fraction like "X/24 pairs"
    assert "/24 pairs" in cp["subtest_b_neutral_arm_pairs_correct"]
