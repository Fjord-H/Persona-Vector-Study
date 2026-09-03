"""End-to-end test of the eval orchestration (subtest_a, subtest_b, bootstrap,
length_correlation, and every method) against a SYNTHETIC activation cache — no real
model or GPU involved. This validates the plumbing (split discipline, validation-only
layer/pooling selection, bootstrap CIs, the random-direction null, and the sub-test B
N=3-arm handling) independently of whether any real extraction has ever run.

Synthetic design: a content direction separates harmful/neutral activations at exactly
one "signal layer" (all other layers are pure noise, so validation-based layer
selection has something real to find), used consistently for sub-test A and both
sub-test B arms. The tone-pole generation cache uses a direction constructed to be
EXACTLY ORTHOGONAL to the content direction, so Method 1's vector should score near
chance on content — the "expected to fail" control actually failing is itself part of
what this test checks.
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
from src.extraction import BatchActivations  # noqa: E402

MODEL_KEY = "synthetic-model"
FORMATTING_VARIANT = "raw"
N_LAYERS_PLUS_1 = 4
SIGNAL_LAYER = 2
HIDDEN_DIM = 32
SIGNAL_STRENGTH = 4.0
NOISE_STD = 1.0


def _orthogonal_direction(rng: np.random.Generator, reference: np.ndarray) -> np.ndarray:
    raw = rng.normal(size=reference.shape)
    raw -= (raw @ reference) * reference  # project out the component along `reference`
    return raw / np.linalg.norm(raw)


def _make_activation_array(labels: list[str], positive_label: str, direction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(labels)
    arr = rng.normal(0, NOISE_STD, size=(n, N_LAYERS_PLUS_1, HIDDEN_DIM)).astype(np.float32)
    sign = np.array([1.0 if lbl == positive_label else -1.0 for lbl in labels])
    arr[:, SIGNAL_LAYER, :] += sign[:, None] * SIGNAL_STRENGTH * direction[None, :]
    return arr


def _write_cache(cache_dir: Path, model_key: str, formatting_variant: str, item_ids: list[str], arr: np.ndarray):
    tensor = torch.from_numpy(arr)
    batch = BatchActivations(masked_mean=tensor, last_token=tensor,
                              n_layers_plus_1=N_LAYERS_PLUS_1, hidden_dim=HIDDEN_DIM)
    checkpoint.write_shard(cache_dir, model_key, formatting_variant, item_ids, batch)


@pytest.fixture(scope="module")
def content_direction():
    rng = np.random.default_rng(0)
    d = rng.normal(size=HIDDEN_DIM)
    return d / np.linalg.norm(d)


@pytest.fixture(scope="module")
def tone_direction(content_direction):
    return _orthogonal_direction(np.random.default_rng(1), content_direction)


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory, content_direction, tone_direction):
    cache_dir = tmp_path_factory.mktemp("synthetic_cache")

    # --- sub-test A content: real PromptIDs from the actual source CSVs ---
    rows = splits.load_combined_rows()
    ids_a = [r.prompt_id for r in rows]
    labels_a = [r.label for r in rows]
    arr_a = _make_activation_array(labels_a, positive_label="harmful", direction=content_direction, seed=10)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, ids_a, arr_a)

    # --- tone-pole generation cache: real question IDs, orthogonal direction ---
    questions = pd.read_csv(config.TONE_POLE_QUESTIONS_CSV)
    gen_ids = [f"{qid}__helpful_harmless" for qid in questions["QuestionID"]] + \
              [f"{qid}__harmful_dishonest" for qid in questions["QuestionID"]]
    gen_labels = ["helpful"] * len(questions) + ["harmful"] * len(questions)
    arr_gen = _make_activation_array(gen_labels, positive_label="harmful", direction=tone_direction, seed=20)
    _write_cache(cache_dir, MODEL_KEY, "generation", gen_ids, arr_gen)

    # --- sub-test B neutral arm: real PairIDs, same content direction, all neutral ---
    neutral_df = pd.read_csv(config.SUBTEST_B_NEUTRAL_CSV)
    b_neutral_ids = [f"{pid}__calm" for pid in neutral_df["PairID"]] + [f"{pid}__hostile" for pid in neutral_df["PairID"]]
    b_neutral_labels = ["neutral"] * (2 * len(neutral_df))
    arr_b_neutral = _make_activation_array(b_neutral_labels, positive_label="harmful",
                                            direction=content_direction, seed=30)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_neutral_ids, arr_b_neutral)

    # --- sub-test B harmful arm (all 9 rows cached; only 3 are "clean" per config) ---
    harmful_df = pd.read_csv(config.SUBTEST_B_HARMFUL_CSV)
    row_indices = list(range(len(harmful_df)))
    b_harmful_ids = [f"STB_H_{i:03d}__calm" for i in row_indices] + [f"STB_H_{i:03d}__hostile" for i in row_indices]
    b_harmful_labels = ["harmful"] * (2 * len(row_indices))
    arr_b_harmful = _make_activation_array(b_harmful_labels, positive_label="harmful",
                                            direction=content_direction, seed=40)
    _write_cache(cache_dir, MODEL_KEY, FORMATTING_VARIANT, b_harmful_ids, arr_b_harmful)

    return cache_dir


def test_content_methods_select_the_signal_layer_and_score_well(synthetic_cache):
    results = subtest_a.run_subtest_a(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)

    for method_name in ("content_pole", "neutral_origin_distance", "neutral_origin_direction"):
        assert method_name in results, f"{method_name} missing from results"
        fitted = results[method_name]
        assert fitted.layer == SIGNAL_LAYER, (
            f"{method_name} selected layer {fitted.layer}, expected the signal layer {SIGNAL_LAYER}"
        )
        assert fitted.test_result["accuracy"] > 0.9, (
            f"{method_name} test accuracy {fitted.test_result['accuracy']} unexpectedly low given a strong synthetic signal"
        )
        # The signal layer's score should track content, not length -- length here is
        # real HarmBench/neutral prompt length, uncorrelated with our synthetic signal
        # by construction, so this should NOT be flagged.
        assert not fitted.test_result["length_correlation"]["flagged"]


def test_tone_pole_fails_on_content_as_expected(synthetic_cache):
    results = subtest_a.run_subtest_a(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    assert "tone_pole" in results
    tone_pole_accuracy = results["tone_pole"].test_result["accuracy"]
    # Orthogonal construction means this should hover near chance (0.5); a loose band
    # keeps this from being a flaky test while still catching a real regression.
    assert 0.3 <= tone_pole_accuracy <= 0.7, (
        f"tone_pole scored {tone_pole_accuracy} on content -- expected near-chance for "
        f"a vector built from a direction orthogonal to the content signal"
    )


def test_random_direction_control_is_near_chance_and_below_content_pole(synthetic_cache):
    results = subtest_a.run_subtest_a(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    fitted = results["content_pole"]
    control = subtest_a.evaluate_random_direction_control(
        synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, fitted, seed=42, n_directions=20,
    )
    assert 0.3 <= control["mean_accuracy"] <= 0.7
    assert control["mean_accuracy"] < fitted.test_result["accuracy"]


def test_subtest_b_neutral_arm_reportable_and_harmful_arm_is_not(synthetic_cache):
    a_results = subtest_a.run_subtest_a(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)

    for method_name in ("content_pole", "neutral_origin_distance", "neutral_origin_direction"):
        neutral_arm = b_results[method_name]["neutral_arm"]
        assert neutral_arm["overall"]["accuracy"] > 0.9
        assert neutral_arm["overall"]["n"] == 48  # 24 pairs x calm+hostile
        assert "ci_low" in neutral_arm["overall"] and "ci_high" in neutral_arm["overall"]

        qualitative = b_results[method_name]["harmful_arm_qualitative"]
        assert qualitative["reportable"] is False
        assert qualitative["n_pairs"] == 3
        assert len(qualitative["per_item"]) == 6
        assert "ci_low" not in qualitative  # never bootstrap-CI'd, per spec


def test_a_b_gap_summary_is_small_for_content_methods(synthetic_cache):
    a_results = subtest_a.run_subtest_a(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, seed=42)
    b_results = subtest_b.run_subtest_b(synthetic_cache, MODEL_KEY, FORMATTING_VARIANT, a_results, seed=42)
    gap_summary = subtest_b.summarize_a_b_gap(a_results, b_results)

    for method_name in ("content_pole", "neutral_origin_distance", "neutral_origin_direction"):
        assert abs(gap_summary[method_name]["a_b_gap"]) < 0.15
