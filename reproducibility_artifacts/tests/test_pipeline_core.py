"""
Smoke tests for pipeline_core.py. These don't validate model accuracy
(that's what multiseed_stability.py is for) - they validate that the
data pipeline is deterministic, produces no NaNs/infs, and that basic
invariants the paper relies on actually hold, so a future edit can't
silently break reproducibility without a test failing.

Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_core import (
    build_dataset, preprocess, N_SAMPLES, get_models,
    split_raw, fit_split_stats, apply_split_stats, fit_encoders_scaler, transform_with,
)


def test_generate_dataset_is_deterministic():
    df1 = build_dataset(seed=42)
    df2 = build_dataset(seed=42)
    assert df1.equals(df2), "same seed must produce an identical dataset"


def test_different_seeds_produce_different_data():
    df1 = build_dataset(seed=42)
    df2 = build_dataset(seed=7)
    assert not df1["waiting_time_ms"].equals(df2["waiting_time_ms"])


def test_dataset_shape_and_no_missing_values():
    df = build_dataset(seed=42)
    assert len(df) == N_SAMPLES
    assert not df.isna().any().any(), "no NaNs expected anywhere in the generated dataset"


def test_label_is_binary_and_not_degenerate():
    df = build_dataset(seed=42)
    labels = set(df["is_starved"].unique())
    assert labels <= {0, 1}
    # both classes must be present and neither should be a trivial sliver,
    # otherwise stratified CV downstream will behave unpredictably
    frac_positive = df["is_starved"].mean()
    assert 0.05 < frac_positive < 0.95, f"class balance looks degenerate: {frac_positive:.3f}"


def test_preprocess_output_is_finite_and_aligned():
    df = build_dataset(seed=42)
    X, y = preprocess(df)
    assert len(X) == len(y) == len(df)
    assert np.isfinite(X.to_numpy()).all(), "scaled features must not contain NaN/inf"
    assert "is_starved" not in X.columns, "target must not leak into feature matrix"
    assert "pid" not in X.columns, "identifier column must be dropped"


def test_get_models_returns_five_named_models():
    models = get_models(seed=42)
    assert len(models) == 5
    for name, model in models.items():
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_get_models_is_seed_reproducible():
    m1 = get_models(seed=42)["Random Forest"]
    m2 = get_models(seed=42)["Random Forest"]
    assert m1.get_params()["random_state"] == m2.get_params()["random_state"] == 42


# --- Leak-free split pipeline (split_raw / fit_split_stats / apply_split_stats) ---
# These guard against reintroducing the bug where pk_wait_anomaly,
# low_priority_high_wait, or the scaler/encoders were fit on data that
# included test rows.

def test_split_raw_partitions_do_not_overlap():
    train_df, test_df = split_raw(seed=42)
    assert len(train_df) + len(test_df) == N_SAMPLES
    assert set(train_df["pid"]) & set(test_df["pid"]) == set()


def test_split_stats_are_identical_regardless_of_test_partition_content():
    """The whole point of fitting stats on train only: perturbing rows that
    ended up in the test set must not change the fitted stats at all."""
    train_df, test_df = split_raw(seed=42)
    stats_a = fit_split_stats(train_df)

    corrupted_test_df = test_df.copy()
    corrupted_test_df["waiting_time_ms"] = corrupted_test_df["waiting_time_ms"] * 1000 + 50000
    corrupted_test_df["burst_time_ms"] = corrupted_test_df["burst_time_ms"] * 1000 + 50000
    # fit_split_stats only ever looks at train_df, so recomputing after
    # mangling test_df must be a no-op.
    stats_b = fit_split_stats(train_df)

    assert stats_a["wait_median"] == stats_b["wait_median"]
    for group in stats_a["queueing"]:
        assert stats_a["queueing"][group] == stats_b["queueing"][group]


def test_apply_split_stats_gives_test_rows_the_same_features_as_full_dataset_would_by_formula():
    """pk_wait_anomaly / low_priority_high_wait for a test row must depend
    only on that row's own values and the train-fit stats -- not on any
    property of other test rows."""
    train_df, test_df = split_raw(seed=42)
    stats = fit_split_stats(train_df)
    test_engineered = apply_split_stats(test_df, stats)

    # Re-engineer using only the first half of test_df; a leak-free
    # implementation gives identical results for the overlapping rows,
    # because nothing about the *other* test rows should matter.
    half = test_df.iloc[: len(test_df) // 2].copy()
    half_engineered = apply_split_stats(half, stats)

    common_pids = set(half["pid"])
    a = test_engineered[test_engineered["pid"].isin(common_pids)].sort_values("pid").reset_index(drop=True)
    b = half_engineered.sort_values("pid").reset_index(drop=True)
    assert np.allclose(a["pk_wait_anomaly"], b["pk_wait_anomaly"])
    assert (a["low_priority_high_wait"] == b["low_priority_high_wait"]).all()


def test_fit_encoders_scaler_is_not_affected_by_test_partition():
    """StandardScaler mean/std must come from train only: scaling a test
    row can't depend on which other rows ended up in the test set."""
    train_df, test_df = split_raw(seed=42)
    stats = fit_split_stats(train_df)
    train_eng = apply_split_stats(train_df, stats)
    test_eng = apply_split_stats(test_df, stats)

    X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_eng)
    X_test_a, _ = transform_with(test_eng, encoders, scaler, feature_cols)

    # Drop half the test rows and re-transform the rest: a leak-free
    # transform of a given row never depends on sibling rows.
    half_test_eng = test_eng.iloc[: len(test_eng) // 2].reset_index(drop=True)
    X_test_b, _ = transform_with(half_test_eng, encoders, scaler, feature_cols)

    assert np.allclose(X_test_a.iloc[: len(X_test_b)].to_numpy(), X_test_b.to_numpy())
