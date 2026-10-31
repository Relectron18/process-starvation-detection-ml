import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_core import split_raw, fit_split_stats, apply_split_stats
from feature_access_and_model_family_check import (
    correlation_check, fresh_mlp, REDUNDANT_WITH_PK, ENGINEERED_COLS,
)


def _engineered_train_df(seed=42):
    """Leak-free equivalent of the old build_dataset(seed) for these tests:
    split first, fit stats on train only, apply to train."""
    train_df, _ = split_raw(seed)
    stats = fit_split_stats(train_df)
    return apply_split_stats(train_df, stats)


def test_redundant_columns_are_a_subset_of_engineered_and_raw_columns():
    # wait_to_burst_ratio is engineered; waiting_time_ms is a raw column.
    # Both must exist in the generated dataset for the ablation to be valid.
    df = _engineered_train_df(seed=42)
    for col in REDUNDANT_WITH_PK:
        assert col in df.columns
    assert 'wait_to_burst_ratio' in ENGINEERED_COLS
    assert 'waiting_time_ms' not in ENGINEERED_COLS  # it's a raw column, not engineered


def test_correlation_check_returns_finite_values_in_valid_range():
    df = _engineered_train_df(seed=42)
    corr = correlation_check(df)
    assert set(corr.index) == set(REDUNDANT_WITH_PK)
    assert np.isfinite(corr.to_numpy()).all()
    assert (corr.abs() <= 1.0).all()


def test_correlation_is_positive_by_construction():
    # pk_wait_anomaly = waiting_time_ms / (theory-predicted expected wait),
    # so it should be positively associated with both raw wait signals.
    df = _engineered_train_df(seed=42)
    corr = correlation_check(df)
    assert (corr > 0).all()


def test_fresh_mlp_is_seed_reproducible():
    m1 = fresh_mlp(seed=42)
    m2 = fresh_mlp(seed=42)
    assert m1.get_params()["random_state"] == m2.get_params()["random_state"] == 42
    assert hasattr(m1, "fit") and hasattr(m1, "predict")
