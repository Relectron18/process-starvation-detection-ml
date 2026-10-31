"""
Control for Section 5.6(b): withholding waiting_time_ms/wait_to_burst_ratio
(the columns pk_wait_anomaly is most correlated with, r=0.27-0.34) turns
pk_wait_anomaly's ablation delta significantly positive. But withholding
*any* two columns shrinks the feature set by the same amount, so part of
that effect could be "fewer competing features in general" rather than
specifically "redundancy with the withheld columns."

This script re-runs the identical reduced-access ablation, but withholds
two columns with near-zero correlation with pk_wait_anomaly instead
(memory_usage_percent, num_threads; |r| < 0.005 with pk_wait_anomaly,
picked by scanning the full correlation matrix -- see
pk_wait_anomaly_correlations_control.csv). If pk_wait_anomaly's ablation
delta stays null (or negative) here, that's evidence the Section 5.6(b)
result is specifically about redundancy with the withheld columns, not a
generic "smaller feature set helps" artifact.

Same pipeline, same 12-seed set, same leak-free split/statistics fitting
as feature_access_and_model_family_check.py, for direct comparability.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import OUTPUT_DIR
from pipeline_core import split_raw, fit_split_stats, apply_split_stats, fit_encoders_scaler, get_models
from multiseed_stability import cohens_d_one_sample

# Near-zero correlation with pk_wait_anomaly (|r| < 0.005 at seed=42; see
# module docstring). Two columns, matching the count withheld in 5.6(b).
CONTROL_WITHHELD = ['memory_usage_percent', 'num_threads']

DEFAULT_SEEDS = [42, 7, 123, 2024, 99, 555, 777, 888, 314, 271, 161, 202]


def run_experiment(seeds):
    access_rows, corr_rows = [], []

    for seed in seeds:
        train_df, test_df = split_raw(seed)
        split_stats = fit_split_stats(train_df)
        train_df = apply_split_stats(train_df, split_stats)
        X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_df)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        corr = X_train[CONTROL_WITHHELD + ['pk_wait_anomaly']].corr()['pk_wait_anomaly'].drop('pk_wait_anomaly')
        corr_rows.append({'seed': seed, **corr.to_dict()})

        reduced_cols = [c for c in X_train.columns if c not in CONTROL_WITHHELD]
        reduced_with_pk = reduced_cols
        reduced_no_pk = [c for c in reduced_cols if c != 'pk_wait_anomaly']

        rf = get_models(seed)['Random Forest']
        cv_reduced_with_pk = cross_val_score(
            rf, X_train[reduced_with_pk], y_train, cv=cv, scoring='f1', n_jobs=-1
        ).mean()
        rf2 = get_models(seed)['Random Forest']
        cv_reduced_no_pk = cross_val_score(
            rf2, X_train[reduced_no_pk], y_train, cv=cv, scoring='f1', n_jobs=-1
        ).mean()
        access_rows.append({
            'seed': seed, 'cv_f1_control_reduced_with_pk': cv_reduced_with_pk,
            'cv_f1_control_reduced_no_pk': cv_reduced_no_pk,
            'delta': cv_reduced_with_pk - cv_reduced_no_pk,
        })
        print(f"seed={seed}  control reduced-access RF: with_pk={cv_reduced_with_pk:.4f} "
              f"no_pk={cv_reduced_no_pk:.4f}  delta={cv_reduced_with_pk - cv_reduced_no_pk:+.4f}")

    return pd.DataFrame(access_rows), pd.DataFrame(corr_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    access_df, corr_df = run_experiment(args.seeds)
    access_df.to_csv(str(OUTPUT_DIR / "control_ablation.csv"), index=False)
    corr_df.to_csv(str(OUTPUT_DIR / "pk_wait_anomaly_correlations_control.csv"), index=False)

    print("\n=== Control: RF without memory_usage_percent/num_threads (near-zero corr with pk_wait_anomaly), with vs without pk_wait_anomaly ===")
    delta = access_df['delta'].to_numpy()
    t, p = sstats.ttest_1samp(delta, 0.0)
    d = cohens_d_one_sample(delta)
    print(f"mean delta={delta.mean():+.4f}, std={delta.std():.4f}, t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")
    print(access_df)
    print("\ncorrelations of withheld columns with pk_wait_anomaly (should be near zero):")
    print(corr_df)


if __name__ == "__main__":
    main()
