"""
Follow-up analysis addressing the paper's own stated Future Work item:
"investigate why the queueing-theory feature's expected benefit did not
materialize -- including whether it would matter more for model families
or feature-access conditions not tested here."

The notebook and multiseed_stability.py already ablate pk_wait_anomaly on
two model families (Random Forest, a model that can approximate feature
interactions on its own; Logistic Regression, a model that can't) and find
no measurable benefit either way. This script covers the two conditions
that were *not* tested:

  (a) Model family: MLPClassifier (a non-linear model that, unlike a tree
      ensemble, does not do implicit feature selection/splitting -- a genuinely
      different architecture from both models already tried).
  (b) Feature-access: what happens if the raw column(s) pk_wait_anomaly is
      most directly redundant with (waiting_time_ms, and the hand-built
      wait_to_burst_ratio) are withheld from the model entirely? If the
      RF/LR ablations came back near-zero because the model was reconstructing
      pk_wait_anomaly's signal from those raw columns, removing them should
      make pk_wait_anomaly's marginal contribution show up.

Also reports (i) the raw Pearson correlation between pk_wait_anomaly and
those same raw columns, and (ii) permutation importance of pk_wait_anomaly
under full vs. reduced feature access -- direct, causal-ish supporting
evidence for the redundancy hypothesis rather than relying on the ablation
deltas and correlation alone. Permutation importance (Breiman 2001, as
implemented by sklearn.inspection.permutation_importance) measures the
drop in held-out F1 when a single column's values are shuffled, which is
less biased than RandomForest's built-in impurity-based
feature_importances_ (which favors high-cardinality/continuous features).

Reuses split_raw/get_models from pipeline_core.py and the same seed set as
multiseed_stability.py (default 12 seeds -- increased from an earlier
5-seed default specifically to give the part-(b) test, which landed at a
borderline p=0.051 with 5 seeds, more power) for direct comparability with
multiseed_ablation.csv.

LEAKAGE FIX (see repro_artifacts CHANGES.md): previously used build_dataset
+ preprocess, which fit the queueing-theory stats, wait-time median, and
LabelEncoders/StandardScaler on the full dataset before any split -- so
test-partition rows leaked into the pk_wait_anomaly feature and the
scaling of every column. This matters most here: the positive finding in
part (b) below is exactly the kind of result that leakage could have
inflated, since it's the one place in the paper where pk_wait_anomaly
"wins". Now uses split_raw + fit_split_stats/apply_split_stats +
fit_encoders_scaler, all fit on the training partition only.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from config import OUTPUT_DIR
from pipeline_core import (
    split_raw, fit_split_stats, apply_split_stats, fit_encoders_scaler, get_models,
)
from multiseed_stability import cohens_d_one_sample

ENGINEERED_COLS = [
    'wait_to_burst_ratio', 'preemption_rate', 'cpu_starvation_gap',
    'low_priority_high_wait', 'pk_wait_anomaly',
]
# Raw/engineered column(s) pk_wait_anomaly is hypothesized to be most
# redundant with: it's waiting_time_ms scaled by a theory-predicted expected
# wait, and wait_to_burst_ratio is the closest hand-built analogue.
REDUNDANT_WITH_PK = ['waiting_time_ms', 'wait_to_burst_ratio']

DEFAULT_SEEDS = [42, 7, 123, 2024, 99, 555, 777, 888, 314, 271, 161, 202]


def fresh_mlp(seed):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-3,
        max_iter=500, early_stopping=True, random_state=seed,
    )


def correlation_check(df: pd.DataFrame) -> pd.Series:
    cols = REDUNDANT_WITH_PK + ['pk_wait_anomaly']
    return df[cols].corr()['pk_wait_anomaly'].drop('pk_wait_anomaly')


def pk_permutation_importance(X_train, y_train, seed, n_repeats=10):
    """Fits RF once on an internal train/holdout split of the training
    partition (never the real test set) and returns pk_wait_anomaly's
    permutation importance (mean F1 drop) under full feature access and
    under reduced access (waiting_time_ms/wait_to_burst_ratio withheld).
    A held-out split within X_train is used for the importance scoring
    itself, since permutation importance computed on training data is
    optimistic (the model can partly "memorize" a shuffled column's
    original association via other correlated columns seen in training)."""
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X_train, y_train, test_size=0.3, random_state=seed, stratify=y_train
    )

    # Full access
    rf_full = get_models(seed)['Random Forest']
    rf_full.fit(X_fit, y_fit)
    pi_full = permutation_importance(
        rf_full, X_hold, y_hold, scoring='f1', n_repeats=n_repeats,
        random_state=seed, n_jobs=-1,
    )
    full_importance = dict(zip(X_fit.columns, pi_full.importances_mean))

    # Reduced access
    reduced_cols = [c for c in X_fit.columns if c not in REDUNDANT_WITH_PK]
    rf_reduced = get_models(seed)['Random Forest']
    rf_reduced.fit(X_fit[reduced_cols], y_fit)
    pi_reduced = permutation_importance(
        rf_reduced, X_hold[reduced_cols], y_hold, scoring='f1', n_repeats=n_repeats,
        random_state=seed, n_jobs=-1,
    )
    reduced_importance = dict(zip(reduced_cols, pi_reduced.importances_mean))

    return full_importance, reduced_importance


def run_experiment(seeds):
    mlp_rows, access_rows, corr_rows, importance_rows = [], [], [], []

    for seed in seeds:
        train_df, test_df = split_raw(seed)
        split_stats = fit_split_stats(train_df)
        train_df = apply_split_stats(train_df, split_stats)
        # test_df / y_test are unused in this script (all comparisons are
        # cross-validated on the training partition), but we still avoid
        # computing pk_wait_anomaly stats from it -- fit_split_stats above
        # uses train_df only, consistent with every other script.
        X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_df)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        raw_cols = [c for c in X_train.columns if c not in ENGINEERED_COLS]

        # --- (a) model family: MLP ---
        cv_full = cross_val_score(fresh_mlp(seed), X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
        cv_raw = cross_val_score(fresh_mlp(seed), X_train[raw_cols], y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
        cv_no_pk = cross_val_score(
            fresh_mlp(seed), X_train.drop(columns=['pk_wait_anomaly']), y_train,
            cv=cv, scoring='f1', n_jobs=-1,
        ).mean()
        mlp_rows.append({
            'seed': seed, 'cv_f1_full': cv_full, 'cv_f1_raw_only': cv_raw,
            'cv_f1_no_pk': cv_no_pk, 'delta_full_vs_raw': cv_full - cv_raw,
        })

        # --- (b) feature-access: withhold the raw column(s) pk_wait_anomaly overlaps ---
        reduced_cols = [c for c in X_train.columns if c not in REDUNDANT_WITH_PK]
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
            'seed': seed, 'cv_f1_reduced_with_pk': cv_reduced_with_pk,
            'cv_f1_reduced_no_pk': cv_reduced_no_pk,
            'delta': cv_reduced_with_pk - cv_reduced_no_pk,
        })

        corr = correlation_check(train_df)
        corr_rows.append({'seed': seed, **corr.to_dict()})

        # --- (c) permutation importance of pk_wait_anomaly, full vs reduced access ---
        full_imp, reduced_imp = pk_permutation_importance(X_train, y_train, seed)
        importance_rows.append({
            'seed': seed,
            'pk_importance_full_access': full_imp.get('pk_wait_anomaly', np.nan),
            'pk_importance_reduced_access': reduced_imp.get('pk_wait_anomaly', np.nan),
            'pk_rank_full_access': int(pd.Series(full_imp).rank(ascending=False)['pk_wait_anomaly']),
            'pk_rank_reduced_access': int(pd.Series(reduced_imp).rank(ascending=False)['pk_wait_anomaly']),
            'n_features_full': len(full_imp), 'n_features_reduced': len(reduced_imp),
        })

        print(f"seed={seed}  MLP: full={cv_full:.3f} raw={cv_raw:.3f} no_pk={cv_no_pk:.3f}  |  "
              f"reduced-access RF: with_pk={cv_reduced_with_pk:.3f} no_pk={cv_reduced_no_pk:.3f}  |  "
              f"pk perm-importance: full={full_imp.get('pk_wait_anomaly', float('nan')):.4f} "
              f"(rank {importance_rows[-1]['pk_rank_full_access']}/{importance_rows[-1]['n_features_full']}) "
              f"reduced={reduced_imp.get('pk_wait_anomaly', float('nan')):.4f} "
              f"(rank {importance_rows[-1]['pk_rank_reduced_access']}/{importance_rows[-1]['n_features_reduced']})")

    return (
        pd.DataFrame(mlp_rows), pd.DataFrame(access_rows),
        pd.DataFrame(corr_rows), pd.DataFrame(importance_rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    mlp_df, access_df, corr_df, importance_df = run_experiment(args.seeds)
    mlp_df.to_csv(str(OUTPUT_DIR / "mlp_model_family_ablation.csv"), index=False)
    access_df.to_csv(str(OUTPUT_DIR / "feature_access_ablation.csv"), index=False)
    corr_df.to_csv(str(OUTPUT_DIR / "pk_wait_anomaly_correlations.csv"), index=False)
    importance_df.to_csv(str(OUTPUT_DIR / "pk_wait_anomaly_permutation_importance.csv"), index=False)

    print("\n=== (a) Model-family check: MLP, full vs raw-only ===")
    delta_mlp = mlp_df['delta_full_vs_raw'].to_numpy()
    t1, p1 = sstats.ttest_1samp(delta_mlp, 0.0)
    d1 = cohens_d_one_sample(delta_mlp)
    print(f"mean delta={delta_mlp.mean():+.4f}, std={delta_mlp.std():.4f}, "
          f"t={t1:.3f}, p={p1:.4f}, Cohen's d={d1:.3f}")
    print(mlp_df)

    print("\n=== (b) Feature-access check: RF without waiting_time_ms/wait_to_burst_ratio, with vs without pk_wait_anomaly ===")
    delta_access = access_df['delta'].to_numpy()
    t2, p2 = sstats.ttest_1samp(delta_access, 0.0)
    d2 = cohens_d_one_sample(delta_access)
    print(f"mean delta={delta_access.mean():+.4f}, std={delta_access.std():.4f}, "
          f"t={t2:.3f}, p={p2:.4f}, Cohen's d={d2:.3f}")
    print(access_df)

    print("\n=== Correlation check: pk_wait_anomaly vs. the raw features it should be redundant with ===")
    print(corr_df.set_index('seed').round(3))

    print("\n=== Permutation importance of pk_wait_anomaly (mean F1 drop when shuffled), full vs reduced access ===")
    print(importance_df.set_index('seed').round(4))
    print(f"Mean importance, full access: {importance_df['pk_importance_full_access'].mean():.4f} "
          f"(mean rank {importance_df['pk_rank_full_access'].mean():.1f} of "
          f"{importance_df['n_features_full'].iloc[0]} features)")
    print(f"Mean importance, reduced access: {importance_df['pk_importance_reduced_access'].mean():.4f} "
          f"(mean rank {importance_df['pk_rank_reduced_access'].mean():.1f} of "
          f"{importance_df['n_features_reduced'].iloc[0]} features)")

    print(
        f"\nEffect size reference (Cohen, 1988): |d| ~0.2 small, ~0.5 medium, "
        f"~0.8 large. n={len(args.seeds)} seeds per test here; report alongside, "
        f"not instead of, the raw per-seed numbers above."
    )


if __name__ == "__main__":
    main()
