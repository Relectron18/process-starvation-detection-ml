"""
Statistical rigor pass for paper-worthiness: the notebook's model comparison
and feature ablation were each run once, on one synthetic draw (seed=42).
A single run can't distinguish a real effect from an artifact of that one
random dataset. This repeats the full pipeline -- fresh synthetic data,
fresh train/test split, fresh CV folds -- across 5 independent seeds, and
reports mean +/- std plus a paired significance test (with effect size)
between the top two models and between HRRN and the best model.

LEAKAGE + SELECTION-BIAS FIX (see repro_artifacts CHANGES.md):
  1. Feature-engineering statistics (queueing-theory group stats, wait-time
     median) and the LabelEncoders/StandardScaler are now fit on the
     training split only (split_raw + fit_split_stats/apply_split_stats +
     fit_encoders_scaler), never on the full pre-split dataset.
  2. The "best untuned ML" model compared against HRRN (Sec. 5.2) is no
     longer chosen by taking whichever of the 5 models happens to score
     highest F1 on the TEST set (a "winner's curse" that inflates the
     ML-vs-HRRN margin) -- it's now the model with the highest mean
     cross-validated F1 on the TRAINING data, selected before the test set
     is touched, exactly as a real deployment decision would be made.
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from config import OUTPUT_DIR
from pipeline_core import (
    split_raw, fit_split_stats, apply_split_stats,
    fit_encoders_scaler, transform_with, get_models, hrrn_risk_score,
)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference / std of the
    differences (ddof=1, matching the sample std used by ttest_rel)."""
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def cohens_d_one_sample(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def run_experiment(seeds):
    """Runs the full multi-seed pipeline. Pure(ish) function - takes seeds
    in, returns result DataFrames out, no argparse/print-as-side-effect
    coupling, so it's usable from tests or other scripts without spawning
    a subprocess."""
    all_rows = []
    paired_rf_xgb = []   # per-seed CV-mean F1 for RF vs XGB, for a paired test
    paired_hrrn_ml = []  # per-seed test-set F1 for CV-selected-best-ML vs HRRN
    ablation_rows = []   # per-seed feature ablation (raw vs full) for RF

    for seed in seeds:
        t0 = time.time()
        train_df, test_df = split_raw(seed)
        split_stats = fit_split_stats(train_df)
        train_df = apply_split_stats(train_df, split_stats)
        test_df = apply_split_stats(test_df, split_stats)

        X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_df)
        X_test, y_test = transform_with(test_df, encoders, scaler, feature_cols)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        models = get_models(seed)

        seed_f1, seed_cv_f1 = {}, {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
            row = {
                'seed': seed, 'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
            }
            all_rows.append(row)
            seed_f1[name] = row['f1']
            seed_cv_f1[name] = row['cv_f1_mean']

        paired_rf_xgb.append((seed_f1['Random Forest'], seed_f1['XGBoost']))

        # Select the "best" model by TRAINING-only CV F1 (never test), so
        # the model that gets compared to HRRN below is chosen the way a
        # real deployment decision would be -- not by peeking at test F1.
        cv_selected_name = max(seed_cv_f1, key=seed_cv_f1.get)
        cv_selected_test_f1 = seed_f1[cv_selected_name]

        hrrn_test_score = hrrn_risk_score(
            test_df['waiting_time_ms'], test_df['burst_time_ms']
        )
        hrrn_pred = (hrrn_test_score >= 0.5).astype(int)
        hrrn_f1 = f1_score(y_test, hrrn_pred)
        paired_hrrn_ml.append((cv_selected_test_f1, hrrn_f1))

        engineered_cols = [c for c in ['wait_to_burst_ratio', 'preemption_rate', 'cpu_starvation_gap',
                                        'low_priority_high_wait', 'pk_wait_anomaly'] if c in X_train.columns]
        raw_cols = [c for c in X_train.columns if c not in engineered_cols]
        rf = get_models(seed)['Random Forest']
        cv_full = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
        rf2 = get_models(seed)['Random Forest']
        cv_raw = cross_val_score(rf2, X_train[raw_cols], y_train, cv=cv, scoring='f1', n_jobs=-1).mean()
        ablation_rows.append({'seed': seed, 'cv_f1_full': cv_full, 'cv_f1_raw_only': cv_raw,
                               'delta': cv_full - cv_raw})

        print(f"seed={seed} done in {time.time()-t0:.1f}s  "
              f"(CV-selected model: {cv_selected_name}, test F1={cv_selected_test_f1:.3f}, "
              f"HRRN={hrrn_f1:.3f})")

    return (
        pd.DataFrame(all_rows),
        pd.DataFrame(ablation_rows),
        np.array(paired_rf_xgb),
        np.array(paired_hrrn_ml),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                         default=[42, 7, 123, 2024, 99, 555, 777, 888, 314, 271, 161, 202])
    args = parser.parse_args()

    results, ablation_df, paired_rf_xgb, paired_hrrn_ml = run_experiment(args.seeds)
    results.to_csv(str(OUTPUT_DIR / "multiseed_results.csv"), index=False)
    ablation_df.to_csv(str(OUTPUT_DIR / "multiseed_ablation.csv"), index=False)
    pd.DataFrame({
        "seed": args.seeds,
        "rf_f1": paired_rf_xgb[:, 0], "xgb_f1": paired_rf_xgb[:, 1],
        "cv_selected_ml_f1": paired_hrrn_ml[:, 0], "hrrn_f1": paired_hrrn_ml[:, 1],
    }).to_csv(str(OUTPUT_DIR / "multiseed_paired_comparisons.csv"), index=False)

    summary = results.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_f1_mean']].agg(
        ['mean', 'std']
    ).round(4)
    print("\n=== Per-model mean +/- std across independent synthetic-data seeds ===")
    print(summary)

    rf_scores, xgb_scores = paired_rf_xgb[:, 0], paired_rf_xgb[:, 1]
    t_stat, p_val = sstats.ttest_rel(rf_scores, xgb_scores)
    d_rf_xgb = cohens_d_paired(rf_scores, xgb_scores)
    print(f"\nPaired t-test, Random Forest vs XGBoost F1 across seeds: "
          f"RF mean={rf_scores.mean():.4f}, XGB mean={xgb_scores.mean():.4f}, "
          f"t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d_rf_xgb:.3f}")

    best_scores, hrrn_scores = paired_hrrn_ml[:, 0], paired_hrrn_ml[:, 1]
    t_stat2, p_val2 = sstats.ttest_rel(best_scores, hrrn_scores)
    d_ml_hrrn = cohens_d_paired(best_scores, hrrn_scores)
    print(f"Paired t-test, CV-selected ML vs HRRN F1 across seeds: "
          f"ML mean={best_scores.mean():.4f}, HRRN mean={hrrn_scores.mean():.4f}, "
          f"t={t_stat2:.3f}, p={p_val2:.4g}, Cohen's d={d_ml_hrrn:.3f}")

    delta = ablation_df['delta'].to_numpy()
    t_stat3, p_val3 = sstats.ttest_1samp(delta, 0.0)
    d_ablation = cohens_d_one_sample(delta)
    print(f"\nFeature ablation (RF, full vs raw-only) across seeds: "
          f"mean delta={delta.mean():+.4f}, std={delta.std():.4f}, "
          f"one-sample t-test vs 0: t={t_stat3:.3f}, p={p_val3:.4f}, Cohen's d={d_ablation:.3f}")
    print(ablation_df)

    print(
        "\nEffect size reference (Cohen, 1988): |d| ~0.2 small, ~0.5 medium, "
        "~0.8 large. Note n=5 seeds per test here means these d estimates "
        "themselves carry wide uncertainty - report alongside, not instead of, "
        "the raw per-seed numbers above."
    )


if __name__ == "__main__":
    main()
