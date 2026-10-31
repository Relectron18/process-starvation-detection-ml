"""
Aggregates per-trial real-hardware validation across all collected trials
(both scenarios), and runs the ground-truth threshold-sensitivity check
reported in Section 5.5 of the paper.

CONFIDENCE-INTERVAL ADDITION (see repro_artifacts CHANGES.md): each trial
has only ~24-26 true-starved windows out of 96, so a per-trial
precision/recall of e.g. 1.000 can rest on a denominator as small as ~20-25
-- and the cross-trial "mean +/- std" over just 3-5 trials understates how
much uncertainty that implies (a std of 0.000 across 3 trials that each
happen to hit 1.000 looks more solid than it is). This adds a per-trial
Wilson score interval for precision and recall, which accounts for the
actual denominator, alongside the existing point estimates.
"""
import glob
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

from config import DATA_DIR, MODELS_DIR
from apply_model_to_real_data import load_artifacts
from real_data_features import score_raw_csv


def wilson_ci(successes, n, alpha=0.05):
    """Wilson score interval for a binomial proportion -- more reliable
    than a normal approximation when n or the proportion is small (both
    apply here: n~20-25 predicted/true positives per trial, precision
    often exactly 1.0)."""
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return (float(lo), float(hi))

_ARTIFACTS = None  # lazily populated by _get_artifacts(), not at import time


def _get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts(str(MODELS_DIR))
    return _ARTIFACTS


def score_file(path):
    model, scaler, feature_cols, encoders, queueing_stats, threshold = _get_artifacts()
    return score_raw_csv(path, model, scaler, encoders, feature_cols, queueing_stats, threshold)


def metrics_for(df, label_col="is_starved"):
    y_true, y_pred, proba = df[label_col], df["model_pred"], df["model_proba"]
    n_flagged = int(y_pred.sum())        # precision denominator
    n_true_pos = int(y_true.sum())        # recall denominator
    n_correct_flagged = int(((y_pred == 1) & (y_true == 1)).sum())
    n_correct_recalled = n_correct_flagged  # same numerator, different denominator

    prec_lo, prec_hi = wilson_ci(n_correct_flagged, n_flagged)
    rec_lo, rec_hi = wilson_ci(n_correct_recalled, n_true_pos)

    out = {
        "n": len(df), "n_starved": n_true_pos,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "precision_n": n_flagged,
        "precision_ci_lo": prec_lo, "precision_ci_hi": prec_hi,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "recall_n": n_true_pos,
        "recall_ci_lo": rec_lo, "recall_ci_hi": rec_hi,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def recompute_label(df, cpu_frac_thresh, wait_frac_thresh):
    is_cpu_bound = df["process_type"] == "cpu_bound"
    starved = is_cpu_bound & (df["burst_time_ms"] < cpu_frac_thresh * df["window_ms"]) & \
              (df["waiting_time_ms"] > wait_frac_thresh * df["window_ms"])
    return starved.astype(int)


def main():
    print("=" * 70)
    print("PER-SCENARIO, PER-TRIAL METRICS (real hardware, repeated trials)")
    print("=" * 70)
    scenario_rows = []
    for scenario in ["rt", "nice"]:
        files = sorted(glob.glob(str(DATA_DIR / f"real_data_{scenario}_trial*.csv")))
        for f in files:
            trial = re.search(r"trial(\d+)", f).group(1)
            df = score_file(f)
            m = metrics_for(df)
            m.update({"scenario": scenario, "trial": trial})
            scenario_rows.append(m)
            print(f"[{scenario} trial {trial}] n={m['n']} starved={m['n_starved']} "
                  f"acc={m['accuracy']:.3f} "
                  f"prec={m['precision']:.3f} (95% CI [{m['precision_ci_lo']:.3f}, {m['precision_ci_hi']:.3f}], n={m['precision_n']}) "
                  f"rec={m['recall']:.3f} (95% CI [{m['recall_ci_lo']:.3f}, {m['recall_ci_hi']:.3f}], n={m['recall_n']}) "
                  f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")

    results = pd.DataFrame(scenario_rows)
    results.to_csv(str(DATA_DIR / "real_world_multitrial_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE ACROSS TRIALS (mean +/- std)")
    print("=" * 70)
    agg = results.groupby("scenario")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    print(agg.round(4))

    print("\n" + "=" * 70)
    print("GROUND-TRUTH THRESHOLD SENSITIVITY (ALL 8 trials, varying the")
    print("operational starvation-label definition)")
    print("=" * 70)
    print("(Extended from a single representative trial in an earlier draft to")
    print(" all 8 trials/both scenarios -- see CHANGES.md.)")
    all_files = sorted(glob.glob(str(DATA_DIR / "real_data_rt_trial*.csv"))) + \
                sorted(glob.glob(str(DATA_DIR / "real_data_nice_trial*.csv")))
    sensitivity_rows = []
    for f in all_files:
        m2 = re.search(r"real_data_(rt|nice)_trial(\d+)", f)
        scenario, trial = m2.group(1), m2.group(2)
        base_df = score_file(f)
        for cpu_t, wait_t in [(0.05, 0.5), (0.10, 0.5), (0.05, 0.4), (0.10, 0.4), (0.15, 0.3), (0.05, 0.6)]:
            lbl = recompute_label(base_df, cpu_t, wait_t)
            tmp = base_df.copy()
            tmp["is_starved"] = lbl
            m = metrics_for(tmp)
            m.update({"scenario": scenario, "trial": trial, "cpu_frac_thresh": cpu_t, "wait_frac_thresh": wait_t})
            sensitivity_rows.append(m)

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(str(DATA_DIR / "threshold_sensitivity.csv"), index=False)

    sens_agg = sens_df.groupby(["cpu_frac_thresh", "wait_frac_thresh"])[
        ["n_starved", "accuracy", "precision", "recall", "f1"]
    ].agg(["mean", "std"])
    print(sens_agg.round(4))
    print(f"\nF1 across all {len(all_files)} trials x 6 threshold definitions "
          f"({len(sens_df)} combinations): "
          f"min={sens_df['f1'].min():.3f}, max={sens_df['f1'].max():.3f}, "
          f"mean={sens_df['f1'].mean():.3f}, std={sens_df['f1'].std():.3f}")
    print(f"Precision: min={sens_df['precision'].min():.3f}, max={sens_df['precision'].max():.3f}")
    print(f"n_starved (out of 96): min={sens_df['n_starved'].min()}, max={sens_df['n_starved'].max()}")


if __name__ == "__main__":
    main()
