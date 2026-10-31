"""
Item 5 follow-up: Table 2 in the paper validates real hardware against a
SINGLE trained model (seed=42). That conflates "the trained artifact
generalizes to real hardware" with "seed 42's particular model happens to
generalize to real hardware" -- two different claims. This trains two
more models (different seeds, same synthetic pipeline/hyperparameter
search) into separate model directories, applies each -- independently,
with no retraining -- to all 8 real-hardware trials, and reports the
per-seed real-hardware aggregate metrics side by side with seed 42's.

This does NOT change Table 2, which stays as the seed-42 result (the
model whose training is described in detail in Section 4/5.4). It adds a
robustness check: if the other seeds' real-hardware numbers are in the
same ballpark, that's evidence Table 2 isn't a lucky seed; if they vary
wildly, that's a limitation worth stating explicitly.

Usage: run train_model.py for each extra seed into its own directory
first (see run_all.sh), then run this script.
"""
import argparse
import glob
import re

import pandas as pd

from config import DATA_DIR, MODELS_DIR
from apply_model_to_real_data import load_artifacts
from real_data_features import score_raw_csv
from aggregate_real_world import metrics_for


def score_with_artifacts(path, artifacts):
    model, scaler, feature_cols, encoders, queueing_stats, threshold = artifacts
    return score_raw_csv(path, model, scaler, encoders, feature_cols, queueing_stats, threshold)


def run_for_seed_dir(models_dir, label):
    artifacts = load_artifacts(models_dir)
    rows = []
    for scenario in ["rt", "nice"]:
        files = sorted(glob.glob(str(DATA_DIR / f"real_data_{scenario}_trial*.csv")))
        for f in files:
            trial = re.search(r"trial(\d+)", f).group(1)
            df = score_with_artifacts(f, artifacts)
            m = metrics_for(df)
            m.update({"model": label, "scenario": scenario, "trial": trial})
            rows.append(m)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs="+", required=True,
                         help="pairs of label=path, e.g. seed42=models seed7=models_seed7")
    args = parser.parse_args()

    all_rows = []
    for spec in args.model_dirs:
        label, path = spec.split("=", 1)
        print(f"Scoring all 8 real-hardware trials with {label} ({path}) ...")
        df = run_for_seed_dir(path, label)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(str(DATA_DIR / "multiseed_real_world_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("PER-MODEL-SEED, PER-SCENARIO AGGREGATE (mean +/- std across trials)")
    print("=" * 70)
    agg = combined.groupby(["model", "scenario"])[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(
        ["mean", "std"]
    )
    print(agg.round(4))

    print("\n" + "=" * 70)
    print("CROSS-SEED SPREAD (std of each model-seed's mean F1, per scenario)")
    print("=" * 70)
    per_seed_mean_f1 = combined.groupby(["model", "scenario"])["f1"].mean().reset_index()
    spread = per_seed_mean_f1.groupby("scenario")["f1"].agg(["mean", "std", "min", "max"])
    print(spread.round(4))
    print(
        "\nA small std here (relative to the within-seed trial-to-trial std "
        "already reported in Table 2) means the real-hardware validation "
        "result is not an artifact of which training seed happened to be "
        "used; a large one means Table 2 should be read as one seed's "
        "result, not a generic property of 'the' trained artifact."
    )


if __name__ == "__main__":
    main()
