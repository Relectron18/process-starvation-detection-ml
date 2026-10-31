"""
Real-world validation: take the model, scaler, and encoders trained purely
on the synthetic dataset in the notebook, and apply them -- via the exact
same feature-engineering formulas used in dashboard.py -- to genuine
/proc/[pid]/schedstat data captured from real competing OS processes,
including one real, verifiable SCHED_FIFO starvation episode.

This does not retrain anything. It's a pure generalization test: does a
classifier that has only ever seen simulated numbers produce sane,
useful output on real kernel scheduler accounting data it has never seen
the likes of before?
"""
import argparse
import json

import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import MODELS_DIR, DATA_DIR
from real_data_features import score_raw_csv


def load_artifacts(models_dir):
    model = joblib.load(f"{models_dir}/best_model.pkl")
    scaler = joblib.load(f"{models_dir}/scaler.pkl")
    feature_cols = joblib.load(f"{models_dir}/feature_columns.pkl")
    encoders = {
        "process_type": joblib.load(f"{models_dir}/encoder_process_type.pkl"),
        "process_state": joblib.load(f"{models_dir}/encoder_process_state.pkl"),
    }
    queueing_stats = joblib.load(f"{models_dir}/queueing_stats.pkl")
    with open(f"{models_dir}/run_metadata.json") as f:
        metadata = json.load(f)
    threshold = metadata.get("recommended_threshold", 0.5)
    return model, scaler, feature_cols, encoders, queueing_stats, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_csv", nargs="?", default=str(DATA_DIR / "real_scheduling_data.csv"))
    parser.add_argument("out_csv", nargs="?", default=str(DATA_DIR / "real_scheduling_data_scored.csv"))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    model, scaler, feature_cols, encoders, queueing_stats, threshold = load_artifacts(args.models_dir)
    engineered = score_raw_csv(args.in_csv, model, scaler, encoders, feature_cols, queueing_stats, threshold)

    y_true = engineered["is_starved"]
    y_pred = engineered["model_pred"]
    proba = engineered["model_proba"]

    print("=== Real-world generalization: synthetic-trained model on genuine schedstat data ===")
    print(f"Rows: {len(engineered)}  |  True starved windows: {y_true.sum()}  |  "
          f"Threshold used: {threshold}\n")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_true, proba):.3f}")
    except ValueError:
        print("ROC-AUC  : n/a (only one class present)")
    print()
    print("Confusion matrix [rows=true, cols=pred] (0=not starved, 1=starved):")
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\nPer-process summary:")
    summary = engineered.groupby(["process_label", "phase"]).agg(
        n=("is_starved", "size"),
        true_starved=("is_starved", "sum"),
        model_flagged=("model_pred", "sum"),
        mean_model_proba=("model_proba", "mean"),
        mean_hrrn_score=("hrrn_risk_score", "mean"),
    ).round(3)
    print(summary)

    engineered.to_csv(args.out_csv, index=False)
    print(f"\nSaved scored dataset to {args.out_csv}")


if __name__ == "__main__":
    main()
