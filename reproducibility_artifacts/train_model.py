"""
Headless equivalent of the notebook's training path (cells 4-51): generate
the synthetic dataset, engineer features, compare 5 untuned models,
RandomizedSearchCV-tune the top 2 tree models, pick the best by F1,
tune the decision threshold, and save everything apply_model_to_real_data.py
and aggregate_real_world.py expect in models/.

Why this exists: without it, the "reproducibility artifacts" package
cannot actually reproduce the real-hardware validation numbers (5.4-5.5)
standalone - those scripts require a trained model, and previously the
only way to produce one was to run the full notebook in Jupyter. This
script is the single-seed (seed=42) canonical run; multiseed_stability.py
covers the repeated-seed robustness claims (5.1-5.3) separately and does
not depend on this script's output.

LEAKAGE FIX (see repro_artifacts CHANGES.md): the previous version fit
feature-engineering statistics (queueing-theory group stats, wait-time
median) AND the StandardScaler/LabelEncoders on the full dataset before
splitting, and tuned the decision threshold by maximizing F1 directly on
the held-out test set. All three leaked test-set information into
choices that were then evaluated on that same test set. This version:
  1. Splits first (split_raw), then fits all cross-row statistics and the
     scaler/encoders on the training partition only (fit_split_stats /
     fit_encoders_scaler), applying them to test without refitting.
  2. Carves an internal validation split out of the training partition to
     select the best model, tune hyperparameters, and tune the decision
     threshold -- the test set is touched exactly once, at the very end,
     to report final metrics.
  3. Refits the chosen model (with its chosen hyperparameters) on the
     full training partition (train+val) before that final test
     evaluation, so no training data is wasted on the deployed artifact.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from config import MODELS_DIR
from pipeline_core import (
    split_raw, fit_split_stats, apply_split_stats,
    fit_encoders_scaler, transform_with, get_models, TARGET_COL,
)

TUNE_PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [8, 10, 12, 16, None],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}


def evaluate(model, X_tr, X_ev, y_tr, y_ev, name, cv=None, fit=True):
    t0 = time.time()
    if fit:
        model.fit(X_tr, y_tr)
    fit_seconds = time.time() - t0
    y_pred = model.predict(X_ev)
    y_proba = model.predict_proba(X_ev)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_ev, y_pred),
        "precision": precision_score(y_ev, y_pred),
        "recall": recall_score(y_ev, y_pred),
        "f1": f1_score(y_ev, y_pred),
        "roc_auc": roc_auc_score(y_ev, y_proba),
        "pr_auc": average_precision_score(y_ev, y_proba),
        "fit_seconds": round(fit_seconds, 2),
    }
    if cv is not None:
        metrics["cv_f1_mean"] = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1).mean()
    return model, metrics, y_pred, y_proba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per tuned model")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    seed = args.seed

    # 1. Split FIRST -- everything downstream is fit on train_full only.
    train_full_df, test_df = split_raw(seed, test_size=0.2)

    # 2. Carve a validation split out of training, for model selection,
    #    hyperparameter tuning, and threshold tuning. Test stays untouched.
    train_df, val_df = train_test_split(
        train_full_df, test_size=0.25, random_state=seed, stratify=train_full_df[TARGET_COL]
    )  # 0.25 of the 80% training partition = 20% of the total, so the
       # overall split is 60% train / 20% val / 20% test.

    # 3. Fit cross-row feature stats (queueing theory, wait-time median) on
    #    the innermost training fold only; apply unchanged to val and test.
    split_stats = fit_split_stats(train_df)
    train_df = apply_split_stats(train_df, split_stats)
    val_df = apply_split_stats(val_df, split_stats)
    test_df = apply_split_stats(test_df, split_stats)

    # 4. Fit encoders/scaler on train only; transform val/test without refitting.
    X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_df)
    X_val, y_val = transform_with(val_df, encoders, scaler, feature_cols)
    X_test, y_test = transform_with(test_df, encoders, scaler, feature_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    models = get_models(seed)
    val_results, rows = {}, []
    for name, model in models.items():
        trained, metrics, y_pred, y_proba = evaluate(model, X_train, X_val, y_train, y_val, name, cv=cv)
        val_results[name] = (trained, metrics)
        rows.append(metrics)
        print(f"[{name}] (val) F1={metrics['f1']:.3f}  Acc={metrics['accuracy']:.3f}  "
              f"AUC={metrics['roc_auc']:.3f}  ({metrics['fit_seconds']}s)")

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    tunable = {"Random Forest", "XGBoost"}
    top_two = comparison["model"].head(2).tolist()
    to_tune = [m for m in top_two if m in tunable] or [m for m in tunable if m in comparison["model"].values]

    for name in to_tune:
        base_model = get_models(seed)[name]
        search = RandomizedSearchCV(
            base_model, TUNE_PARAM_DISTRIBUTIONS[name], n_iter=args.n_iter, scoring="f1",
            cv=cv, random_state=seed, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        tuned_model, tuned_metrics, tuned_pred, tuned_proba = evaluate(
            search.best_estimator_, X_train, X_val, y_train, y_val, f"{name} (tuned)", fit=False
        )
        val_results[f"{name} (tuned)"] = (tuned_model, tuned_metrics)
        rows.append(tuned_metrics)
        print(f"[{name} (tuned)] (val) F1={tuned_metrics['f1']:.3f}  best params={search.best_params_}")

    # 5. Pick the best model by VALIDATION F1 (never test).
    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]
    print(f"\nBest model by validation F1: {best_name}  (val F1={comparison.iloc[0]['f1']:.3f})")

    # 6. Tune the decision threshold on VALIDATION predictions (never test).
    best_model_val, best_metrics_val = val_results[best_name]
    y_val_proba = best_model_val.predict_proba(X_val)[:, 1] if hasattr(best_model_val, "predict_proba") else best_model_val.predict(X_val)
    thresholds = np.linspace(0.05, 0.95, 37)
    f1s = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    print(f"Tuned decision threshold (on validation): {best_threshold:.2f} "
          f"(default 0.50 gave val F1={f1s[len(f1s)//2]:.3f})")

    # 7. Refit the chosen model+hyperparameters on train+val combined (no
    #    test rows), so the deployed artifact isn't left with 20% less
    #    training data than the original (leaky) version used.
    if best_name.endswith("(tuned)"):
        base_name = best_name.replace(" (tuned)", "")
        search_final = RandomizedSearchCV(
            get_models(seed)[base_name], TUNE_PARAM_DISTRIBUTIONS[base_name], n_iter=args.n_iter,
            scoring="f1", cv=cv, random_state=seed, n_jobs=-1, refit=True,
        )
        search_final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        final_model = search_final.best_estimator_
    else:
        final_model = get_models(seed)[best_name]
        final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # 8. Touch the test set exactly once, with the model/threshold chosen
    #    entirely from train+val, to report the final held-out metrics.
    y_test_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else final_model.predict(X_test)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    final_metrics = {
        "model": best_name,
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "pr_auc": average_precision_score(y_test, y_test_proba),
    }
    print(f"\nFinal held-out TEST metrics (model+threshold chosen from train+val only):")
    for k, v in final_metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    # 9. Refit split_stats/encoders/scaler on the FULL training partition
    #    (train+val = 80% of data) for the deployed artifact, consistent
    #    with the model above -- maximizes data used for real-hardware
    #    deployment while still never touching the test set.
    deploy_split_stats = fit_split_stats(train_full_df)
    train_full_engineered = apply_split_stats(train_full_df, deploy_split_stats)
    _, _, deploy_encoders, deploy_scaler, deploy_feature_cols = fit_encoders_scaler(train_full_engineered)

    models_dir = args.models_dir
    joblib.dump(final_model, f"{models_dir}/best_model.pkl")
    joblib.dump(deploy_scaler, f"{models_dir}/scaler.pkl")
    joblib.dump(deploy_feature_cols, f"{models_dir}/feature_columns.pkl")
    for col, enc in deploy_encoders.items():
        joblib.dump(enc, f"{models_dir}/encoder_{col}.pkl")
    joblib.dump(deploy_split_stats["queueing"], f"{models_dir}/queueing_stats.pkl")
    with open(f"{models_dir}/best_model_name.txt", "w") as f:
        f.write(best_name)

    run_metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "seed": seed,
        "best_model": best_name,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in final_metrics.items() if k != "model"},
        "recommended_threshold": round(best_threshold, 2),
        "feature_columns": deploy_feature_cols,
        "n_train_samples": int(len(train_df)),
        "n_val_samples": int(len(val_df)),
        "n_test_samples": int(len(test_df)),
        "model_selection_and_threshold_tuned_on": "validation split only (test set touched once, at the end)",
    }
    with open(f"{models_dir}/run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"\nSaved model artifacts to {models_dir}/")


if __name__ == "__main__":
    main()
