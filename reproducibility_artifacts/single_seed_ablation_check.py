"""
Item 4 follow-up: Section 5.3(i) and 5.3(ii) in the paper (Random Forest
and Logistic Regression single-seed ablations of the engineered features)
originally came from the leaky notebook pipeline (build_dataset+preprocess,
stats fit on the full dataset before the split -- see CHANGES.md). They
were retained in the revised paper with a caveat that they "predate the
leakage fix... retained for continuity." This script reruns both checks,
seed=42, on the corrected leak-free split_raw pipeline, so that caveat can
be removed and Section 5.3 can cite three checks that are all leak-free.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline_core import (
    split_raw, fit_split_stats, apply_split_stats, fit_encoders_scaler, get_models,
)

ENGINEERED_COLS = [
    'wait_to_burst_ratio', 'preemption_rate', 'cpu_starvation_gap',
    'low_priority_high_wait', 'pk_wait_anomaly',
]


def main(seed=42):
    train_df, test_df = split_raw(seed)
    split_stats = fit_split_stats(train_df)
    train_df = apply_split_stats(train_df, split_stats)
    X_train, y_train, encoders, scaler, feature_cols = fit_encoders_scaler(train_df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    raw_cols = [c for c in X_train.columns if c not in ENGINEERED_COLS]

    rf = get_models(seed)['Random Forest']
    cv_full_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    rf2 = get_models(seed)['Random Forest']
    cv_raw_rf = cross_val_score(rf2, X_train[raw_cols], y_train, cv=cv, scoring='f1', n_jobs=-1)
    delta_rf = cv_full_rf.mean() - cv_raw_rf.mean()
    print(f"(i) Random Forest, seed={seed}, leak-free: full={cv_full_rf.mean():.4f} "
          f"raw_only={cv_raw_rf.mean():.4f}  delta={delta_rf:+.4f}")

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    cv_full_lr = cross_val_score(lr, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    lr2 = LogisticRegression(max_iter=1000, class_weight='balanced')
    cv_raw_lr = cross_val_score(lr2, X_train[raw_cols], y_train, cv=cv, scoring='f1', n_jobs=-1)
    delta_lr = cv_full_lr.mean() - cv_raw_lr.mean()
    print(f"(ii) Logistic Regression, seed={seed}, leak-free: full={cv_full_lr.mean():.4f} "
          f"raw_only={cv_raw_lr.mean():.4f}  delta={delta_lr:+.4f}")


if __name__ == "__main__":
    main()
