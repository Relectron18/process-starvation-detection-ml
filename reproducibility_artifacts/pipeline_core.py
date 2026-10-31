"""Core synthetic-data pipeline, extracted verbatim from the notebook so
multi-seed statistical testing runs the *exact* same methodology, just
repeated across different random draws."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

N_SAMPLES = 12000


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid = np.arange(1000, 1000 + n_samples)
    priority = rng.integers(0, 10, size=n_samples)
    process_type = rng.choice(
        ['cpu_bound', 'io_bound', 'interactive', 'batch'],
        size=n_samples, p=[0.35, 0.3, 0.2, 0.15]
    )
    burst_time = np.where(
        process_type == 'cpu_bound', rng.normal(180, 60, n_samples),
        np.where(process_type == 'io_bound', rng.normal(40, 15, n_samples),
                 np.where(process_type == 'interactive', rng.normal(20, 8, n_samples),
                          rng.normal(250, 80, n_samples)))
    )
    burst_time = np.clip(burst_time, 1, None)
    queue_length_at_arrival = rng.poisson(lam=6, size=n_samples)
    times_preempted = rng.poisson(lam=(priority * 0.6 + queue_length_at_arrival * 0.15))
    aging_applied = rng.random(n_samples) < 0.35
    base_wait = (priority * 25) + (queue_length_at_arrival * 8) + (times_preempted * 30)
    aging_reduction = np.where(aging_applied, rng.uniform(0.3, 0.7, n_samples), 1.0)
    waiting_time = base_wait * aging_reduction + rng.normal(0, 15, n_samples)
    waiting_time = np.clip(waiting_time, 0, None)
    turnaround_time = waiting_time + burst_time
    cpu_usage_percent = np.clip(
        rng.normal(30 - priority * 1.8 - times_preempted * 1.2, 8, n_samples), 0, 100
    )
    memory_usage_percent = np.clip(rng.normal(15, 8, n_samples), 0.1, 90)
    io_wait_time = np.where(
        process_type == 'io_bound', rng.normal(60, 20, n_samples), rng.normal(10, 6, n_samples)
    )
    io_wait_time = np.clip(io_wait_time, 0, None)
    context_switches = rng.poisson(lam=(5 + times_preempted * 2), size=n_samples)
    num_threads = rng.integers(1, 16, size=n_samples)
    nice_value = np.clip(priority + rng.integers(-2, 3, n_samples), -20, 19)
    process_state = rng.choice(
        ['ready', 'waiting', 'running', 'blocked'],
        size=n_samples, p=[0.4, 0.35, 0.15, 0.1]
    )

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    risk_score = (
        0.35 * norm(waiting_time) + 0.20 * norm(priority) + 0.15 * norm(times_preempted) +
        0.15 * (1 - norm(cpu_usage_percent)) + 0.10 * norm(queue_length_at_arrival) +
        0.05 * norm(context_switches)
    )
    noise = rng.normal(0, 0.06, n_samples)
    is_starved = ((risk_score + noise) > 0.55).astype(int)

    return pd.DataFrame({
        'pid': pid, 'process_type': process_type, 'priority': priority, 'nice_value': nice_value,
        'burst_time_ms': burst_time.round(2), 'waiting_time_ms': waiting_time.round(2),
        'turnaround_time_ms': turnaround_time.round(2),
        'queue_length_at_arrival': queue_length_at_arrival, 'times_preempted': times_preempted,
        'cpu_usage_percent': cpu_usage_percent.round(2), 'memory_usage_percent': memory_usage_percent.round(2),
        'io_wait_time_ms': io_wait_time.round(2), 'context_switches': context_switches,
        'num_threads': num_threads, 'process_state': process_state, 'is_starved': is_starved,
    })


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['wait_to_burst_ratio'] = data['waiting_time_ms'] / (data['burst_time_ms'] + 1e-6)
    data['preemption_rate'] = data['times_preempted'] / (data['context_switches'] + 1)
    fair_share = 100 / (data['priority'] + 1)
    data['cpu_starvation_gap'] = (fair_share - data['cpu_usage_percent']).clip(lower=0)
    data['low_priority_high_wait'] = (
        (data['priority'] >= 6) & (data['waiting_time_ms'] > data['waiting_time_ms'].median())
    ).astype(int)
    return data


def add_row_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features only (no cross-row statistics, so this
    is safe to call before a train/test split without leaking test rows
    into anything). Does NOT add low_priority_high_wait or pk_wait_anomaly,
    both of which depend on a threshold/statistic fit across many rows --
    see fit_split_stats()/apply_split_stats() for those, which must be fit
    on the training partition only."""
    data = data.copy()
    data['wait_to_burst_ratio'] = data['waiting_time_ms'] / (data['burst_time_ms'] + 1e-6)
    data['preemption_rate'] = data['times_preempted'] / (data['context_switches'] + 1)
    fair_share = 100 / (data['priority'] + 1)
    data['cpu_starvation_gap'] = (fair_share - data['cpu_usage_percent']).clip(lower=0)
    return data


def fit_queueing_theory_stats(data: pd.DataFrame, group_col: str = 'process_type') -> dict:
    stats = {}
    for g, gdf in data.groupby(group_col):
        S = gdf['burst_time_ms'].to_numpy()
        E_S, E_S2 = S.mean(), (S ** 2).mean()
        L, W = gdf['queue_length_at_arrival'].mean(), gdf['turnaround_time_ms'].mean()
        lam = L / (W + 1e-6)
        rho = min(lam * E_S, 0.98)
        stats[g] = {'lambda': lam, 'E_S': E_S, 'E_S2': E_S2, 'rho': rho}
    return stats


def pk_expected_wait(row, stats, fallback):
    s = stats.get(row['process_type'], fallback)
    lam, E_S2, rho = s['lambda'], s['E_S2'], s['rho']
    return (lam * E_S2) / (2 * (1 - rho) + 1e-6)


def add_queueing_feature(data, stats):
    data = data.copy()
    fallback = {
        'lambda': np.mean([v['lambda'] for v in stats.values()]),
        'E_S2': np.mean([v['E_S2'] for v in stats.values()]),
        'rho': np.mean([v['rho'] for v in stats.values()]),
    }
    expected_wait = data.apply(lambda r: pk_expected_wait(r, stats, fallback), axis=1)
    data['pk_wait_anomaly'] = data['waiting_time_ms'] / (expected_wait + 1e-6)
    return data


CATEGORICAL_COLS = ['process_type', 'process_state']
TARGET_COL = 'is_starved'
DROP_COLS = ['pid']


def preprocess(df):
    """DEPRECATED for train/test evaluation: fits the LabelEncoders and
    StandardScaler on whatever is passed in, so calling this on a full
    dataset *before* splitting leaks test-row statistics (feature means/
    stds, category sets) into training. Kept only for the smoke tests in
    tests/test_pipeline_core.py, which don't do a train/test split at all.
    Evaluation code should use fit_encoders_scaler()/transform_with() below,
    called after splitting."""
    data = df.copy()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    y = data[TARGET_COL]
    X = data.drop(columns=[TARGET_COL])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y


def fit_split_stats(train_df: pd.DataFrame) -> dict:
    """Fit every cross-row statistic used by feature engineering
    (queueing-theory group stats for pk_wait_anomaly, and the waiting-time
    median for low_priority_high_wait) on the TRAINING partition only."""
    return {
        'queueing': fit_queueing_theory_stats(train_df),
        'wait_median': float(train_df['waiting_time_ms'].median()),
    }


def apply_split_stats(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply stats fit on the training partition (via fit_split_stats) to
    any dataframe (train or test) -- so test rows never influence the
    threshold/group-stats used to compute their own features."""
    data = data.copy()
    data['low_priority_high_wait'] = (
        (data['priority'] >= 6) & (data['waiting_time_ms'] > stats['wait_median'])
    ).astype(int)
    data = add_queueing_feature(data, stats['queueing'])
    return data


def fit_encoders_scaler(train_df: pd.DataFrame):
    """Fit LabelEncoders + StandardScaler on the training partition only.
    Returns (X_train, y_train, encoders, scaler, feature_cols)."""
    data = train_df.copy()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    y = data[TARGET_COL]
    X = data.drop(columns=[TARGET_COL])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y, encoders, scaler, list(X.columns)


def transform_with(df: pd.DataFrame, encoders, scaler, feature_cols):
    """Apply already-fit encoders/scaler (from fit_encoders_scaler on the
    training partition) to another dataframe (test/val), never refitting."""
    data = df.copy()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])
    for col, le in encoders.items():
        data[col] = data[col].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
        data[col] = le.transform(data[col])
    y = data[TARGET_COL]
    X = data[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X_scaled, y


def split_raw(seed, test_size=0.2, n_samples=N_SAMPLES):
    """Leak-free entrypoint for evaluation scripts: generate synthetic data,
    add only the row-wise engineered features, then split -- BEFORE any
    cross-row statistic (queueing-theory stats, wait-time median, label
    encoders, feature scaler) is fit. Callers must fit those on the
    returned train_df only and apply them to test_df with apply_split_stats
    / fit_encoders_scaler / transform_with."""
    df = generate_dataset(n_samples=n_samples, seed=seed)
    df = add_row_engineered_features(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET_COL]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_models(seed):
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight='balanced', random_state=seed, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
        ),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            eval_metric='logloss', random_state=seed, n_jobs=-1
        ),
    }


def build_dataset(seed):
    """DEPRECATED for train/test evaluation: fits queueing-theory stats and
    the wait-time median on the FULL dataset, so a downstream split on this
    output leaks test rows into those statistics. Safe only for uses that
    never split the result (e.g. the determinism/shape smoke tests in
    tests/test_pipeline_core.py). Evaluation code should use split_raw()
    plus fit_split_stats()/apply_split_stats() instead."""
    df = generate_dataset(seed=seed)
    df = add_engineered_features(df)
    stats = fit_queueing_theory_stats(df)
    df = add_queueing_feature(df, stats)
    return df


def hrrn_risk_score(waiting_time_ms, burst_time_ms, center=2.0, scale=0.75):
    r = (waiting_time_ms + burst_time_ms) / (burst_time_ms + 1e-6)
    return 1 / (1 + np.exp(-(r - center) / scale))
