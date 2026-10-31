"""
Shared logic for turning a raw real-hardware schedstat CSV (from
collect_real_data.py) into the exact feature set the synthetic-trained
model expects.

This used to be copy-pasted independently inside apply_model_to_real_data.py
and aggregate_real_world.py. Two copies of the same mapping is a
correctness risk in a reproducibility artifact: a fix applied to one path
silently doesn't apply to the other, and the paper's real-hardware numbers
(Table 2, Section 5.4-5.5) depend on both paths agreeing.
"""
import numpy as np
import pandas as pd

from pipeline_core import hrrn_risk_score, pk_expected_wait

STATE_MAP = {"R": "running", "S": "waiting", "D": "blocked", "T": "blocked", "Z": "blocked"}

# Fallback only used if a row is missing mem_total_kb (e.g. older CSVs
# collected before this column existed). New collections always carry the
# real /proc/meminfo value with them - see collect_real_data.py.
_FALLBACK_MEM_TOTAL_KB = 4_000_000


REQUIRED_RAW_COLUMNS = [
    "rss_kb", "waiting_time_ms", "burst_time_ms", "window_ms", "nice",
    "process_label", "state_char", "nonvoluntary_ctxt_switches",
    "voluntary_ctxt_switches", "process_type", "is_starved",
]


def validate_raw_columns(raw: pd.DataFrame, source: str = "<input>") -> None:
    """Fail fast with a clear message instead of a KeyError deep inside
    pandas/sklearn when a collected CSV is missing an expected column
    (e.g. from a hand-edited file, an older collect_real_data.py version,
    or a different data source entirely)."""
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required column(s) {missing}. "
            f"Expected at least: {REQUIRED_RAW_COLUMNS}. "
            "This CSV doesn't match the schema collect_real_data.py produces - "
            "check it wasn't truncated, hand-edited, or from a different collector."
        )
    if len(raw) == 0:
        raise ValueError(f"{source}: file has a header but zero data rows.")


def raw_to_model_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw collect_real_data.py columns onto the training feature schema."""
    validate_raw_columns(raw)
    data = raw.copy()

    mem_total_kb = data["mem_total_kb"] if "mem_total_kb" in data.columns else _FALLBACK_MEM_TOTAL_KB
    data["memory_usage_percent"] = (100 * data["rss_kb"] / mem_total_kb).clip(upper=100)
    data["turnaround_time_ms"] = data["waiting_time_ms"] + data["burst_time_ms"]
    data["cpu_usage_percent"] = (100 * data["burst_time_ms"] / data["window_ms"]).clip(upper=100)
    data["nice_value"] = data["nice"].clip(-20, 19)

    # RT tasks report priority < 0 in /proc/pid/stat; everything else maps
    # from nice value onto the 0-9 scale used at train time, matching
    # dashboard.py's own formula. hog_rt is forced to priority 0 (highest)
    # since chrt -f puts it outside the nice-value priority scale entirely.
    data["priority"] = data["nice_value"].apply(lambda n: min(max(int((n or 0) // 2) + 5, 0), 9))
    data.loc[data["process_label"] == "hog_rt", "priority"] = 0

    data["process_state"] = data["state_char"].map(STATE_MAP).fillna("waiting")
    data["times_preempted"] = data["nonvoluntary_ctxt_switches"]
    data["context_switches"] = data["voluntary_ctxt_switches"] + data["nonvoluntary_ctxt_switches"]
    return data


def add_engineered_features(data: pd.DataFrame, queueing_stats: dict | None = None) -> pd.DataFrame:
    data = data.copy()
    data["wait_to_burst_ratio"] = data["waiting_time_ms"] / (data["burst_time_ms"] + 1e-6)
    data["preemption_rate"] = data["times_preempted"] / (data["context_switches"] + 1)
    fair_share = 100 / (data["priority"] + 1)
    data["cpu_starvation_gap"] = (fair_share - data["cpu_usage_percent"]).clip(lower=0)
    data["low_priority_high_wait"] = (
        (data["priority"] >= 6) & (data["waiting_time_ms"] > data["waiting_time_ms"].median())
    ).astype(int)

    if queueing_stats:
        fallback = {
            "lambda": np.mean([v["lambda"] for v in queueing_stats.values()]),
            "E_S2": np.mean([v["E_S2"] for v in queueing_stats.values()]),
            "rho": np.mean([v["rho"] for v in queueing_stats.values()]),
        }
        expected_wait = data.apply(lambda r: pk_expected_wait(r, queueing_stats, fallback), axis=1)
        data["pk_wait_anomaly"] = data["waiting_time_ms"] / (expected_wait + 1e-6)
    else:
        data["pk_wait_anomaly"] = 1.0

    data["hrrn_risk_score"] = hrrn_risk_score(data["waiting_time_ms"], data["burst_time_ms"])
    return data


def predict_starvation(raw_df, model, scaler, encoders, feature_cols):
    data = raw_df.copy()
    for col, le in encoders.items():
        if col in data.columns:
            data[col] = data[col].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
            data[col] = le.transform(data[col])
    data = data.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(data)
    proba = model.predict_proba(X_scaled)[:, 1]
    return proba


def score_raw_csv(path, model, scaler, encoders, feature_cols, queueing_stats, threshold):
    """Full pipeline: raw collected CSV -> engineered features -> scored predictions."""
    raw = pd.read_csv(path)
    validate_raw_columns(raw, source=str(path))
    mapped = raw_to_model_input(raw)
    engineered = add_engineered_features(mapped, queueing_stats)
    proba = predict_starvation(engineered, model, scaler, encoders, feature_cols)
    engineered["model_proba"] = proba
    engineered["model_pred"] = (proba >= threshold).astype(int)
    return engineered
