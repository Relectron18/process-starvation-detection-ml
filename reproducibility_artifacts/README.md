# Reproducibility artifacts — Process Starvation Detection

Companion code for the paper's synthetic-data experiments (Sections 5.1–5.3),
real-hardware validation (5.4–5.5), and statistical robustness checks.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh
```

Runs tests, trains the model headlessly, scores all 8 collected real-hardware
trials, and repeats the model comparison across 5 seeds — everything except
re-collecting fresh real-hardware data (see below for why).

## Reproducing each result in the paper

| Paper section | Script | What it does |
|---|---|---|
| — (prerequisite for 5.4–5.5) | `train_model.py` | Headless equivalent of the notebook's training cells: generates the synthetic dataset (seed=42), compares 5 models, tunes the top 2 tree models, picks the best by F1, tunes the decision threshold, and saves everything to `models/`. Run this before `apply_model_to_real_data.py` / `aggregate_real_world.py` if you haven't run the notebook. |
| 5.1 Model comparison, 5.2 HRRN baseline, 5.3 Ablation | `multiseed_stability.py` | Runs the full pipeline (synthetic data → 5 models → HRRN baseline → feature ablation) across 5 independent seeds, with paired t-tests. `--seeds` overrides the default `42 7 123 2024 99`. |
| 5.4 Real-hardware validation | `collect_real_data.py rt <trial>` / `collect_real_data.py nice <trial>` | Spawns real competing processes on the host and samples `/proc/[pid]/schedstat` directly. **Requires Linux with `chrt`/`nice` and permission to set `SCHED_FIFO`.** The script now verifies via `os.sched_getscheduler` that the RT hog process actually got `SCHED_FIFO` and warns loudly (writing `<trial>_rt_verification.json`) if it didn't — a silent privilege failure here would otherwise produce a trial mislabeled as "RT contention" with no real preemption behind it. The 8 CSVs already included (`real_data_{rt,nice}_trial*.csv`) are the trials reported in the paper — you don't need to re-collect them to reproduce Table 2. |
| 5.4–5.5 Scoring real data | `apply_model_to_real_data.py [in_csv] [out_csv]` | Applies the synthetic-trained model to one real-hardware CSV. |
| 5.4–5.5 Aggregated real-world results + threshold sensitivity | `aggregate_real_world.py` | Runs `apply_model_to_real_data`'s logic over every collected trial, aggregates mean ± std per scenario, and reruns the ground-truth threshold sweep. Reproduces `real_world_multitrial_results.csv` and `threshold_sensitivity.csv`. |
| 5.6 Feature-access + model-family follow-up | `feature_access_and_model_family_check.py` | Answers the paper's own Future Work question about `pk_wait_anomaly`: (a) ablates it on MLPClassifier, a model family not tried elsewhere in the paper, and (b) reruns the Random Forest ablation with `waiting_time_ms`/`wait_to_burst_ratio` withheld, to test whether its near-zero effect elsewhere is because those raw columns make it redundant. Also reports the raw correlation between `pk_wait_anomaly` and those columns. Reproduces `mlp_model_family_ablation.csv`, `feature_access_ablation.csv`, and `pk_wait_anomaly_correlations.csv`. `--seeds` overrides the default 5 seeds, same as `multiseed_stability.py`. |
| 5.6(b) control | `control_ablation_check.py` | Control for the reduced-access ablation above: withholds two columns essentially *uncorrelated* with `pk_wait_anomaly` (`memory_usage_percent`, `num_threads`) instead of the redundant ones, to check whether 5.6(b)'s positive effect is specifically about redundancy or just "fewer features helps." Reproduces `control_ablation.csv` and `pk_wait_anomaly_correlations_control.csv`. See CHANGES.md item 4 (third revision). |

The notebook (`starvation_detection_project.ipynb`) is the primary,
narrated walkthrough (single seed=42 run, plots, dashboard). The scripts
here are the seed-repeated / real-hardware companion pieces used for the
statistical claims in the paper — `pipeline_core.py` holds the exact
same data-generation and feature logic as the notebook so results agree.

## Tests

```bash
pytest tests/
```

Smoke tests for determinism, absence of NaNs, and basic invariants in
`pipeline_core.py` (not model accuracy — that's a separate concern
covered by `multiseed_stability.py`'s own significance tests).

## Cross-seed real-hardware check: matched tuning budget

`multiseed_real_world_results.csv` now holds results from `multiseed_real_world_check.py`
run with `models_seed7_matched/`/`models_seed123_matched/` (both retrained with the same
`--n-iter 20` budget as seed 42's `models/`), which is what the paper's Sections 5.4 and 6
report. The original reduced-budget run (`n_iter=8` for seeds 7/123) is kept for provenance
as `multiseed_real_world_results_reduced_budget_ORIGINAL.csv`; the two runs agree closely
(see CHANGES.md item 5, third revision), so the threshold-transfer finding is not an
artifact of unequal tuning effort.

## Known limitations (see paper Section 6 for the full list)

- Real-hardware validation is confined to a single machine and CPU core.
- `mem_total_kb` is now recorded per-collection in `collect_real_data.py`
  output; CSVs collected before this fix fall back to an approximate
  constant in `real_data_features.py` (`_FALLBACK_MEM_TOTAL_KB`) — the 8
  trial CSVs shipped with this package predate the fix and use that
  fallback, which is why the model's `memory_usage_percent` input carries
  a small amount of approximation noise in those specific trials.
