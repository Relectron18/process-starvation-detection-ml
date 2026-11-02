# Reproducibility artifacts — Process Starvation Detection

Companion code and data for the paper's synthetic experiments, real hardware
validation, added workloads, overhead measurements and statistical checks.

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh
```

Runs tests, trains the model headlessly, scores all 8 collected real-hardware
trials, and repeats the model comparison across 12 seeds — everything except
re-collecting fresh real-hardware data (see below for why).

## Reproducing each result in the paper

Section and table numbers refer to the revised manuscript. Paths in `revision/` are relative to that folder.

| Paper result | Script | Output |
|---|---|---|
| Stored models for Sections 4 and 5 | `train_model.py` | `models/`, `models_seed7_matched/`, `models_seed123_matched/` |
| Table 3 learned models, Fig. 1 | `multiseed_stability.py`, `revision/synth_eval.py` | `multiseed_results.csv`, `revision/results/synth/metrics_5models.csv` |
| Table 3 HRRN, tuned HRRN and waiting time rows | `revision/synth_eval.py` | `revision/results/synth/hrrn_fair.csv` |
| Table 4 class imbalance | `revision/synth_eval.py` | `revision/results/synth/imbalance.csv` |
| Table 5 feature importance | `revision/synth_eval.py`, `revision/real_eval.py` | `revision/results/synth/feature_importance.csv`, `revision/results/o2/real_perm_importance.csv` |
| Table 6 ablations | `multiseed_stability.py`, `feature_access_and_model_family_check.py`, `control_ablation_check.py`, `significance_summary.py` | `multiseed_ablation.csv`, `feature_access_ablation.csv`, `mlp_model_family_ablation.csv`, `control_ablation.csv`, `multiple_comparisons_correction.csv` |
| Table 7, Fig. 2, threshold sensitivity | `collect_real_data.py`, `aggregate_real_world.py` | `real_data_*_trial*.csv`, `real_world_multitrial_results.csv`, `threshold_sensitivity.csv` |
| Table 8 matched budget check | `multiseed_real_world_check.py` | `multiseed_real_world_results.csv` |
| Table 9 comparison rules | `revision/real_eval.py`, `revision/real_deep.py` | `revision/results/o2/real_baselines_by_trial.csv`, `revision/results/o2/recall_by_process.csv` |
| Table 10 all scenarios, Fig. 3 | `revision/collect_ext.py`, `revision/app_workloads.py`, `revision/ext_eval.py` | `revision/results/res/ext_*_trial*.csv`, `revision/results/ext/` |
| Table 11 collection overhead | `revision/overhead.py` | `revision/results/res/overhead.csv`, `revision/results/res/overhead_throughput.csv` |
| Table 12 first revision fixes | documented in `CHANGES.md` | none |

`cd revision && python gen_tables.py` rebuilds Tables 3, 4, 5, 9, 10 and 11 as LaTeX in `revision/tables/`. `python make_figs.py` rebuilds Figs. 1 to 3 in `revision/figures/`.

`revision/collect_all.sh <outdir>` collects the 30 added trials and the overhead measurements on a Linux host. It needs root for SCHED_FIFO, negative nice values and cgroups, and it expects a Python environment at `revision/venv` unless `PY` is set. Evaluation scripts take this folder as their first argument, for example `python synth_eval.py .. results/synth`.

`pipeline_core.py` holds the data generation and feature logic.

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
`--n-iter 20` budget as seed 42's `models/`), which is what the paper's Sections 5 and 6
report. The original reduced-budget run (`n_iter=8` for seeds 7/123) agrees closely
(see CHANGES.md item 5, third revision), so the threshold-transfer finding is not an
artifact of unequal tuning effort.

## Known limitations (see paper Section 6 for the full list)

- Real-hardware validation is confined to virtualized environments, two kernels and at most two vCPUs.
- `mem_total_kb` is now recorded per-collection in `collect_real_data.py`
  output; CSVs collected before this fix fall back to an approximate
  constant in `real_data_features.py` (`_FALLBACK_MEM_TOTAL_KB`) — the 8
  trial CSVs shipped with this package predate the fix and use that
  fallback, which is why the model's `memory_usage_percent` input carries
  a small amount of approximation noise in those specific trials.
