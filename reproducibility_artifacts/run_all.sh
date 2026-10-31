#!/usr/bin/env bash
# Runs the full headless reproduction pipeline in dependency order.
# Does NOT re-collect real-hardware data (collect_real_data.py) since that
# requires spawning real competing processes on a Linux host with RT
# scheduling privileges and takes ~46s per trial x 8 trials; the 8 CSVs
# already in this directory are the trials reported in the paper.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs models models_seed7 models_seed123
echo "== 0/8: recording environment =="
{
  echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python: $(python3 --version 2>&1)"
  echo "platform: $(uname -a)"
  echo "---- pip freeze ----"
  pip freeze
} > outputs/environment.txt
echo "  -> outputs/environment.txt"

echo "== 1/8: unit tests =="
pytest tests/ -q

echo "== 2/8: train model (primary seed=42, notebook-equivalent) =="
python3 train_model.py --seed 42 --models-dir models

echo "== 3/8: real-hardware validation, all 8 collected trials + threshold sensitivity =="
python3 aggregate_real_world.py

echo "== 4/8: multi-seed statistical robustness (12 seeds) =="
python3 multiseed_stability.py

echo "== 5/8: feature-access + model-family follow-up (why doesn't pk_wait_anomaly help?) =="
python3 feature_access_and_model_family_check.py

echo "== 6/8: multiple-comparisons correction across all 5 significance tests =="
python3 significance_summary.py

echo "== 7/8: train 2 extra seeds and check real-hardware generalization across seeds =="
echo "  (reduced RandomizedSearchCV iterations -- this is a robustness check on Table 2,"
echo "   not the primary reported number, so it doesn't need the full tuning budget)"
python3 train_model.py --seed 7   --n-iter 8 --models-dir models_seed7
python3 train_model.py --seed 123 --n-iter 8 --models-dir models_seed123
python3 multiseed_real_world_check.py --model-dirs seed42=models seed7=models_seed7 seed123=models_seed123

echo "== 8/8: done =="
echo "To re-collect fresh real-hardware trials instead of using the shipped CSVs:"
echo "  python3 collect_real_data.py rt <trial_number>"
echo "  python3 collect_real_data.py nice <trial_number>"
