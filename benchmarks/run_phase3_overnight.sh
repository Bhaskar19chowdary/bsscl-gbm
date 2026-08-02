#!/bin/bash

# Overnight Phase 3 Benchmark Orchestrator
# This script sequentially runs all heavy-duty scientific benchmarks.
# The 'caffeinate' command is used to prevent the Mac from sleeping.

echo "================================================================="
echo "  STARTING OVERNIGHT PHASE 3 BENCHMARK SUITE"
echo "  Start Time: $(date)"
echo "  Note: 'caffeinate' is actively keeping your Mac awake."
echo "================================================================="

cd "$(dirname "$0")/.."
export PYTHONPATH=src

# We wrap the entire block in caffeinate so the Mac physically cannot sleep until this finishes
caffeinate -i sh -c '
    echo ""
    echo ">>> 1. Running Statistical Significance (5-Fold CV)..."
    python3 benchmarks/suite_kfold_stats.py | tee benchmarks/logs_kfold.txt

    echo ""
    echo ">>> 2. Running Ablation Study (Component Importance)..."
    python3 benchmarks/suite_ablation.py | tee benchmarks/logs_ablation.txt

    echo ""
    echo ">>> 3. Running Scaling Graphs (10k to 1M rows)..."
    python3 benchmarks/suite_scaling_graphs.py | tee benchmarks/logs_scaling.txt

    echo ""
    echo ">>> 4. Running Optuna Hyperparameter Fairness (50 trials per model)..."
    python3 benchmarks/suite_optuna_tuning.py | tee benchmarks/logs_optuna.txt

    echo ""
    echo ">>> 5. Running Extreme Scale Test (11 Million Row Higgs Dataset)..."
    python3 benchmarks/suite_extreme_scale.py --confirm | tee benchmarks/logs_extreme.txt
'

echo ""
echo "================================================================="
echo "  OVERNIGHT SUITE COMPLETED SUCCESSFULLY!"
echo "  End Time: $(date)"
echo "  Your Mac is now allowed to go to sleep."
echo "================================================================="
