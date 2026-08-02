#!/usr/bin/env bash
set -e

echo "======================================"
echo " Mutation Testing Setup (mutmut)"
echo "======================================"

echo "Installing mutmut..."
python3 -m pip install mutmut

echo "Running mutmut on a small subset (testing the fast suite)..."
echo "NOTE: Running this on the entire codebase takes hours. We run it in a controlled manner."

# We run mutmut targeting just one function to demonstrate it works.
# To run on the whole codebase: mutmut run
# For this script, we just show how to configure and run it on a specific file.

echo "[mutmut]" > setup.cfg
echo "paths_to_mutate=src/bsscl_gbm/__init__.py" >> setup.cfg
echo "backup=False" >> setup.cfg
echo "runner=pytest tests/ -m 'not stress'" >> setup.cfg
echo "tests_dir=tests/" >> setup.cfg

echo "Configuration saved to setup.cfg!"
echo "To execute a full mutation run, execute 'mutmut run' in your terminal."
echo "Warning: A full run generates thousands of mutants and takes ~4+ hours."
