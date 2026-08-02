#!/usr/bin/env bash
set -e

echo "======================================"
echo " Packaging & Installation QA Test"
echo "======================================"

# 1. Clean previous builds
echo "🧹 Cleaning dist/ directory..."
rm -rf dist/ build/ *.egg-info

# 2. Build Wheel and Source Distribution
echo "📦 Building wheel and sdist..."
python3 -m pip install --upgrade build twine
python3 -m build

# 3. Check with Twine
echo "🔎 Running Twine check..."
python3 -m twine check dist/*

# 4. Test Installation in a temporary virtual environment
echo "🧪 Testing pip install from wheel..."
VENV_DIR=$(mktemp -d)
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install the wheel
pip install dist/*.whl

# Verify it can be imported (and doesn't silently fail)
echo "🔄 Verifying module import..."
python -c "from bsscl_gbm import HybridHistGBMNumbaV2; print('✅ Successfully imported HybridHistGBMNumbaV2 from installed wheel!')"

# Clean up
deactivate
rm -rf $VENV_DIR

echo "✅ Packaging test passed perfectly!"
