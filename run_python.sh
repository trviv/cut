#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_DIR="$ROOT_DIR/interface/python"

# Activate the virtual environment
echo "==> Activating virtual environment at $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Build and install the latest cut changes
echo ""
echo "==> Installing cut Python package (editable, with rebuild)"
pip install --no-build-isolation -e "$PYTHON_DIR"

# Run all tests
echo ""
echo "==> Running tests"
pytest "$PYTHON_DIR/tests" -v

# Run all benchmarks
echo ""
echo "==> Running benchmarks"
# python "$PYTHON_DIR/benchmarks/benchmark.py" --backend vulkan
# echo ""
# python "$PYTHON_DIR/benchmarks/benchmark_chained_ops.py"
echo ""
python "$PYTHON_DIR/benchmarks/run_benchmarks.py"
