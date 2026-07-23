#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="${PROJECT_DIR}/build"

# Parse args
WARMUP=3
ITERATIONS=8
OUTPUT="tuning_data.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warmup) WARMUP="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Derive raw output path (same dir as output)
OUTPUT_DIR="$(dirname "$OUTPUT")"
RAW_OUTPUT="${OUTPUT_DIR}/autotune_raw.json"

echo "==> Building autotune binary..."
cmake --build "$BUILD_DIR" --target autotune 2>&1 | tail -3

echo "==> Running autotune (warmup=${WARMUP}, iterations=${ITERATIONS})..."
echo "    This may take several minutes depending on GPU."
"${BUILD_DIR}/benchmarks/autotune/autotune" "$WARMUP" "$ITERATIONS" "$RAW_OUTPUT"

echo "==> Deriving selection rules..."
python3 "${SCRIPT_DIR}/derive_rules.py" "$RAW_OUTPUT" --output "$OUTPUT"

echo ""
echo "Done! Files produced:"
echo "  Raw data:    $RAW_OUTPUT"
echo "  Tuning data: $OUTPUT"
