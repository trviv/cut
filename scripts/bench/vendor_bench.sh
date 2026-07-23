#!/usr/bin/env bash
set -euo pipefail

# This script builds and runs all vendor comparison benchmarks, then produces a
# single combined CUT-vs-vendor table. See benchmarks/vendor/README.md for the
# methodology.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
BUILD_DIR="${PROJECT_DIR}/build-cuda-rel"
OUT_DIR="${PROJECT_DIR}/vendor_bench_results"
REPETITIONS=3
MIN_TIME=""
FILTER=""
DO_BUILD=1
QUICK=0
NO_LARGE=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --min-time) MIN_TIME="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    --quick) QUICK=1; shift ;;
    --no-large) NO_LARGE=1; shift ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --build-dir DIR       override BUILD_DIR (default: ${PROJECT_DIR}/build-cuda-rel)"
      echo "  --out-dir DIR         override OUT_DIR (default: ${PROJECT_DIR}/vendor_bench_results)"
      echo "  --repetitions N       override REPETITIONS (default: 3)"
      echo "  --min-time T          passed through as --benchmark_min_time=T (e.g. 0.1s)"
      echo "  --filter REGEX        passed through as --benchmark_filter=REGEX"
      echo "  --no-build            skip the cmake build step"
      echo "  --quick               skip the two largest sort_radix sizes (N=4M and N=16M)"
      echo "  --no-large            skip the model-scale cases (*_large and every conv2d)"
      echo "  -h|--help             print usage and exit"
      echo ""
      echo "Notes:"
      echo "  - A full run is dominated by CUT's pathologically slow multi-pass sortRadix."
      echo "    Use --quick to skip the two largest sort sizes."
      echo "  - The model-scale cases hold up to 20 GB of VRAM and take minutes; --no-large"
      echo "    drops them. Cases that do not fit the card are skipped automatically."
      echo "  - --quick and --no-large combine; either can be used with --filter only if you"
      echo "    write the exclusion into the filter yourself."
      echo "  - Individual binaries take the full Google Benchmark flag set; this script is a"
      echo "    convenience wrapper, not a replacement."
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Handle --quick / --no-large. Both are exclusions, and Google Benchmark takes
# exactly one filter, so they are combined into a single negated alternation
# rather than applied one at a time.
EXCLUDES=()
[[ $QUICK -eq 1 ]] && EXCLUDES+=("sort_radix/N=(4194304|16777216)")
[[ $NO_LARGE -eq 1 ]] && EXCLUDES+=("(_large|conv2d)/")
if [[ ${#EXCLUDES[@]} -gt 0 ]]; then
  if [[ -n "$FILTER" ]]; then
    echo "Error: --quick/--no-large and --filter cannot be used together" >&2
    exit 1
  fi
  FILTER="-($(IFS='|'; echo "${EXCLUDES[*]}"))"
fi

# Build step
if [[ $DO_BUILD -eq 1 ]]; then
  if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: BUILD_DIR does not exist: $BUILD_DIR" >&2
    echo "Please configure first with:" >&2
    echo "  cmake -B build-cuda-rel -DENABLE_CUDA_BACKEND=ON -DCMAKE_BUILD_TYPE=Release" >&2
    exit 1
  fi

  echo "==> Building vendor benchmarks..."
  for target in cublas_matmul_bench cublas_extras_bench cudnn_softmax_bench cudnn_conv_bench cub_scan_sort_bench rocblas_matmul_bench rocprim_scan_sort_bench; do
    cmake --build "$BUILD_DIR" --target "$target" >/dev/null 2>&1 || true
  done
  echo "    (Targets built one at a time; missing targets are skipped if their vendor SDK was not found)"
fi

# Run step
mkdir -p "$OUT_DIR"
JSONS=()
FAILED=0

# Candidate binaries as "<subdir>/<name>" pairs
candidates=(
  "cuda/cublas_matmul_bench"
  "cuda/cublas_extras_bench"
  "cuda/cudnn_softmax_bench"
  "cuda/cudnn_conv_bench"
  "cuda/cub_scan_sort_bench"
  "amd/rocblas_matmul_bench"
  "amd/rocprim_scan_sort_bench"
)

for candidate in "${candidates[@]}"; do
  IFS='/' read -r subdir name <<< "$candidate"
  binary_path="${BUILD_DIR}/benchmarks/vendor/${subdir}/${name}"

  if [[ ! -x "$binary_path" ]]; then
    echo "    skipped (not built): ${name}"
    continue
  fi

  echo "==> Running ${name}..."
  json="${OUT_DIR}/${name}.json"
  flags=(
    --benchmark_repetitions="$REPETITIONS"
    --benchmark_report_aggregates_only=true
    --benchmark_out="$json"
    --benchmark_out_format=json
  )
  [[ -n "$MIN_TIME" ]] && flags+=("--benchmark_min_time=$MIN_TIME")
  [[ -n "$FILTER" ]] && flags+=("--benchmark_filter=$FILTER")

  # rc must be reset every iteration: `|| rc=$?` only assigns on failure, so a
  # stale value would make every benchmark after the first failure report FAILED,
  # and an unset rc trips `set -u` on the first iteration.
  rc=0
  "$binary_path" "${flags[@]}" >/dev/null || rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED: ${name}"
    # Assignment, not ((FAILED++)): post-increment evaluates to the OLD value, so
    # when FAILED is 0 the arithmetic result is 0, which bash reports as exit
    # status 1 — under `set -e` that would abort on the first failing benchmark,
    # the opposite of recording it and carrying on.
    FAILED=$((FAILED + 1))
    continue
  fi

  # A filter that matches nothing in this binary still exits 0 but leaves an
  # empty --benchmark_out file. That is expected when filtering for one operator,
  # so report it and move on rather than handing an unparseable file to
  # vendor_compare.py. It is not a failure and must not count as one.
  if [[ ! -s "$json" ]] || ! grep -q '"name"' "$json" 2>/dev/null; then
    echo "    no cases matched: ${name}"
    continue
  fi

  JSONS+=("$json")
done

# Compare step
if [[ ${#JSONS[@]} -eq 0 ]]; then
  echo "Error: no benchmarks produced results" >&2
  exit 1
fi

echo ""
echo "==> Combined comparison"
echo ""
python3 "${SCRIPT_DIR}/vendor_compare.py" "${JSONS[@]}"

echo ""
echo "Done! JSON written to: $OUT_DIR"
if [[ $FAILED -gt 0 ]]; then
  echo "Note: $FAILED benchmarks failed"
  exit 1
fi
