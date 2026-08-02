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
INTERLEAVE=0

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
    # Removes the systematic ordering bias: without it Google Benchmark runs
    # cut/<case> to completion and then <vendor>/<case>, so the vendor half
    # always inherits a card the CUT half just warmed. Measured at ~5% on
    # compute-bound shapes (and nothing on bandwidth-bound ones), which matters
    # only for the near-parity rows — hence opt-in rather than default.
    #
    # Implies --no-large, and that is not a convenience: random interleaving
    # reorders repetitions across every registered benchmark, which defeats the
    # model-scale tier's one-resident-case scheme and would re-upload multiple
    # gigabytes per repetition instead of once per case.
    --interleave) INTERLEAVE=1; NO_LARGE=1; shift ;;
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
      echo "  --interleave          randomly interleave repetitions, removing the"
      echo "                        cut-then-vendor ordering bias (implies --no-large)"
      echo "  -h|--help             print usage and exit"
      echo ""
      echo "Notes:"
      echo "  - A full run is dominated by CUT's pathologically slow multi-pass sortRadix."
      echo "    Use --quick to skip the two largest sort sizes."
      echo "  - The model-scale cases hold up to 20 GB of VRAM and take minutes; --no-large"
      echo "    drops them. Cases that do not fit the card are skipped automatically."
      echo "  - --quick, --no-large and --interleave combine; any of them can be used with"
      echo "    --filter only if you write the exclusion into the filter yourself."
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
    echo "Error: --quick/--no-large/--interleave and --filter cannot be used together" >&2
    exit 1
  fi
  FILTER="-($(IFS='|'; echo "${EXCLUDES[*]}"))"
fi

TARGETS=(cublas_matmul_bench cublas_extras_bench cudnn_softmax_bench
         cudnn_conv_bench cub_scan_sort_bench rocblas_matmul_bench
         rocprim_scan_sort_bench)

# Build step
if [[ $DO_BUILD -eq 1 ]]; then
  # Configure the build directory ourselves rather than telling the user to.
  #
  # Vulkan is not an option in this project — CMakeLists.txt does
  # find_package(Vulkan REQUIRED) — so every build already has the Vulkan
  # backend. CUDA is the opt-in one, and it is the one that matters here:
  # without ENABLE_CUDA_BACKEND every target under vendor/cuda is skipped at
  # configure time, and the run then reports "skipped (not built)" for all of
  # them. That is indistinguishable from a missing cuBLAS/cuDNN SDK, so a
  # wrongly-configured directory produces an empty table and no explanation.
  need_configure=0
  if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "==> ${BUILD_DIR} not configured yet"
    need_configure=1
  elif ! grep -q "^ENABLE_CUDA_BACKEND:BOOL=ON" "${BUILD_DIR}/CMakeCache.txt"; then
    echo "==> ${BUILD_DIR} was configured without the CUDA backend"
    need_configure=1
  fi
  if [[ $need_configure -eq 1 ]]; then
    echo "    configuring with CUDA backend ON (Vulkan is always on)"
    # Both streams go to a log: the configure emits pages of third-party
    # FetchContent policy warnings that would bury the benchmark output. The
    # log is replayed only if configure actually fails, where it is the thing
    # you need to read.
    configure_log="$(mktemp)"
    if ! cmake -B "$BUILD_DIR" -DENABLE_CUDA_BACKEND=ON \
         -DCMAKE_BUILD_TYPE=Release >"$configure_log" 2>&1; then
      echo "Error: cmake configure failed:" >&2
      cat "$configure_log" >&2
      rm -f "$configure_log"
      exit 1
    fi
    # Which vendor SDKs were found is the one genuinely useful line, and it
    # explains any target missing from the table below.
    grep -E "^-- vendor/(cuda|amd):" "$configure_log" | sed 's/^/    /' || true
    rm -f "$configure_log"
  fi

  # Only build targets this configuration actually defines. `cmake --build`
  # fails the WHOLE invocation on one unknown target, so the AMD benches on an
  # NVIDIA box (or the cuDNN ones without cuDNN) would take everything down
  # with them if passed blindly.
  available=()
  target_list="$(cmake --build "$BUILD_DIR" --target help 2>/dev/null || true)"
  for target in "${TARGETS[@]}"; do
    if [[ -z "$target_list" ]] || grep -qE "^\.\.\. ${target}$" <<<"$target_list"; then
      available+=("$target")
    fi
  done

  if [[ ${#available[@]} -eq 0 ]]; then
    echo "Error: none of the vendor benchmark targets exist in ${BUILD_DIR}." >&2
    echo "       Check the configure output for which vendor SDKs were found." >&2
    exit 1
  fi

  echo "==> Building ${#available[@]} vendor benchmark(s): ${available[*]}"
  # One parallel invocation. Building one target at a time re-ran the
  # up-to-date and shader-compile checks per target, and left the bulk of the
  # work — CUTLib's own translation units — compiling serially.
  build_log="$(mktemp)"
  if ! cmake --build "$BUILD_DIR" --target "${available[@]}" \
       -j"$(nproc 2>/dev/null || echo 4)" >"$build_log" 2>&1; then
    echo "Error: build failed:" >&2
    cat "$build_log" >&2
    rm -f "$build_log"
    exit 1
  fi
  rm -f "$build_log"
  echo "    (Targets whose vendor SDK was not found are not defined and were skipped)"
fi

# Run step
mkdir -p "$OUT_DIR"
JSONS=()
FAILED=0
INCORRECT=0

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
  # Needs --benchmark_repetitions > 1 to do anything, which is always set above.
  [[ $INTERLEAVE -eq 1 ]] && flags+=("--benchmark_enable_random_interleaving=true")

  # rc must be reset every iteration: `|| rc=$?` only assigns on failure, so a
  # stale value would make every benchmark after the first failure report FAILED,
  # and an unset rc trips `set -u` on the first iteration.
  rc=0
  "$binary_path" "${flags[@]}" >/dev/null || rc=$?
  # Exit 2 means the binary ran fine but at least one case disagreed with its
  # vendor reference. Its JSON is valid and MUST still be compared — that is
  # where the FAIL rows come from — so this is recorded and falls through
  # rather than being treated as a crash. Dropping the file here would hide
  # exactly the failures the correctness gate exists to surface.
  if [[ $rc -eq 2 ]]; then
    echo "    CORRECTNESS FAILURES: ${name} (see the table below)"
    INCORRECT=$((INCORRECT + 1))
    rc=0
  fi
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
# vendor_compare.py exits 2 when any comparison failed its correctness gate.
# Captured rather than allowed to abort under `set -e`, so the summary below
# still prints and the exit status can distinguish the two kinds of failure.
compare_rc=0
python3 "${SCRIPT_DIR}/vendor_compare.py" "${JSONS[@]}" || compare_rc=$?
if [[ $compare_rc -eq 2 ]]; then
  INCORRECT=$((INCORRECT + 1))
elif [[ $compare_rc -ne 0 ]]; then
  echo "Error: vendor_compare.py failed with status $compare_rc" >&2
  exit $compare_rc
fi

echo ""
echo "Done! JSON written to: $OUT_DIR"
if [[ $FAILED -gt 0 ]]; then
  echo "Note: $FAILED benchmarks failed to run"
fi
if [[ $INCORRECT -gt 0 ]]; then
  echo "Note: at least one comparison FAILED its correctness gate — CUT's output"
  echo "      disagreed with the vendor reference. Those timings are not quotable."
fi
# Correctness failures outrank run failures in the exit status: a benchmark that
# did not run is a gap, but one that ran fast and wrong is a wrong answer.
if [[ $INCORRECT -gt 0 ]]; then
  exit 2
fi
if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
