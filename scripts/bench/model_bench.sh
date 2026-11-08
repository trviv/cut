#!/usr/bin/env bash
set -euo pipefail

# Builds the CUT runner, measures every model case against llama.cpp, and
# prints one combined table. The model-scale counterpart of vendor_bench.sh;
# methodology is in benchmarks/model/README.md.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BUILD_DIR="${PROJECT_DIR}/build-cuda-rel"
OUT_DIR="${PROJECT_DIR}/model_bench_results"
LLAMA_DIR="${PROJECT_DIR}/build/external_runners/llama.cpp"
DO_BUILD=1
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --llama-dir) LLAMA_DIR="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    # Everything else is model_bench.py's: --quick, --runs, --warmup, --tier,
    # --backend, --gpu, --include-igpu, --no-check, and explicit model paths.
    -h|--help)
      echo "Usage: $0 [options] [model.gguf ...]"
      echo ""
      echo "  --build-dir DIR   CMake build directory (default: build-cuda-rel)"
      echo "  --out-dir DIR     where the per-GPU JSON lands (default: model_bench_results)"
      echo "  --llama-dir DIR   llama.cpp build tree (default: build/external_runners/llama.cpp)"
      echo "  --no-build        skip the build step"
      echo ""
      echo "Passed through to model_bench.py:"
      python3 "${SCRIPT_DIR}/model_bench.py" --help | sed -n '/^options:/,$p' | tail -n +2
      exit 0
      ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

# Build step. The runner is rebuilt by default for the same reason the vendor
# benches are: the operator variant table is compiled in and tuning_data.json is
# read by index at runtime, so a stale binary silently measures a different
# kernel than the one under test.
if [[ $DO_BUILD -eq 1 ]]; then
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
    configure_log="$(mktemp)"
    if ! cmake -B "$BUILD_DIR" -DENABLE_CUDA_BACKEND=ON \
         -DCMAKE_BUILD_TYPE=Release >"$configure_log" 2>&1; then
      echo "Error: cmake configure failed:" >&2
      cat "$configure_log" >&2
      rm -f "$configure_log"
      exit 1
    fi
    rm -f "$configure_log"
  fi

  echo "==> Building gguf_example"
  build_log="$(mktemp)"
  if ! cmake --build "$BUILD_DIR" --target gguf_example \
       -j"$(nproc 2>/dev/null || echo 4)" >"$build_log" 2>&1; then
    echo "Error: build failed:" >&2
    cat "$build_log" >&2
    rm -f "$build_log"
    exit 1
  fi
  rm -f "$build_log"
fi

CUT_BINARY="${BUILD_DIR}/interface/runner/llama/gguf_example"
if [[ ! -x "$CUT_BINARY" ]]; then
  echo "Error: ${CUT_BINARY} not built" >&2
  exit 1
fi

# The reference has to exist before anything is measured. Without it every case
# would report a CUT number and no comparison, which is a benchmark that cannot
# answer its own question.
if [[ ! -x "${LLAMA_DIR}/build_vk/bin/llama-bench" && \
      ! -x "${LLAMA_DIR}/build_cuda/bin/llama-bench" ]]; then
  echo "Error: no llama-bench found under ${LLAMA_DIR}" >&2
  echo "       Build the reference runners first:" >&2
  echo "         ./scripts/setup/setup_benchmark_runners.sh" >&2
  exit 1
fi

echo "==> Measuring"
rc=0
python3 "${SCRIPT_DIR}/model_bench.py" \
  --cut-binary "$CUT_BINARY" --llama-dir "$LLAMA_DIR" --out-dir "$OUT_DIR" \
  ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} || rc=$?
# Exit 2 is "a case failed its coherence check": the JSON is valid and MUST
# still be compared, since that is where the VOID rows come from.
if [[ $rc -ne 0 && $rc -ne 2 ]]; then
  echo "Error: model_bench.py failed with status $rc" >&2
  exit $rc
fi
FAILED=$rc

shopt -s nullglob
JSONS=("${OUT_DIR}"/*.json)
shopt -u nullglob
if [[ ${#JSONS[@]} -eq 0 ]]; then
  echo "Error: no results produced" >&2
  exit 1
fi

echo ""
echo "==> Combined comparison"
echo ""
compare_rc=0
python3 "${SCRIPT_DIR}/model_compare.py" "${JSONS[@]}" || compare_rc=$?
if [[ $compare_rc -ne 0 && $compare_rc -ne 2 ]]; then
  echo "Error: model_compare.py failed with status $compare_rc" >&2
  exit $compare_rc
fi

echo ""
echo "Done! JSON written to: $OUT_DIR"
echo "Record it on the results page with:"
echo "  python3 scripts/bench/render_model_bench.py --snapshot"

if [[ $FAILED -eq 2 || $compare_rc -eq 2 ]]; then
  echo ""
  echo "Note: at least one case is VOID — CUT stopped producing language, or a"
  echo "      side produced no number at all. Both are named above the table,"
  echo "      and neither leaves a timing that can be quoted."
  exit 2
fi
