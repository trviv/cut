#!/usr/bin/env bash
set -uo pipefail

# Build and run the CUT gtest suite against BOTH compute backends (Vulkan and
# CUDA), selected via the CUT_TEST_BACKEND environment variable, and print a
# per-backend pass/fail summary.
#
# Usage: scripts/build/run_tests_both_backends.sh [build_dir]
#   build_dir: build directory (default: build-cuda-rel). The CUDA backend
#              requires that build be configured with -DENABLE_CUDA_BACKEND=ON;
#              the Vulkan backend also runs from the same build.
#   Honors the GTEST_FILTER env var (default: "*").

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUILD_DIR="${1:-build-cuda-rel}"
GTEST_FILTER="${GTEST_FILTER:-*}"

echo "Building tests in $BUILD_DIR..."
cmake --build "$BUILD_DIR" --target tests -j"$(nproc)" \
  || { echo "build failed" >&2; exit 1; }

TESTBIN="$(find "$BUILD_DIR/tests" -maxdepth 2 -name tests -type f | head -1)"
if [[ ! -x "$TESTBIN" ]]; then
  echo "Error: test binary not found or not executable: $TESTBIN" >&2
  exit 1
fi

run_backend() {
  local backend="$1"
  echo
  echo "======================================================================"
  echo "==> Running tests on backend: $backend"
  echo "======================================================================"
  CUT_TEST_BACKEND="$backend" ASAN_OPTIONS=detect_leaks=0 \
    "$TESTBIN" --gtest_filter="$GTEST_FILTER"
  local rc=$?
  echo "[$backend] exit code: $rc"
  return "$rc"
}

status() { [[ "$1" -eq 0 ]] && echo PASS || echo FAIL; }

vulkan_rc=0
cuda_rc=0
run_backend vulkan || vulkan_rc=$?
run_backend cuda || cuda_rc=$?

echo
echo "RESULT: vulkan=$(status "$vulkan_rc") cuda=$(status "$cuda_rc")"

[[ "$vulkan_rc" -eq 0 && "$cuda_rc" -eq 0 ]] || exit 1
