#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

if [[ "$(uname)" == "Darwin" ]]; then
  echo "==> Configuring (cmake - Xcode)"
  cmake -B "$BUILD_DIR" -G Xcode

  echo ""
  echo "==> Building (library + tests)"
  cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.logicalcpu)"

  echo ""
  echo "==> Running C++ tests"
  "$BUILD_DIR/tests/Debug/tests"
else
  LSAN_OPTS="suppressions=$ROOT_DIR/cmake/lsan.supp:fast_unwind_on_malloc=0"

  echo "==> Configuring (cmake)"
  cmake -B "$BUILD_DIR"

  echo ""
  echo "==> Building (library + tests)"
  cmake --build "$BUILD_DIR" -j "$(nproc)"

  echo ""
  echo "==> Running C++ tests (Vulkan)"
  vulkan_rc=0
  LSAN_OPTIONS="$LSAN_OPTS" "$BUILD_DIR/tests/tests" || vulkan_rc=$?

  # CUDA backend: configure + build + run the same suite when an NVIDIA GPU is
  # present. A missing CUDA toolkit surfaces as a build failure and is treated
  # as a visible skip (not a hard failure); actual CUDA test failures fail the
  # script. Use run_tests_both_backends.sh for stricter CI gating.
  cuda_rc=0
  cuda_status="skipped (no NVIDIA GPU)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CUDA_BUILD_DIR="$ROOT_DIR/build-cuda"
    echo ""
    echo "==> NVIDIA GPU detected; configuring + building CUDA backend ($CUDA_BUILD_DIR)"
    if cmake -B "$CUDA_BUILD_DIR" -DENABLE_CUDA_BACKEND=ON \
        && cmake --build "$CUDA_BUILD_DIR" -j "$(nproc)" --target tests; then
      echo ""
      echo "==> Running C++ tests (CUDA)"
      CUT_TEST_BACKEND=cuda LSAN_OPTIONS="$LSAN_OPTS" \
        "$CUDA_BUILD_DIR/tests/tests" || cuda_rc=$?
      cuda_status="$([[ $cuda_rc -eq 0 ]] && echo PASS || echo FAIL)"
    else
      echo ""
      echo "==> WARNING: CUDA configure/build failed (missing CUDA toolkit?); skipping CUDA tests"
      cuda_status="skipped (CUDA build failed)"
    fi
  fi

  echo ""
  echo "======================================================================"
  echo "==> Summary: vulkan=$([[ $vulkan_rc -eq 0 ]] && echo PASS || echo FAIL)  cuda=$cuda_status"
  [[ "$vulkan_rc" -eq 0 && "$cuda_rc" -eq 0 ]] || exit 1
fi
