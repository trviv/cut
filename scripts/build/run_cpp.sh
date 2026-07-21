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
  echo "==> Configuring (cmake)"
  cmake -B "$BUILD_DIR"

  echo ""
  echo "==> Building (library + tests)"
  cmake --build "$BUILD_DIR" -j "$(nproc)"

  echo ""
  echo "==> Running C++ tests"
  LSAN_OPTIONS="suppressions=$ROOT_DIR/cmake/lsan.supp:fast_unwind_on_malloc=0" "$BUILD_DIR/tests/tests"
fi
