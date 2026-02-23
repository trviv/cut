#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "==> Configuring (cmake - Xcode)"
cmake -B "$BUILD_DIR" -G Xcode

echo ""
echo "==> Building (library + tests)"
cmake --build "$BUILD_DIR" -j 9

echo ""
echo "==> Running C++ tests"
"$BUILD_DIR/tests/Debug/tests"
