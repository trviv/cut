#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DIRS=("$ROOT_DIR/core" "$ROOT_DIR/operators" "$ROOT_DIR/tests" "$ROOT_DIR/benchmarks" "$ROOT_DIR/runtime")

echo "==> Running clang-format on .cpp and .h files"
find "${DIRS[@]}" \
  \( -name "*.cpp" -o -name "*.h" \) \
  | grep -v '\.generated\.h$' \
  | grep -v 'operators/compiled.*\.cpp$' \
  | xargs clang-format -i -style=file

echo "==> Done"
