#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Running clang-format on .cpp and .h files"
find "$ROOT_DIR" \
  -path "$ROOT_DIR/build" -prune -o \
  -path "$ROOT_DIR/.venv" -prune -o \
  \( -name "*.cpp" -o -name "*.h" \) -print \
  | grep -v '\.generated\.h$' \
  | grep -v 'operators/compiled.*\.cpp$' \
  | xargs clang-format -i -style=file

echo "==> Done"
