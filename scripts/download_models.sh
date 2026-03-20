#!/usr/bin/env bash
# Download SmolLM2 and LLaMA 3 GGUF models that fit within 16GB RAM.
#
# Usage:
#   ./scripts/download_models.sh              # Download all models
#   ./scripts/download_models.sh smol          # SmolLM2 models only
#   ./scripts/download_models.sh llama         # LLaMA 3 models only
#   ./scripts/download_models.sh --list        # Show available models and sizes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

# ── Model catalog ──────────────────────────────────────────────────────────
# Format: "repo_id|filename|approx_size_MB|description"
# All models chosen to fit comfortably in 16GB RAM.

SMOL_MODELS=(
  "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF|smollm2-135m-instruct-f16.gguf|270|SmolLM2 135M F16"
  "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF|smollm2-135m-instruct-q8_0.gguf|143|SmolLM2 135M Q8"
  "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF|smollm2-135m-instruct-q4_k_m.gguf|86|SmolLM2 135M Q4"
  "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF|smollm2-360m-instruct-f16.gguf|726|SmolLM2 360M F16"
  "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF|smollm2-360m-instruct-q8_0.gguf|386|SmolLM2 360M Q8"
  "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF|smollm2-360m-instruct-q4_k_m.gguf|234|SmolLM2 360M Q4"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF|smollm2-1.7b-instruct-f16.gguf|3420|SmolLM2 1.7B F16"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF|smollm2-1.7b-instruct-q8_0.gguf|1817|SmolLM2 1.7B Q8"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF|smollm2-1.7b-instruct-q4_k_m.gguf|1060|SmolLM2 1.7B Q4"
)

LLAMA_MODELS=(
  "bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-f16.gguf|2480|LLaMA 3.2 1B F16"
  "bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q8_0.gguf|1320|LLaMA 3.2 1B Q8"
  "bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_K_M.gguf|776|LLaMA 3.2 1B Q4"
  "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-f16.gguf|6430|LLaMA 3.2 3B F16"
  "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q8_0.gguf|3420|LLaMA 3.2 3B Q8"
  "bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_K_M.gguf|2020|LLaMA 3.2 3B Q4"
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q8_0.gguf|8540|LLaMA 3.1 8B Q8"
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|4920|LLaMA 3.1 8B Q4"
)
# Note: LLaMA 3.1 8B F16 (~16GB) excluded as it leaves no room for runtime overhead.

# ── Helpers ────────────────────────────────────────────────────────────────

print_table() {
  local -n models_ref=$1
  printf "  %-45s %8s  %s\n" "Filename" "Size" "Description"
  printf "  %-45s %8s  %s\n" "--------" "----" "-----------"
  for entry in "${models_ref[@]}"; do
    IFS='|' read -r repo filename size_mb desc <<< "$entry"
    if (( size_mb >= 1024 )); then
      size="$(awk "BEGIN{printf \"%.1f GB\", $size_mb/1024}")"
    else
      size="${size_mb} MB"
    fi
    local status=""
    if [[ -f "$MODELS_DIR/$filename" ]]; then
      status=" [downloaded]"
    fi
    printf "  %-45s %8s  %s%s\n" "$filename" "$size" "$desc" "$status"
  done
}

download_models() {
  local -n models_ref=$1
  local total=${#models_ref[@]}
  local idx=0
  local skipped=0
  local downloaded=0
  local failed=0

  for entry in "${models_ref[@]}"; do
    IFS='|' read -r repo filename size_mb desc <<< "$entry"
    idx=$((idx + 1))

    if [[ -f "$MODELS_DIR/$filename" ]]; then
      echo "[$idx/$total] Already exists: $filename (skipping)"
      skipped=$((skipped + 1))
      continue
    fi

    echo "[$idx/$total] Downloading $desc ($filename) ..."
    if python3 "$SCRIPT_DIR/download_model.py" "$repo" "$filename"; then
      downloaded=$((downloaded + 1))
    else
      echo "  FAILED to download $filename"
      failed=$((failed + 1))
    fi
  done

  echo ""
  echo "Done: $downloaded downloaded, $skipped skipped, $failed failed"
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-all}" in
  --list|-l)
    echo "SmolLM2 models:"
    print_table SMOL_MODELS
    echo ""
    echo "LLaMA 3 models:"
    print_table LLAMA_MODELS
    echo ""
    echo "All models fit within 16GB RAM. Total if all downloaded: ~35 GB disk."
    ;;
  smol|smolm|smollm|smollm2)
    echo "=== Downloading SmolLM2 models ==="
    download_models SMOL_MODELS
    ;;
  llama|llama3)
    echo "=== Downloading LLaMA 3 models ==="
    download_models LLAMA_MODELS
    ;;
  all)
    echo "=== Downloading SmolLM2 models ==="
    download_models SMOL_MODELS
    echo ""
    echo "=== Downloading LLaMA 3 models ==="
    download_models LLAMA_MODELS
    ;;
  --help|-h)
    echo "Usage: $0 [all|smol|llama|--list|--help]"
    echo ""
    echo "  all     Download all models (default)"
    echo "  smol    Download SmolLM2 models only"
    echo "  llama   Download LLaMA 3 models only"
    echo "  --list  Show available models and sizes"
    ;;
  *)
    echo "Unknown option: $1"
    echo "Usage: $0 [all|smol|llama|--list|--help]"
    exit 1
    ;;
esac
