#!/usr/bin/env bash
# Setup all benchmark runners for comparing against CUT.
#
# Prerequisites:
#   - Vulkan SDK headers (libvulkan-dev)
#   - libshaderc-dev
#   - Python venv at .venv/
#
# Usage:
#   ./scripts/setup_benchmark_runners.sh
#
# After setup, run benchmarks with:
#   .venv/bin/python scripts/benchmark_compare.py models/SmolLM2-135M-Instruct-f16.gguf

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "============================================================"
echo "Setting up benchmark runners"
echo "============================================================"
echo ""

# ----------------------------------------------------------------
# 1. Build CUT (this project)
# ----------------------------------------------------------------
echo "--- 1. Building CUT ---"
if [ ! -d build ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Debug
fi
cmake --build build --config Debug --target gguf_example -j"$(nproc)"
echo ""

# ----------------------------------------------------------------
# 2. Clone and build llama.cpp
# ----------------------------------------------------------------
LLAMA_DIR="/tmp/llama_cpp_vulkan"

echo "--- 2. Cloning llama.cpp ---"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
else
    echo "  Already cloned at $LLAMA_DIR"
fi
echo ""

# ----------------------------------------------------------------
# 2a. Build llama.cpp CPU (always works)
# ----------------------------------------------------------------
echo "--- 2a. Building llama.cpp CPU ---"
cd "$LLAMA_DIR"
cmake -B build_cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu --config Release -j"$(nproc)" --target llama-bench
echo "  Built: $LLAMA_DIR/build_cpu/bin/llama-bench"
cd "$ROOT"
echo ""

# ----------------------------------------------------------------
# 2b. Build llama.cpp Vulkan GPU
# ----------------------------------------------------------------
echo "--- 2b. Building llama.cpp Vulkan GPU ---"

# glslc is required but often not installed. Try to find or extract it.
GLSLC=""
if command -v glslc &>/dev/null; then
    GLSLC=$(command -v glslc)
elif [ -f /tmp/glslc_pkg/usr/bin/glslc ]; then
    GLSLC="/tmp/glslc_pkg/usr/bin/glslc"
else
    echo "  glslc not found. Attempting to extract from apt package..."
    if apt download glslc 2>/dev/null; then
        mkdir -p /tmp/glslc_pkg
        dpkg-deb -x glslc*.deb /tmp/glslc_pkg
        rm -f glslc*.deb
        if [ -f /tmp/glslc_pkg/usr/bin/glslc ]; then
            GLSLC="/tmp/glslc_pkg/usr/bin/glslc"
            echo "  Extracted glslc to $GLSLC"
        fi
    fi
fi

if [ -n "$GLSLC" ]; then
    echo "  Using glslc: $GLSLC"
    cd "$LLAMA_DIR"
    cmake -B build_vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
        -DVulkan_GLSLC_EXECUTABLE="$GLSLC"
    cmake --build build_vk --config Release -j"$(nproc)" --target llama-bench
    echo "  Built: $LLAMA_DIR/build_vk/bin/llama-bench"
    cd "$ROOT"
else
    echo "  SKIPPED: could not find or install glslc"
    echo "  Install with: sudo apt install glslc"
fi
echo ""

# ----------------------------------------------------------------
# 3. Install Python packages
# ----------------------------------------------------------------
echo "--- 3. Installing Python packages ---"
if [ ! -d .venv ]; then
    echo "  ERROR: .venv not found. Create it first with: python -m venv .venv"
    exit 1
fi

.venv/bin/pip install --quiet llama-cpp-python transformers torch huggingface_hub
echo "  Installed: llama-cpp-python, transformers, torch, huggingface_hub"
echo ""

# ----------------------------------------------------------------
# 4. Download a test model
# ----------------------------------------------------------------
echo "--- 4. Downloading test model ---"
MODEL="models/SmolLM2-135M-Instruct-f16.gguf"
if [ -f "$MODEL" ]; then
    echo "  Already exists: $MODEL"
else
    .venv/bin/python scripts/download_model.py \
        bartowski/SmolLM2-135M-Instruct-GGUF \
        SmolLM2-135M-Instruct-f16.gguf \
        -o SmolLM2-135M-Instruct-f16.gguf
fi
echo ""

# ----------------------------------------------------------------
# Done
# ----------------------------------------------------------------
echo "============================================================"
echo "Setup complete. Available runners:"
echo "============================================================"
echo ""
echo "  CUT (Vulkan GPU):"
echo "    ./build/interface/loader/gguf/gguf_example $MODEL"
echo ""
echo "  llama.cpp CPU benchmark:"
echo "    $LLAMA_DIR/build_cpu/bin/llama-bench -m $MODEL -p 15 -n 32 -r 1"
echo ""
if [ -f "$LLAMA_DIR/build_vk/bin/llama-bench" ]; then
    echo "  llama.cpp Vulkan GPU benchmark (default device):"
    echo "    $LLAMA_DIR/build_vk/bin/llama-bench -m $MODEL -p 15 -n 32 -r 1 -ngl 99"
    echo ""
    echo "  llama.cpp Vulkan GPU benchmark (specific device, e.g. RTX 3090 = device 1):"
    echo "    GGML_VK_DEVICE=1 $LLAMA_DIR/build_vk/bin/llama-bench -m $MODEL -p 15 -n 32 -r 1 -ngl 99"
    echo ""
fi
echo "  Full comparison script:"
echo "    .venv/bin/python scripts/benchmark_compare.py $MODEL"
