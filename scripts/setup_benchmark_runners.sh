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
LLAMA_DIR="build/external_runners/llama.cpp"

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
elif [ -f build/external_runners/glslc_pkg/usr/bin/glslc ]; then
    GLSLC="build/external_runners/glslc_pkg/usr/bin/glslc"
else
    echo "  glslc not found. Attempting to extract from apt package..."
    mkdir -p build/external_runners/glslc_pkg
    pushd build/external_runners/glslc_pkg > /dev/null
    if apt download glslc 2>/dev/null; then
        dpkg-deb -x glslc*.deb .
        rm -f glslc*.deb
    fi
    popd > /dev/null
    if [ -f build/external_runners/glslc_pkg/usr/bin/glslc ]; then
        GLSLC="build/external_runners/glslc_pkg/usr/bin/glslc"
        echo "  Extracted glslc to $GLSLC"
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
# 2c. Build llama.cpp CUDA GPU
# ----------------------------------------------------------------
echo "--- 2c. Building llama.cpp CUDA GPU ---"

# Try to find nvcc
NVCC=""
if command -v nvcc &>/dev/null; then
    NVCC=$(command -v nvcc)
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    NVCC="/usr/local/cuda/bin/nvcc"
elif ls /usr/local/cuda-*/bin/nvcc 1>/dev/null 2>&1; then
    NVCC=$(ls /usr/local/cuda-*/bin/nvcc | head -n 1)
fi

if [ -n "$NVCC" ]; then
    echo "  Using nvcc: $NVCC"
    cd "$LLAMA_DIR"
    if cmake -B build_cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$NVCC" && \
       cmake --build build_cuda --config Release -j"$(nproc)" --target llama-bench; then
        echo "  Built: $LLAMA_DIR/build_cuda/bin/llama-bench"
    else
        echo "  CUDA build failed (llama.cpp CUDA benchmarks will be skipped)"
    fi
    cd "$ROOT"
else
    echo "  SKIPPED: nvcc (CUDA toolkit) not found."
    echo "  llama.cpp's CUDA backend requires the full CUDA toolkit (nvcc)."
    echo "  Install it (e.g. 'sudo apt install cuda-toolkit') then re-run this script."
    echo "  Note: CUT's own CUDA backend uses NVRTC and does NOT need nvcc."
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
echo "    ./build/interface/runner/llama/gguf_example $MODEL"
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
if [ -f "$LLAMA_DIR/build_cuda/bin/llama-bench" ]; then
    echo "  llama.cpp CUDA GPU benchmark (device 0):"
    echo "    CUDA_VISIBLE_DEVICES=0 $LLAMA_DIR/build_cuda/bin/llama-bench -m $MODEL -p 15 -n 32 -r 1 -ngl 99"
    echo ""
fi
echo "  Full comparison script (CPU + GPU + python runners):"
echo "    .venv/bin/python scripts/benchmark_compare.py $MODEL"
echo ""
echo "  Per-GPU CUT vs llama.cpp matrix (CUDA+Vulkan on NVIDIA, Vulkan on AMD):"
echo "    python scripts/benchmark_gpus.py $MODEL"
