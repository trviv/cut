#!/usr/bin/env bash
# Install the Linux build dependencies for this project.
#
# Usage:
#   ./scripts/setup_linux.sh
#
# Steps:
#   1. apt: g++, Vulkan loader/headers, shaderc, glslang
#   2. DXC (DirectX Shader Compiler) — HLSL -> SPIR-V, not in apt; fetched from
#      the microsoft/DirectXShaderCompiler GitHub release.
#   3. LunarG Vulkan SDK glslc — the apt shaderc (2023.8 / glslang 14) is too
#      old for GL_EXT_integer_dot_product used by Q8-dot-product shaders.
#
# Installs per-user to ~/.local/{bin,lib}. Requires sudo only for the apt step.
# Idempotent: safe to re-run.

set -euo pipefail

LOCAL_BIN="$HOME/.local/bin"
LOCAL_LIB="$HOME/.local/lib"
mkdir -p "$LOCAL_BIN" "$LOCAL_LIB"

DXC_URL="https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/linux_dxc_2024_07_31.x86_64.tar.gz"
VULKAN_SDK_URL="https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk-linux-x86_64.tar.xz"

echo "==> [1/3] apt packages (needs sudo)"
sudo apt-get update
sudo apt-get install -y \
    g++ \
    cmake \
    libvulkan-dev \
    libshaderc-dev \
    glslang-dev \
    glslang-tools \
    glslc \
    wget \
    python3

echo ""
echo "==> [2/3] DXC (DirectX Shader Compiler)"
if [[ -x "$LOCAL_BIN/dxc" ]] && [[ -f "$LOCAL_LIB/libdxcompiler.so" ]]; then
    echo "    dxc already installed at $LOCAL_BIN/dxc — skipping"
else
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    wget -q "$DXC_URL" -O "$tmpdir/dxc.tar.gz"
    tar -xzf "$tmpdir/dxc.tar.gz" -C "$tmpdir"
    install -m 0755 "$tmpdir/bin/dxc" "$LOCAL_BIN/dxc"
    install -m 0755 "$tmpdir/lib/libdxcompiler.so" "$LOCAL_LIB/libdxcompiler.so"
    install -m 0755 "$tmpdir/lib/libdxil.so" "$LOCAL_LIB/libdxil.so"
    rm -rf "$tmpdir"
    trap - EXIT
    echo "    installed dxc -> $LOCAL_BIN/dxc"
fi

echo ""
echo "==> [3/3] LunarG Vulkan SDK glslc (newer than apt — needed for GL_EXT_integer_dot_product)"
need_sdk_glslc=1
if [[ -x "$LOCAL_BIN/glslc" ]]; then
    ver="$("$LOCAL_BIN/glslc" --version 2>&1 | head -1 || true)"
    if [[ "$ver" == *"v2024"* || "$ver" == *"v2025"* || "$ver" == *"v2026"* ]]; then
        echo "    $LOCAL_BIN/glslc already recent ($ver) — skipping"
        need_sdk_glslc=0
    fi
fi
if [[ "$need_sdk_glslc" == "1" ]]; then
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    wget -q "$VULKAN_SDK_URL" -O "$tmpdir/vulkan-sdk.tar.xz"
    tar -xJf "$tmpdir/vulkan-sdk.tar.xz" -C "$tmpdir"
    sdk_glslc="$(find "$tmpdir" -path '*/x86_64/bin/glslc' -type f | head -1)"
    if [[ -z "$sdk_glslc" ]]; then
        echo "ERROR: could not find glslc in extracted Vulkan SDK" >&2
        exit 1
    fi
    install -m 0755 "$sdk_glslc" "$LOCAL_BIN/glslc"
    rm -rf "$tmpdir"
    trap - EXIT
    echo "    installed glslc -> $LOCAL_BIN/glslc"
fi

echo ""
echo "==> Done."
echo ""
echo "Make sure these are set for every build (add to ~/.bashrc or equivalent):"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "    export LD_LIBRARY_PATH=\"\$HOME/.local/lib:\${LD_LIBRARY_PATH:-}\""
echo ""
echo "Then run:"
echo "    ./scripts/run_cpp.sh"
