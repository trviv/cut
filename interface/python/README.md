# CUT Python Interface & GPU Benchmarks

Python bindings for the CUT (Compute Unified Toolkit) Vulkan GPU compute library, with a
multi-backend benchmark suite comparing against PyTorch CUDA, CuPy, JAX,
NVIDIA Warp, and NumPy.

## Prerequisites

- Python 3.8+
- Vulkan SDK (vulkan-dev, shaderc, glslang, SPIRV-Tools)
- CMake 3.15+
- C++17 compiler
- (Optional) NVIDIA GPU + driver for CUDA backends

### System packages (Ubuntu/Debian)

```bash
sudo apt install python3-dev python3-venv cmake \
    libvulkan-dev libshaderc-dev glslang-dev spirv-tools
```

## Setup

All Python work uses the project-local venv at `cut/.venv`.

### 1. Create venv and install pip

```bash
cd /path/to/cut
python3 -m venv .venv --without-pip
wget -qO- https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
```

### 2. Install base dependencies

```bash
.venv/bin/python -m pip install numpy pybind11 scikit-build-core pytest
```

### 3. Build the CUT Python extension

**Option A: pip editable install** (requires python3-dev headers)

```bash
.venv/bin/python -m pip install --no-build-isolation -e interface/python
```

**Option B: Manual cmake build** (if python3-dev is not installable)

```bash
mkdir -p build_python && cd build_python

cmake ../interface/python \
    -DPython_ROOT_DIR="/usr" \
    -DPython_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR=$(.venv/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# Copy the built module into the package
cp _cut_compute.cpython-*.so ../interface/python/cut/
cd ..
```

If your Python dev headers are in a non-standard location (e.g. extracted
manually), add:

```bash
-DCMAKE_CXX_FLAGS="-I/path/to/python/include"
```

### 4. Verify CUT loads

```bash
PYTHONPATH=interface/python .venv/bin/python -c \
    "import cut.compute as cut; print('Vulkan:', cut.is_vulkan_available())"
```

## Installing GPU Libraries for Benchmarks

### PyTorch (CUDA)

```bash
# With NVIDIA GPU (CUDA 12.6 wheels)
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU-only fallback (no NVIDIA GPU needed)
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### CuPy (CUDA)

```bash
.venv/bin/python -m pip install cupy-cuda12x
```

### JAX (GPU)

```bash
.venv/bin/python -m pip install "jax[cuda12]"
```

### NVIDIA Warp

```bash
.venv/bin/python -m pip install warp-lang
```

### Install everything at once (NVIDIA GPU system)

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install cupy-cuda12x "jax[cuda12]" warp-lang
```

### Install everything at once (CPU-only system)

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install jax
```

## Running Benchmarks

All commands below assume you are in the **project root** (`cut/`) with the
venv activated:

```bash
cd /path/to/cut
source .venv/bin/activate
```

`PYTHONPATH` must point to `interface/python` (the directory containing the
`cut/` and `common/` packages). **It is a directory path, not a python binary.**

### Multi-library comparative benchmark

This is the main benchmark. It auto-detects every installed library and GPU.

```bash
# Default: 1M elements, 10 iterations, all detected backends
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py

# 10M elements for more accurate GPU timing
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py -n 10000000

# GPU backends only (skip PyTorch CPU, keep NumPy as baseline)
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py --gpu-only

# Quick sanity check
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py -n 100000 -i 5 -w 2

# Export results to file
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py --json results.json --csv results.csv

# Pipe-friendly (no ANSI color codes)
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py --no-color > results.txt 2>&1

# Summary only (skip per-op table)
cd interface/python/benchmarks && PYTHONPATH=.. python run_benchmarks.py -q
```

Alternatively, run from the project root without cd:

```bash
PYTHONPATH=interface/python python interface/python/benchmarks/run_benchmarks.py -n 1000000
```

### CUT vs NumPy benchmark (simple)

```bash
cd interface/python/benchmarks && PYTHONPATH=.. python benchmark.py --backend vulkan
```

### Chained operations benchmark

Tests multi-op pipelines (sigmoid, quadratic formula, distance, etc.):

```bash
cd interface/python/benchmarks && PYTHONPATH=.. python benchmark_chained_ops.py --backend vulkan --size 1000000
cd interface/python/benchmarks && PYTHONPATH=.. python benchmark_chained_ops.py --include-jax
```

## Benchmark CLI Reference

### `run_benchmarks.py`

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --num-elements` | 1000000 | Elements per array |
| `-i, --iterations` | 10 | Timed iterations per operation |
| `-w, --warmup` | 3 | Warmup iterations (not timed) |
| `-s, --seed` | 42 | Random seed for reproducibility |
| `--gpu-only` | off | Only benchmark GPU backends |
| `--json FILE` | — | Export results to JSON |
| `--csv FILE` | — | Export results to CSV |
| `--no-color` | off | Disable ANSI colors |
| `-q, --quiet` | off | Print summary only |

### Backends auto-detected

| Backend | pip package | Device | Notes |
|---------|-------------|--------|-------|
| CUT Vulkan | built from source | GPU | Vulkan compute shaders |
| PyTorch CUDA | `torch` (cu126) | GPU | Requires NVIDIA GPU + driver |
| PyTorch CPU | `torch` | CPU | Always available with torch |
| CuPy | `cupy-cuda12x` | GPU | Requires NVIDIA GPU |
| JAX | `jax[cuda12]` | GPU/CPU | Uses GPU when available |
| Warp | `warp-lang` | GPU | Kernel framework (limited tensor ops) |
| NumPy | `numpy` | CPU | Always present, used as baseline |

## Operations Benchmarked

107 operations across 16 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Binary arithmetic | 7 | add, multiply, mod, floor_divide |
| Binary math | 6 | arctan2, hypot, copysign, logaddexp |
| Comparisons | 6 | equal, less, greater_equal |
| Min/Max | 2 | minimum, maximum |
| Unary ops | 22 | sqrt, exp, log, sin, cos, tanh |
| Extended math | 12 | expm1, log1p, rsqrt, arcsinh |
| Activations | 14 | relu, sigmoid, gelu, silu, elu, mish |
| Vec-scalar | 16 | add_scalar, multiply_scalar, etc. |
| Reductions | 5 | sum, mean, min, max, prod |
| Statistics | 2 | var, std |
| Cumulative | 2 | cumsum, cumprod |
| Matrix ops | 2 | matmul, transpose |
| Normalization | 2 | softmax, log_softmax |
| Tensor creation | 5 | zeros, ones, full, arange, linspace |
| Conditional/norm | 3 | where, norm |
| Loss functions | 2 | mse_loss, l1_loss |

## Usage

### High-Level API

```python
import cut.compute as cut

cut.init(cut.Backend.Vulkan)

a = cut.Tensor([1.0, 2.0, 3.0, 4.0])
b = cut.Tensor([5.0, 6.0, 7.0, 8.0])

print(cut.add(a, b).tolist())      # [6.0, 8.0, 10.0, 12.0]
print(cut.sigmoid(a).tolist())     # [0.731, 0.881, 0.953, 0.982]
print(cut.sum(a))                  # 10.0

m1 = cut.Tensor([[1, 2], [3, 4]])
m2 = cut.Tensor([[5, 6], [7, 8]])
print(cut.matmul(m1, m2).tolist()) # [[19, 22], [43, 50]]

cut.shutdown()
```

## Running Tests

```bash
# From project root, with venv activated
PYTHONPATH=interface/python python -m pytest interface/python/tests/ -v
```
