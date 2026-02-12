# CUT — Compute Unified Toolkit

A GPU-accelerated tensor compute library built on Vulkan. CUT provides a PyTorch-like Python API backed by runtime-generated GLSL compute shaders compiled to SPIR-V.

## Features

- **Vulkan GPU backend** with runtime shader generation and SPIR-V caching
- **163+ operations** across element-wise math, reductions, linear algebra, activations, and more
- **Pure Python API** — no NumPy dependency (uses native `array.array` for data transfer)
- **Data types**: `float32`, `float16`, `int32`, `uint32`
- **Multi-dimensional tensors** with automatic vec4 alignment for GPU efficiency

## Quick Start

```python
import cut.compute as cc

# Initialize Vulkan backend (auto-initializes on first use)
cc.init(cc.Backend.Vulkan)

# Create tensors
a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
b = cc.Tensor([5.0, 6.0, 7.0, 8.0])

# Element-wise operations
c = cc.add(a, b)           # [6.0, 8.0, 10.0, 12.0]
d = cc.multiply(a, b)      # [5.0, 12.0, 21.0, 32.0]
e = cc.exp(a)              # element-wise exponential

# Operator overloading
f = a + b                  # same as cc.add(a, b)
g = a * b                  # same as cc.multiply(a, b)

# Reductions
s = cc.sum(a)              # 10.0
m = cc.mean(a)             # 2.5

# Matrix operations
A = cc.Tensor([[1, 2], [3, 4]])
B = cc.Tensor([[5, 6], [7, 8]])
C = cc.matmul(A, B)        # [[19, 22], [43, 50]]

# Activations
x = cc.Tensor([-1.0, 0.0, 1.0, 2.0])
y = cc.relu(x)             # [0.0, 0.0, 1.0, 2.0]
z = cc.gelu(x)             # GELU activation

# Dimension-wise reduction
X = cc.Tensor([[1, 2, 3], [4, 5, 6]])
row_sums = cc.sum(X, dim=1)    # [6, 15]
col_sums = cc.sum(X, dim=0)    # [5, 7, 9]

# Get results back to Python
result = c.tolist()        # [6.0, 8.0, 10.0, 12.0]

cc.shutdown()
```

## Supported Data Types

| DType | Python | Description |
|-------|--------|-------------|
| `cc.float32` | `float` | 32-bit floating point (default) |
| `cc.float16` | — | 16-bit floating point |
| `cc.int32` | `int` | 32-bit signed integer |
| `cc.uint32` | — | 32-bit unsigned integer |

---

## Operations Reference

### Binary Arithmetic (vec-vec and vec-scalar)

Each operation has both a tensor-tensor variant and a tensor-scalar variant (suffixed with `_scalar`).

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `add` / `add_scalar` | `torch.add` | Element-wise addition |
| `subtract` / `subtract_scalar` | `torch.sub` | Element-wise subtraction |
| `multiply` / `multiply_scalar` | `torch.mul` | Element-wise multiplication |
| `divide` / `divide_scalar` | `torch.div` | Element-wise division |
| `mod` / `mod_scalar` | `torch.remainder` | Element-wise modulo |
| `power` / `power_scalar` | `torch.pow` | Element-wise power |
| `floor_divide` / `floor_divide_scalar` | `torch.floor_divide` | Floor division |
| `fmod` / `fmod_scalar` | `torch.fmod` | Floating-point remainder |
| `minimum` / `minimum_scalar` | `torch.minimum` | Element-wise minimum |
| `maximum` / `maximum_scalar` | `torch.maximum` | Element-wise maximum |
| `arctan2` / `arctan2_scalar` | `torch.atan2` | Arc tangent of a/b |
| `hypot` / `hypot_scalar` | `torch.hypot` | Hypotenuse: sqrt(a^2 + b^2) |
| `copysign` / `copysign_scalar` | `torch.copysign` | Copy sign of b to a |
| `logaddexp` / `logaddexp_scalar` | `torch.logaddexp` | Numerically stable log(exp(a) + exp(b)) |
| `logaddexp2` / `logaddexp2_scalar` | `torch.logaddexp2` | Base-2 log-sum-exp |

### Comparison Operations (vec-vec and vec-scalar)

Returns `1.0` for True, `0.0` for False.

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `equal` / `equal_scalar` | `torch.eq` | Element-wise equality |
| `not_equal` / `not_equal_scalar` | `torch.ne` | Element-wise inequality |
| `less` / `less_scalar` | `torch.lt` | Less than |
| `less_equal` / `less_equal_scalar` | `torch.le` | Less than or equal |
| `greater` / `greater_scalar` | `torch.gt` | Greater than |
| `greater_equal` / `greater_equal_scalar` | `torch.ge` | Greater than or equal |

### Bitwise Operations (vec-vec and vec-scalar, integers)

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `bitwise_and` / `bitwise_and_scalar` | `torch.bitwise_and` | Bitwise AND |
| `bitwise_or` / `bitwise_or_scalar` | `torch.bitwise_or` | Bitwise OR |
| `bitwise_xor` / `bitwise_xor_scalar` | `torch.bitwise_xor` | Bitwise XOR |
| `left_shift` / `left_shift_scalar` | `torch.bitwise_left_shift` | Left shift |
| `right_shift` / `right_shift_scalar` | `torch.bitwise_right_shift` | Right shift |

### Logical Operations (vec-vec and vec-scalar)

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `logical_and` / `logical_and_scalar` | `torch.logical_and` | Logical AND |
| `logical_or` / `logical_or_scalar` | `torch.logical_or` | Logical OR |
| `logical_xor` / `logical_xor_scalar` | `torch.logical_xor` | Logical XOR |

### Unary Math Operations

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `negative` | `torch.neg` | Negate |
| `abs` | `torch.abs` | Absolute value |
| `sqrt` | `torch.sqrt` | Square root |
| `square` | `torch.square` | Square (x^2) |
| `reciprocal` | `torch.reciprocal` | Reciprocal (1/x) |
| `sign` | `torch.sign` | Sign (-1, 0, +1) |
| `cbrt` | — | Cube root |
| `exp` | `torch.exp` | Natural exponential |
| `exp2` | — | Base-2 exponential |
| `expm1` | `torch.expm1` | exp(x) - 1 (accurate for small x) |
| `log` | `torch.log` | Natural logarithm |
| `log2` | `torch.log2` | Base-2 logarithm |
| `log10` | `torch.log10` | Base-10 logarithm |
| `log1p` | `torch.log1p` | log(1+x) (accurate for small x) |
| `sin` | `torch.sin` | Sine |
| `cos` | `torch.cos` | Cosine |
| `tan` | `torch.tan` | Tangent |
| `arcsin` | `torch.asin` | Inverse sine |
| `arccos` | `torch.acos` | Inverse cosine |
| `arctan` | `torch.atan` | Inverse tangent |
| `sinh` | `torch.sinh` | Hyperbolic sine |
| `cosh` | `torch.cosh` | Hyperbolic cosine |
| `tanh` | `torch.tanh` | Hyperbolic tangent |
| `arcsinh` | `torch.asinh` | Inverse hyperbolic sine |
| `arccosh` | `torch.acosh` | Inverse hyperbolic cosine |
| `arctanh` | `torch.atanh` | Inverse hyperbolic tangent |
| `floor` | `torch.floor` | Floor |
| `ceil` | `torch.ceil` | Ceiling |
| `round` | `torch.round` | Round to nearest integer |
| `trunc` | `torch.trunc` | Truncate to integer part |
| `frac` | `torch.frac` | Fractional part |
| `rsqrt` | `torch.rsqrt` | Reciprocal square root |
| `degrees` | `torch.rad2deg` | Radians to degrees |
| `radians` | `torch.deg2rad` | Degrees to radians |
| `logical_not` | `torch.logical_not` | Logical NOT |
| `invert` | `torch.bitwise_not` | Bitwise NOT (integers) |
| `isnan` | `torch.isnan` | Check if NaN |
| `isinf` | `torch.isinf` | Check if infinite |
| `isfinite` | `torch.isfinite` | Check if finite |

### Activation Functions

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `relu` | `F.relu` | max(0, x) |
| `relu6` | `F.relu6` | clamp(x, 0, 6) |
| `sigmoid` | `torch.sigmoid` | 1/(1+exp(-x)) |
| `gelu` | `F.gelu` | Gaussian Error Linear Unit |
| `silu` | `F.silu` | x * sigmoid(x) / Swish |
| `softplus` | `F.softplus` | log(1 + exp(x)) |
| `leaky_relu` | `F.leaky_relu` | x if x>0 else alpha*x (scalar param) |
| `elu` | `F.elu` | Exponential Linear Unit |
| `selu` | `F.selu` | Scaled ELU |
| `celu` | `F.celu` | Continuously differentiable ELU |
| `mish` | `F.mish` | x * tanh(softplus(x)) |
| `hardswish` | `F.hardswish` | x * clamp(x+3, 0, 6) / 6 |
| `hardsigmoid` | `F.hardsigmoid` | clamp(x/6 + 0.5, 0, 1) |
| `hardtanh` | `F.hardtanh` | clamp(x, -1, 1) |
| `softsign` | `F.softsign` | x / (1 + \|x\|) |
| `logsigmoid` | `F.logsigmoid` | log(sigmoid(x)) |
| `tanhshrink` | `F.tanhshrink` | x - tanh(x) |
| `prelu` | `F.prelu` | Parameterized ReLU (scalar param) |
| `hardshrink` | `F.hardshrink` | Hard shrink (scalar param) |
| `softshrink` | `F.softshrink` | Soft shrink (scalar param) |
| `softmax` | `F.softmax` | Softmax along dimension |
| `log_softmax` | `F.log_softmax` | Log-softmax along dimension |

### Reduction Operations

All reductions support both global (returns scalar) and dimension-wise (returns tensor) via the `dim` parameter.

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `sum(x)` / `sum(x, dim=d)` | `torch.sum` | Sum of elements |
| `mean(x)` / `mean(x, dim=d)` | `torch.mean` | Mean of elements |
| `min(x)` / `min(x, dim=d)` | `torch.min` | Minimum element |
| `max(x)` / `max(x, dim=d)` | `torch.max` | Maximum element |
| `prod(x)` / `prod(x, dim=d)` | `torch.prod` | Product of elements |
| `any(x)` / `any(x, dim=d)` | `torch.any` | True if any non-zero |
| `all(x)` / `all(x, dim=d)` | `torch.all` | True if all non-zero |
| `norm(x, p=2)` / `norm(x, dim=d)` | `torch.norm` | L2 norm (also supports L1, Linf) |
| `var(x)` / `var(x, dim=d)` | `torch.var` | Variance (with Bessel's correction) |
| `std(x)` / `std(x, dim=d)` | `torch.std` | Standard deviation |
| `argmax(x)` / `argmax(x, dim=d)` | `torch.argmax` | Index of maximum element |
| `argmin(x)` / `argmin(x, dim=d)` | `torch.argmin` | Index of minimum element |
| `cumsum(x, dim=d)` | `torch.cumsum` | Cumulative sum along dimension |
| `cumprod(x, dim=d)` | `torch.cumprod` | Cumulative product along dimension |

### Matrix Operations

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `matmul(A, B)` | `torch.matmul` | Matrix multiplication (2D) |
| `transpose(A)` | `torch.transpose` | Matrix transpose (2D) |
| `dot(a, b)` | `torch.dot` | Dot product of vectors |

### Ternary Operations

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `clamp(x, min, max)` | `torch.clamp` | Clamp values to [min, max] |
| `where(cond, x, y)` | `torch.where` | Select x where cond else y |

### Tensor Manipulation

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `concat(tensors, axis=0)` | `torch.cat` | Concatenate along axis |
| `stack(tensors, axis=0)` | `torch.stack` | Stack along new axis |
| `flatten(x, start_dim, end_dim)` | `torch.flatten` | Flatten dimensions |
| `reshape(x, *shape)` | `torch.reshape` | Reshape tensor (zero-copy when possible) |
| `view(x, *shape)` | `Tensor.view` | Alias for reshape |
| `squeeze(x, dim=None)` | `torch.squeeze` | Remove size-1 dimensions |
| `unsqueeze(x, dim)` | `torch.unsqueeze` | Insert size-1 dimension |
| `unflatten(x, dim, sizes)` | `torch.unflatten` | Expand a dimension into multiple |

### Tensor Creation

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `zeros(*shape)` | `torch.zeros` | Tensor of zeros |
| `ones(*shape)` | `torch.ones` | Tensor of ones |
| `full(*shape, fill_value=v)` | `torch.full` | Tensor filled with v |
| `arange(start, end, step)` | `torch.arange` | Evenly spaced values |
| `linspace(start, end, steps)` | `torch.linspace` | Linearly spaced values |
| `zeros_like(t)` | `torch.zeros_like` | Zeros with same shape/dtype |
| `ones_like(t)` | `torch.ones_like` | Ones with same shape/dtype |
| `full_like(t, v)` | `torch.full_like` | Filled with v, same shape/dtype |

### Loss Functions

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `mse_loss(input, target)` | `F.mse_loss` | Mean squared error loss |
| `l1_loss(input, target)` | `F.l1_loss` | Mean absolute error loss |
| `cross_entropy_loss(input, target)` | `F.cross_entropy` | Cross-entropy loss (with softmax) |

---

## PyTorch Compatibility Roadmap

Status: **Supported** | **Planned** | Not planned

### Element-wise Math

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.abs` | Supported | `cc.abs` |
| `torch.neg` | Supported | `cc.negative` |
| `torch.add` / `sub` / `mul` / `div` | Supported | + operator overloading |
| `torch.pow` | Supported | `cc.power` |
| `torch.sqrt` | Supported | `cc.sqrt` |
| `torch.rsqrt` | Supported | `cc.rsqrt` |
| `torch.square` | Supported | `cc.square` |
| `torch.reciprocal` | Supported | `cc.reciprocal` |
| `torch.sign` | Supported | `cc.sign` |
| `torch.floor` / `ceil` / `round` | Supported | |
| `torch.trunc` | Supported | `cc.trunc` |
| `torch.frac` | Supported | `cc.frac` |
| `torch.clamp` | Supported | `cc.clamp` |
| `torch.exp` / `exp2` / `expm1` | Supported | |
| `torch.log` / `log2` / `log10` / `log1p` | Supported | |
| `torch.sin` / `cos` / `tan` | Supported | |
| `torch.asin` / `acos` / `atan` / `atan2` | Supported | |
| `torch.sinh` / `cosh` / `tanh` | Supported | |
| `torch.asinh` / `acosh` / `atanh` | Supported | `cc.arcsinh` / `cc.arccosh` / `cc.arctanh` |
| `torch.erf` / `erfc` | Not planned | special functions |
| `torch.logaddexp` / `logaddexp2` | Supported | `cc.logaddexp` / `cc.logaddexp2` |
| `torch.isnan` / `isinf` | Supported | |
| `torch.isfinite` | Supported | `cc.isfinite` |
| `torch.copysign` / `hypot` / `fmod` | Supported | |

### Activation Functions

| PyTorch | Status | Notes |
|---------|--------|-------|
| `F.relu` | Supported | `cc.relu` |
| `F.relu6` | Supported | `cc.relu6` |
| `F.sigmoid` | Supported | `cc.sigmoid` |
| `F.tanh` | Supported | `cc.tanh` |
| `F.gelu` | Supported | `cc.gelu` |
| `F.silu` / `swish` | Supported | `cc.silu` |
| `F.softplus` | Supported | `cc.softplus` |
| `F.leaky_relu` | Supported | `cc.leaky_relu` |
| `F.elu` | Supported | `cc.elu` |
| `F.selu` | Supported | `cc.selu` |
| `F.celu` | Supported | `cc.celu` |
| `F.mish` | Supported | `cc.mish` |
| `F.hardswish` | Supported | `cc.hardswish` |
| `F.hardsigmoid` | Supported | `cc.hardsigmoid` |
| `F.hardtanh` | Supported | `cc.hardtanh` |
| `F.softsign` | Supported | `cc.softsign` |
| `F.logsigmoid` | Supported | `cc.logsigmoid` |
| `F.tanhshrink` | Supported | `cc.tanhshrink` |
| `F.prelu` | Supported | `cc.prelu` |
| `F.hardshrink` | Supported | `cc.hardshrink` |
| `F.softshrink` | Supported | `cc.softshrink` |
| `F.softmax` | Supported | `cc.softmax` |
| `F.log_softmax` | Supported | `cc.log_softmax` |

### Reductions

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.sum` / `nansum` | Supported / Not planned | |
| `torch.mean` / `nanmean` | Supported / Not planned | |
| `torch.min` / `max` | Supported | global + dim-wise |
| `torch.prod` | Supported | global + dim-wise |
| `torch.any` / `all` | Supported | global + dim-wise |
| `torch.norm` | Supported | L1, L2, Linf |
| `torch.var` / `std` | Supported | `cc.var` / `cc.std` (composed) |
| `torch.argmax` / `argmin` | Supported | `cc.argmax` / `cc.argmin` (global + dim) |
| `torch.median` | Not planned | requires sorting |
| `torch.logsumexp` | Not planned | |
| `torch.cumsum` / `cumprod` | Supported | `cc.cumsum` / `cc.cumprod` |
| `torch.count_nonzero` | Not planned | |
| `torch.sort` / `argsort` | Not planned | GPU sort is complex |
| `torch.topk` | Not planned | |

### Matrix / Linear Algebra

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.matmul` / `mm` | Supported | 2D matrix multiply |
| `torch.dot` | Supported | vector dot product |
| `torch.transpose` | Supported | 2D transpose |
| `torch.bmm` | Not planned | batched matmul |
| `torch.mv` | Not planned | matrix-vector |
| `torch.inner` / `outer` | Not planned | |
| `torch.cross` | Not planned | |
| `torch.det` / `inverse` | Not planned | decomposition-based |
| `torch.svd` / `qr` / `lu` | Not planned | |

### Shape / Indexing

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.reshape` / `view` | Supported | `cc.reshape` / `cc.view` (zero-copy) |
| `torch.squeeze` / `unsqueeze` | Supported | `cc.squeeze` / `cc.unsqueeze` (zero-copy) |
| `torch.flatten` | Supported | `cc.flatten` |
| `torch.cat` / `stack` | Supported | `cc.concat` / `cc.stack` |
| `torch.permute` | Not planned | generalized transpose |
| `torch.flip` / `roll` | Not planned | |
| `torch.gather` / `scatter` | Not planned | advanced indexing |
| `torch.index_select` | Not planned | |
| `torch.where` | Supported | `cc.where` |

### Tensor Creation

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.zeros` / `ones` / `full` | Supported | |
| `torch.arange` / `linspace` | Supported | |
| `torch.zeros_like` / `ones_like` / `full_like` | Supported | `cc.zeros_like` / `cc.ones_like` / `cc.full_like` |
| `torch.empty` / `empty_like` | Not planned | (use `zeros` / `zeros_like`) |
| `torch.eye` | Not planned | |
| `torch.logspace` | Not planned | (compose from `linspace` + `power`) |
| `torch.rand` / `randn` / `randint` | Not planned | requires GPU PRNG |

### Loss Functions

| PyTorch | Status | Notes |
|---------|--------|-------|
| `F.mse_loss` | Supported | `cc.mse_loss` (composed) |
| `F.l1_loss` | Supported | `cc.l1_loss` (composed) |
| `F.smooth_l1_loss` / `huber_loss` | Not planned | |
| `F.cross_entropy` | Supported | `cc.cross_entropy_loss` (composed) |
| `F.binary_cross_entropy` | Not planned | |
| `F.nll_loss` | Not planned | requires gather |

### Neural Network Layers

| PyTorch | Status | Notes |
|---------|--------|-------|
| `nn.Conv1d/2d/3d` | Not planned | complex spatial ops |
| `nn.MaxPool/AvgPool` | Not planned | |
| `nn.BatchNorm` / `LayerNorm` / `RMSNorm` | Not planned | |
| `nn.Dropout` | Not planned | requires GPU PRNG |
| `nn.Embedding` | Not planned | |
| `nn.Linear` | Not planned | (use matmul + add) |
| `nn.MultiheadAttention` | Not planned | (compose from matmul + softmax) |

---

## Architecture

```
Python API  (cut/compute.py, cut/_ops.py)
     |
pybind11    (cut/_compute_bindings.cpp)
     |
Runtime     (Dispatcher → ComputeInterface)
     |
Vulkan      (VulkanCompute → GLSL→SPIR-V shaders)
```

- **Shader generation**: Operations are defined as GLSL expressions, assembled into compute shader templates, and compiled to SPIR-V at runtime
- **Caching**: Compiled shaders are cached by (operator, dtype) key
- **Vectorization**: All element-wise ops use vec4 SIMD with 256-thread workgroups
- **Alignment**: Innermost tensor dimension is padded to multiples of 4 for GPU efficiency

## Building

```bash
mkdir build && cd build
cmake ..
make -j8
```

Requires:
- CMake 3.16+
- C++17 compiler
- Vulkan SDK

## Testing

```bash
# C++ tests
cd build && ctest --verbose

# Python tests
python -m pytest interface/python/tests/test_compute.py -v
```
