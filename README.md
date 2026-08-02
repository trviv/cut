# CUT — Compute Unified Toolkit

A Vulkan-based GPU compute toolkit: cross-vendor, from-scratch shader pipeline, LLM-inference and general tensor ops on top. PyTorch-like Python API backed by precompiled SPIR-V kernels.

## Features

- **Vulkan GPU backend** with runtime shader generation and SPIR-V caching
- **163+ operations** across element-wise math, reductions, linear algebra, activations, and more
- **Pure Python API** — no NumPy dependency (uses native `array.array` for data transfer)
- **Data types**: `float32`, `float16`, `int32`, `uint32`
- **Multi-dimensional tensors** with automatic vec4 alignment for GPU efficiency

## Performance

<!-- BENCH:BADGES -->
[![sort_radix_1sweep vs CUB](https://img.shields.io/badge/sort__radix__1sweep_vs_CUB-0.95x-3fb950)](benchmarks/)
[![sort_radix_1pass vs CUB](https://img.shields.io/badge/sort__radix__1pass_vs_CUB-0.95x-3fb950)](benchmarks/)
[![scan vs CUB](https://img.shields.io/badge/scan_vs_CUB-0.95x-d29922)](benchmarks/)
[![softmax vs cuDNN](https://img.shields.io/badge/softmax_vs_cuDNN-0.80x-d29922)](benchmarks/)
[![sgemv vs cuBLAS](https://img.shields.io/badge/sgemv_vs_cuBLAS-0.54x-d29922)](benchmarks/)
[![sgemm vs cuBLAS](https://img.shields.io/badge/sgemm_vs_cuBLAS-0.53x-d29922)](benchmarks/)
[![conv2d vs cuDNN](https://img.shields.io/badge/conv2d_vs_cuDNN-0.52x-d29922)](benchmarks/)
<!-- /BENCH:BADGES -->

**Per-shape tables, history and methodology: [benchmarks/](benchmarks/)**

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

<details>
<summary><b>Binary Arithmetic (vec-vec and vec-scalar)</b> — 15 ops</summary>

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

</details>

<details>
<summary><b>Comparison Operations (vec-vec and vec-scalar)</b> — 6 ops</summary>

Returns `1.0` for True, `0.0` for False.

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `equal` / `equal_scalar` | `torch.eq` | Element-wise equality |
| `not_equal` / `not_equal_scalar` | `torch.ne` | Element-wise inequality |
| `less` / `less_scalar` | `torch.lt` | Less than |
| `less_equal` / `less_equal_scalar` | `torch.le` | Less than or equal |
| `greater` / `greater_scalar` | `torch.gt` | Greater than |
| `greater_equal` / `greater_equal_scalar` | `torch.ge` | Greater than or equal |

</details>

<details>
<summary><b>Bitwise Operations (vec-vec and vec-scalar, integers)</b> — 5 ops</summary>

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `bitwise_and` / `bitwise_and_scalar` | `torch.bitwise_and` | Bitwise AND |
| `bitwise_or` / `bitwise_or_scalar` | `torch.bitwise_or` | Bitwise OR |
| `bitwise_xor` / `bitwise_xor_scalar` | `torch.bitwise_xor` | Bitwise XOR |
| `left_shift` / `left_shift_scalar` | `torch.bitwise_left_shift` | Left shift |
| `right_shift` / `right_shift_scalar` | `torch.bitwise_right_shift` | Right shift |

</details>

<details>
<summary><b>Logical Operations (vec-vec and vec-scalar)</b> — 3 ops</summary>

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `logical_and` / `logical_and_scalar` | `torch.logical_and` | Logical AND |
| `logical_or` / `logical_or_scalar` | `torch.logical_or` | Logical OR |
| `logical_xor` / `logical_xor_scalar` | `torch.logical_xor` | Logical XOR |

</details>

<details>
<summary><b>Unary Math Operations</b> — 39 ops</summary>

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

</details>

<details>
<summary><b>Activation Functions</b> — 22 ops</summary>

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

</details>

<details>
<summary><b>Reduction Operations</b> — 14 ops</summary>

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

</details>

<details>
<summary><b>Matrix Operations</b> — 3 ops</summary>

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `matmul(A, B)` | `torch.matmul` | Matrix multiplication (2D) |
| `transpose(A)` | `torch.transpose` | Matrix transpose (2D) |
| `dot(a, b)` | `torch.dot` | Dot product of vectors |

</details>

<details>
<summary><b>Ternary Operations</b> — 2 ops</summary>

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `clamp(x, min, max)` | `torch.clamp` | Clamp values to [min, max] |
| `where(cond, x, y)` | `torch.where` | Select x where cond else y |

</details>

<details>
<summary><b>Tensor Manipulation</b> — 8 ops</summary>

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

</details>

<details>
<summary><b>Tensor Creation</b> — 8 ops</summary>

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

</details>

<details>
<summary><b>Loss Functions</b> — 3 ops</summary>

| CUT | PyTorch | Description |
|-----|---------|-------------|
| `mse_loss(input, target)` | `F.mse_loss` | Mean squared error loss |
| `l1_loss(input, target)` | `F.l1_loss` | Mean absolute error loss |
| `cross_entropy_loss(input, target)` | `F.cross_entropy` | Cross-entropy loss (with softmax) |

</details>

---

## PyTorch Compatibility Roadmap

Status: **Supported** | Not implemented

<details>
<summary><b>Pointwise Ops</b> — 51/80 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.abs` | **Supported** | `cc.abs` |
| `torch.neg` | **Supported** | `cc.negative` |
| `torch.add` | **Supported** | `cc.add` / `cc.add_scalar` + operator overloading |
| `torch.sub` | **Supported** | `cc.subtract` / `cc.subtract_scalar` + operator overloading |
| `torch.mul` | **Supported** | `cc.multiply` / `cc.multiply_scalar` + operator overloading |
| `torch.div` | **Supported** | `cc.divide` / `cc.divide_scalar` + operator overloading |
| `torch.remainder` | **Supported** | `cc.mod` / `cc.mod_scalar` |
| `torch.pow` | **Supported** | `cc.power` / `cc.power_scalar` |
| `torch.floor_divide` | **Supported** | `cc.floor_divide` / `cc.floor_divide_scalar` |
| `torch.fmod` | **Supported** | `cc.fmod` / `cc.fmod_scalar` |
| `torch.sqrt` | **Supported** | `cc.sqrt` |
| `torch.rsqrt` | **Supported** | `cc.rsqrt` |
| `torch.square` | **Supported** | `cc.square` |
| `torch.reciprocal` | **Supported** | `cc.reciprocal` |
| `torch.sign` | **Supported** | `cc.sign` |
| `torch.floor` | **Supported** | `cc.floor` |
| `torch.ceil` | **Supported** | `cc.ceil` |
| `torch.round` | **Supported** | `cc.round` |
| `torch.trunc` | **Supported** | `cc.trunc` |
| `torch.frac` | **Supported** | `cc.frac` |
| `torch.clamp` | **Supported** | `cc.clamp` |
| `torch.exp` | **Supported** | `cc.exp` |
| `torch.exp2` | **Supported** | `cc.exp2` |
| `torch.expm1` | **Supported** | `cc.expm1` |
| `torch.log` | **Supported** | `cc.log` |
| `torch.log2` | **Supported** | `cc.log2` |
| `torch.log10` | **Supported** | `cc.log10` |
| `torch.log1p` | **Supported** | `cc.log1p` |
| `torch.sin` | **Supported** | `cc.sin` |
| `torch.cos` | **Supported** | `cc.cos` |
| `torch.tan` | **Supported** | `cc.tan` |
| `torch.asin` | **Supported** | `cc.arcsin` |
| `torch.acos` | **Supported** | `cc.arccos` |
| `torch.atan` | **Supported** | `cc.arctan` |
| `torch.atan2` | **Supported** | `cc.arctan2` / `cc.arctan2_scalar` |
| `torch.sinh` | **Supported** | `cc.sinh` |
| `torch.cosh` | **Supported** | `cc.cosh` |
| `torch.tanh` | **Supported** | `cc.tanh` |
| `torch.asinh` | **Supported** | `cc.arcsinh` |
| `torch.acosh` | **Supported** | `cc.arccosh` |
| `torch.atanh` | **Supported** | `cc.arctanh` |
| `torch.logaddexp` | **Supported** | `cc.logaddexp` / `cc.logaddexp_scalar` |
| `torch.logaddexp2` | **Supported** | `cc.logaddexp2` / `cc.logaddexp2_scalar` |
| `torch.isnan` | **Supported** | `cc.isnan` |
| `torch.isinf` | **Supported** | `cc.isinf` |
| `torch.isfinite` | **Supported** | `cc.isfinite` |
| `torch.copysign` | **Supported** | `cc.copysign` / `cc.copysign_scalar` |
| `torch.hypot` | **Supported** | `cc.hypot` / `cc.hypot_scalar` |
| `torch.rad2deg` | **Supported** | `cc.degrees` |
| `torch.deg2rad` | **Supported** | `cc.radians` |
| `torch.cbrt` | **Supported** | `cc.cbrt` (no PyTorch equivalent) |
| `torch.addcdiv` | Not implemented | compound op |
| `torch.addcmul` | Not implemented | compound op |
| `torch.angle` | Not implemented | complex number op |
| `torch.conj_physical` | Not implemented | complex number op |
| `torch.digamma` | Not implemented | special function |
| `torch.erf` | Not implemented | special function |
| `torch.erfc` | Not implemented | special function |
| `torch.erfinv` | Not implemented | special function |
| `torch.float_power` | Not implemented | use `cc.power` with float dtype |
| `torch.frexp` | Not implemented | decomposition op |
| `torch.gradient` | Not implemented | numerical differentiation |
| `torch.heaviside` | Not implemented | compose from `cc.where` |
| `torch.i0` | Not implemented | Bessel function |
| `torch.igamma` / `igammac` | Not implemented | special functions |
| `torch.ldexp` | Not implemented | compose from `cc.power` + `cc.multiply` |
| `torch.lerp` | Not implemented | compose from add + mul |
| `torch.lgamma` | Not implemented | special function |
| `torch.logit` | Not implemented | compose from log + div |
| `torch.mvlgamma` | Not implemented | special function |
| `torch.nan_to_num` | Not implemented | compose from `cc.where` + `cc.isnan` |
| `torch.nextafter` | Not implemented | low-level float op |
| `torch.polygamma` | Not implemented | special function |
| `torch.positive` | Not implemented | identity op |
| `torch.sgn` | Not implemented | complex sign; use `cc.sign` for real |
| `torch.signbit` | Not implemented | compose from `cc.less_scalar(x, 0)` |
| `torch.sinc` | Not implemented | compose from sin + div |
| `torch.xlogy` | Not implemented | compose from mul + log + where |
| `torch.complex` / `torch.polar` | Not implemented | complex number ops |
| `torch.real` / `torch.imag` | Not implemented | complex number ops |

</details>

<details>
<summary><b>Comparison Ops</b> — 8/21 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.eq` | **Supported** | `cc.equal` / `cc.equal_scalar` |
| `torch.ne` | **Supported** | `cc.not_equal` / `cc.not_equal_scalar` |
| `torch.lt` | **Supported** | `cc.less` / `cc.less_scalar` |
| `torch.le` | **Supported** | `cc.less_equal` / `cc.less_equal_scalar` |
| `torch.gt` | **Supported** | `cc.greater` / `cc.greater_scalar` |
| `torch.ge` | **Supported** | `cc.greater_equal` / `cc.greater_equal_scalar` |
| `torch.minimum` | **Supported** | `cc.minimum` / `cc.minimum_scalar` |
| `torch.maximum` | **Supported** | `cc.maximum` / `cc.maximum_scalar` |
| `torch.allclose` | Not implemented | compose from abs + sub + all |
| `torch.argsort` | Not implemented | requires GPU sort |
| `torch.isclose` | Not implemented | compose from abs + sub |
| `torch.isin` | Not implemented | set membership |
| `torch.isposinf` | Not implemented | compose from isinf + greater |
| `torch.isneginf` | Not implemented | compose from isinf + less |
| `torch.isreal` | Not implemented | complex number op |
| `torch.kthvalue` | Not implemented | requires partial sort |
| `torch.fmax` | Not implemented | NaN-ignoring max |
| `torch.fmin` | Not implemented | NaN-ignoring min |
| `torch.sort` | Not implemented | GPU sort is complex |
| `torch.topk` | Not implemented | requires partial sort |
| `torch.msort` | Not implemented | GPU sort is complex |

</details>

<details>
<summary><b>Bitwise Ops</b> — 6/6 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.bitwise_and` | **Supported** | `cc.bitwise_and` / `cc.bitwise_and_scalar` |
| `torch.bitwise_or` | **Supported** | `cc.bitwise_or` / `cc.bitwise_or_scalar` |
| `torch.bitwise_xor` | **Supported** | `cc.bitwise_xor` / `cc.bitwise_xor_scalar` |
| `torch.bitwise_not` | **Supported** | `cc.invert` |
| `torch.bitwise_left_shift` | **Supported** | `cc.left_shift` / `cc.left_shift_scalar` |
| `torch.bitwise_right_shift` | **Supported** | `cc.right_shift` / `cc.right_shift_scalar` |

</details>

<details>
<summary><b>Logical Ops</b> — 4/4 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.logical_and` | **Supported** | `cc.logical_and` / `cc.logical_and_scalar` |
| `torch.logical_or` | **Supported** | `cc.logical_or` / `cc.logical_or_scalar` |
| `torch.logical_xor` | **Supported** | `cc.logical_xor` / `cc.logical_xor_scalar` |
| `torch.logical_not` | **Supported** | `cc.logical_not` |

</details>

<details>
<summary><b>Reduction Ops</b> — 14/28 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.sum` | **Supported** | global + dim-wise |
| `torch.mean` | **Supported** | global + dim-wise |
| `torch.min` | **Supported** | global + dim-wise |
| `torch.max` | **Supported** | global + dim-wise |
| `torch.prod` | **Supported** | global + dim-wise |
| `torch.any` | **Supported** | global + dim-wise |
| `torch.all` | **Supported** | global + dim-wise |
| `torch.norm` | **Supported** | L1, L2, Linf; global + dim-wise |
| `torch.var` | **Supported** | `cc.var` (composed) |
| `torch.std` | **Supported** | `cc.std` (composed) |
| `torch.argmax` | **Supported** | global + dim-wise |
| `torch.argmin` | **Supported** | global + dim-wise |
| `torch.cumsum` | **Supported** | `cc.cumsum` |
| `torch.cumprod` | **Supported** | `cc.cumprod` |
| `torch.amax` / `amin` | Not implemented | use `cc.max` / `cc.min` with dim |
| `torch.aminmax` | Not implemented | compose from min + max |
| `torch.dist` | Not implemented | compose from sub + norm |
| `torch.logsumexp` | Not implemented | compose from max + exp + sum + log |
| `torch.nanmean` | Not implemented | NaN-handling variant |
| `torch.nansum` | Not implemented | NaN-handling variant |
| `torch.nanmedian` | Not implemented | requires sorting + NaN handling |
| `torch.median` | Not implemented | requires sorting |
| `torch.mode` | Not implemented | requires sorting |
| `torch.quantile` / `nanquantile` | Not implemented | requires sorting |
| `torch.std_mean` | Not implemented | compose from `cc.std` + `cc.mean` |
| `torch.var_mean` | Not implemented | compose from `cc.var` + `cc.mean` |
| `torch.unique` / `unique_consecutive` | Not implemented | requires sorting |
| `torch.count_nonzero` | Not implemented | compose from ne_scalar + sum |

</details>

<details>
<summary><b>Matrix / Linear Algebra</b> — 3/33 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.matmul` / `mm` | **Supported** | 2D matrix multiply |
| `torch.dot` | **Supported** | vector dot product |
| `torch.transpose` | **Supported** | 2D transpose |
| `torch.bmm` | Not implemented | batched matmul |
| `torch.mv` | Not implemented | matrix-vector multiply |
| `torch.inner` | Not implemented | inner product |
| `torch.outer` / `ger` | Not implemented | outer product |
| `torch.cross` | Not implemented | cross product |
| `torch.det` / `logdet` / `slogdet` | Not implemented | determinants |
| `torch.inverse` / `pinverse` | Not implemented | matrix inverse |
| `torch.svd` / `svd_lowrank` / `pca_lowrank` | Not implemented | decomposition |
| `torch.qr` | Not implemented | QR decomposition |
| `torch.lu` / `lu_solve` / `lu_unpack` | Not implemented | LU decomposition |
| `torch.cholesky` / `cholesky_solve` / `cholesky_inverse` | Not implemented | Cholesky decomposition |
| `torch.matrix_power` | Not implemented | compose from repeated matmul |
| `torch.matrix_exp` | Not implemented | matrix exponential |
| `torch.addbmm` | Not implemented | batched matmul + add |
| `torch.addmm` | Not implemented | matmul + add |
| `torch.addmv` | Not implemented | matrix-vector + add |
| `torch.addr` | Not implemented | outer product + add |
| `torch.baddbmm` | Not implemented | batched matmul + add |
| `torch.chain_matmul` | Not implemented | compose from repeated matmul |
| `torch.tensordot` | Not implemented | tensor contraction |
| `torch.einsum` | Not implemented | Einstein summation |
| `torch.vdot` | Not implemented | use `cc.dot` |
| `torch.triangular_solve` | Not implemented | triangular system solve |
| `torch.geqrf` / `orgqr` / `ormqr` | Not implemented | LAPACK routines |
| `torch.lobpcg` | Not implemented | eigenvalue solver |
| `torch.trapezoid` / `cumulative_trapezoid` | Not implemented | numerical integration |
| `torch.trace` | Not implemented | matrix trace |
| `torch.diag` / `diag_embed` / `diagflat` / `diagonal` | Not implemented | diagonal ops |
| `torch.tril` / `triu` | Not implemented | triangular matrices |
| `torch.kron` | Not implemented | Kronecker product |

</details>

<details>
<summary><b>Shape / Indexing</b> — 9/32 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.reshape` | **Supported** | `cc.reshape` (zero-copy view) |
| `Tensor.view` | **Supported** | `cc.view` (alias for reshape) |
| `torch.squeeze` | **Supported** | `cc.squeeze` (zero-copy view) |
| `torch.unsqueeze` | **Supported** | `cc.unsqueeze` (zero-copy view) |
| `torch.flatten` | **Supported** | `cc.flatten` |
| `torch.unflatten` | **Supported** | `cc.unflatten` (zero-copy view) |
| `torch.cat` | **Supported** | `cc.concat` |
| `torch.stack` | **Supported** | `cc.stack` |
| `torch.where` | **Supported** | `cc.where` |
| `torch.permute` | Not implemented | generalized transpose |
| `torch.gather` | Not implemented | advanced indexing |
| `torch.scatter` / `scatter_add` / `scatter_reduce` | Not implemented | advanced indexing |
| `torch.index_select` | Not implemented | index-based selection |
| `torch.index_add` / `index_copy` / `index_reduce` | Not implemented | index-based ops |
| `torch.split` / `chunk` / `tensor_split` | Not implemented | tensor splitting |
| `torch.tile` | Not implemented | compose from concat |
| `torch.unbind` | Not implemented | unstack along dim |
| `torch.narrow` / `narrow_copy` | Not implemented | slice along dim |
| `torch.take` / `take_along_dim` | Not implemented | advanced indexing |
| `torch.nonzero` / `argwhere` | Not implemented | index finding |
| `torch.masked_select` | Not implemented | boolean indexing |
| `torch.movedim` / `moveaxis` / `swapaxes` / `swapdims` | Not implemented | axis reordering |
| `torch.flip` / `fliplr` / `flipud` | Not implemented | element reversal |
| `torch.roll` / `rot90` | Not implemented | circular shift / rotation |
| `torch.select` | Not implemented | indexing along dim |
| `torch.diagonal_scatter` / `select_scatter` / `slice_scatter` | Not implemented | scatter variants |
| `torch.dsplit` / `hsplit` / `vsplit` | Not implemented | splitting ops |
| `torch.column_stack` / `dstack` / `hstack` / `vstack` / `row_stack` | Not implemented | stacking variants |
| `torch.t` | Not implemented | use `cc.transpose` |
| `torch.ravel` / `unravel_index` | Not implemented | flatten variants |
| `torch.segment_reduce` | Not implemented | segment reduction |
| `torch.repeat_interleave` | Not implemented | element repetition |

</details>

<details>
<summary><b>Tensor Creation</b> — 8/19 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.zeros` | **Supported** | `cc.zeros` |
| `torch.ones` | **Supported** | `cc.ones` |
| `torch.full` | **Supported** | `cc.full` |
| `torch.arange` | **Supported** | `cc.arange` |
| `torch.linspace` | **Supported** | `cc.linspace` |
| `torch.zeros_like` | **Supported** | `cc.zeros_like` |
| `torch.ones_like` | **Supported** | `cc.ones_like` |
| `torch.full_like` | **Supported** | `cc.full_like` |
| `torch.empty` / `empty_like` | Not implemented | use `cc.zeros` / `cc.zeros_like` |
| `torch.eye` | Not implemented | identity matrix |
| `torch.logspace` | Not implemented | compose from `linspace` + `power` |
| `torch.rand` / `rand_like` | Not implemented | requires GPU PRNG |
| `torch.randn` / `randn_like` | Not implemented | requires GPU PRNG |
| `torch.randint` / `randint_like` | Not implemented | requires GPU PRNG |
| `torch.randperm` | Not implemented | requires GPU PRNG |
| `torch.from_numpy` | Not implemented | interop; use `Tensor(data)` |
| `torch.as_tensor` / `asarray` | Not implemented | interop |
| `torch.from_dlpack` | Not implemented | interop |
| `torch.frombuffer` / `from_file` | Not implemented | interop |

</details>

<details>
<summary><b>Activation Functions</b> — 23/28 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `F.relu` | **Supported** | `cc.relu` |
| `F.relu6` | **Supported** | `cc.relu6` |
| `F.sigmoid` | **Supported** | `cc.sigmoid` |
| `F.tanh` | **Supported** | `cc.tanh` |
| `F.gelu` | **Supported** | `cc.gelu` |
| `F.silu` / swish | **Supported** | `cc.silu` |
| `F.softplus` | **Supported** | `cc.softplus` |
| `F.leaky_relu` | **Supported** | `cc.leaky_relu` (scalar param) |
| `F.elu` | **Supported** | `cc.elu` |
| `F.selu` | **Supported** | `cc.selu` |
| `F.celu` | **Supported** | `cc.celu` |
| `F.mish` | **Supported** | `cc.mish` |
| `F.hardswish` | **Supported** | `cc.hardswish` |
| `F.hardsigmoid` | **Supported** | `cc.hardsigmoid` |
| `F.hardtanh` | **Supported** | `cc.hardtanh` |
| `F.softsign` | **Supported** | `cc.softsign` |
| `F.logsigmoid` | **Supported** | `cc.logsigmoid` |
| `F.tanhshrink` | **Supported** | `cc.tanhshrink` |
| `F.prelu` | **Supported** | `cc.prelu` (scalar param) |
| `F.hardshrink` | **Supported** | `cc.hardshrink` (scalar param) |
| `F.softshrink` | **Supported** | `cc.softshrink` (scalar param) |
| `F.softmax` | **Supported** | `cc.softmax` |
| `F.log_softmax` | **Supported** | `cc.log_softmax` |
| `F.threshold` | Not implemented | compose from where + clamp |
| `F.rrelu` | Not implemented | requires GPU PRNG |
| `F.glu` | Not implemented | gated linear unit |
| `F.softmin` | Not implemented | compose from neg + softmax |
| `F.gumbel_softmax` | Not implemented | requires GPU PRNG |

</details>

<details>
<summary><b>Loss Functions</b> — 3/20 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `F.mse_loss` | **Supported** | `cc.mse_loss` (composed) |
| `F.l1_loss` | **Supported** | `cc.l1_loss` (composed) |
| `F.cross_entropy` | **Supported** | `cc.cross_entropy_loss` (composed) |
| `F.smooth_l1_loss` / `huber_loss` | Not implemented | |
| `F.binary_cross_entropy` | Not implemented | |
| `F.binary_cross_entropy_with_logits` | Not implemented | |
| `F.poisson_nll_loss` | Not implemented | |
| `F.cosine_embedding_loss` | Not implemented | |
| `F.ctc_loss` | Not implemented | |
| `F.gaussian_nll_loss` | Not implemented | |
| `F.hinge_embedding_loss` | Not implemented | |
| `F.kl_div` | Not implemented | |
| `F.margin_ranking_loss` | Not implemented | |
| `F.multilabel_margin_loss` | Not implemented | |
| `F.multilabel_soft_margin_loss` | Not implemented | |
| `F.multi_margin_loss` | Not implemented | |
| `F.nll_loss` | Not implemented | requires gather |
| `F.soft_margin_loss` | Not implemented | |
| `F.triplet_margin_loss` | Not implemented | |
| `F.triplet_margin_with_distance_loss` | Not implemented | |

</details>

<details>
<summary><b>Other Utility Ops</b> — 0/18 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.cummax` / `cummin` | Not implemented | cumulative extrema |
| `torch.diff` | Not implemented | finite differences |
| `torch.meshgrid` | Not implemented | coordinate grids |
| `torch.logcumsumexp` | Not implemented | log cumulative sum exp |
| `torch.renorm` | Not implemented | tensor renormalization |
| `torch.searchsorted` | Not implemented | binary search |
| `torch.vander` | Not implemented | Vandermonde matrix |
| `torch.clone` | Not implemented | compose from `cc.add_scalar(x, 0)` |
| `torch.bincount` | Not implemented | histogram-like |
| `torch.block_diag` | Not implemented | block diagonal |
| `torch.broadcast_tensors` / `broadcast_to` | Not implemented | broadcasting |
| `torch.bucketize` | Not implemented | bucket assignment |
| `torch.cartesian_prod` | Not implemented | Cartesian product |
| `torch.cdist` | Not implemented | pairwise distance |
| `torch.combinations` | Not implemented | combinatorics |
| `torch.corrcoef` / `cov` | Not implemented | statistics |
| `torch.histc` / `histogram` | Not implemented | histograms |
| `torch.gcd` / `lcm` | Not implemented | integer math |

</details>

<details>
<summary><b>Spectral Ops</b> — 0/6 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.stft` / `istft` | Not implemented | Short-time Fourier transform |
| `torch.bartlett_window` | Not implemented | window function |
| `torch.blackman_window` | Not implemented | window function |
| `torch.hamming_window` | Not implemented | window function |
| `torch.hann_window` | Not implemented | window function |
| `torch.kaiser_window` | Not implemented | window function |

</details>

<details>
<summary><b>Random Sampling</b> — 0/4 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `torch.bernoulli` | Not implemented | requires GPU PRNG |
| `torch.multinomial` | Not implemented | requires GPU PRNG |
| `torch.normal` | Not implemented | requires GPU PRNG |
| `torch.poisson` | Not implemented | requires GPU PRNG |

</details>

<details>
<summary><b>Neural Network Layers</b> — 0/23 supported</summary>

| PyTorch | Status | Notes |
|---------|--------|-------|
| `nn.Conv1d` / `Conv2d` / `Conv3d` | Not implemented | spatial convolution |
| `nn.ConvTranspose1d` / `2d` / `3d` | Not implemented | transposed convolution |
| `nn.MaxPool1d` / `2d` / `3d` | Not implemented | max pooling |
| `nn.AvgPool1d` / `2d` / `3d` | Not implemented | average pooling |
| `nn.AdaptiveMaxPool` / `AdaptiveAvgPool` | Not implemented | adaptive pooling |
| `F.batch_norm` | Not implemented | batch normalization |
| `F.group_norm` | Not implemented | group normalization |
| `F.instance_norm` | Not implemented | instance normalization |
| `F.layer_norm` | Not implemented | layer normalization |
| `F.rms_norm` | Not implemented | RMS normalization |
| `F.normalize` | Not implemented | Lp normalization |
| `F.linear` | Not implemented | use matmul + add |
| `F.bilinear` | Not implemented | bilinear transform |
| `F.dropout` / `dropout2d` / `dropout3d` | Not implemented | requires GPU PRNG |
| `F.embedding` / `embedding_bag` | Not implemented | embedding lookup |
| `F.pairwise_distance` | Not implemented | distance function |
| `F.cosine_similarity` | Not implemented | compose from dot + norm |
| `F.pdist` | Not implemented | pairwise distance |
| `F.pad` | Not implemented | tensor padding |
| `F.interpolate` / `upsample` | Not implemented | spatial interpolation |
| `F.grid_sample` / `affine_grid` | Not implemented | spatial transform |
| `F.pixel_shuffle` / `pixel_unshuffle` | Not implemented | spatial rearrange |
| `F.scaled_dot_product_attention` | Not implemented | compose from matmul + softmax |

</details>

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
