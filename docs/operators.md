# CUT Operator Reference

Complete list of all operations with arguments and implementation status.

## Status Legend

- **Y** = Implemented
- **N** = Not implemented

---

## Element-wise Binary (Tensor, Tensor) → Tensor

All have vec-scalar variants (Tensor, scalar) → Tensor.

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 1 | add(a, b) | BinaryVecVecAdd | BinaryVecScalarAdd | Y |
| 2 | subtract(a, b) | BinaryVecVecSub | BinaryVecScalarSub | Y |
| 3 | multiply(a, b) | BinaryVecVecMul | BinaryVecScalarMul | Y |
| 4 | divide(a, b) | BinaryVecVecDiv | BinaryVecScalarDiv | Y |
| 5 | mod(a, b) | BinaryVecVecMod | BinaryVecScalarMod | Y |
| 6 | power(a, b) | BinaryVecVecPow | BinaryVecScalarPow | Y |
| 7 | floor_divide(a, b) | BinaryVecVecFloorDiv | BinaryVecScalarFloorDiv | Y |

## Element-wise Comparison (Tensor, Tensor) → Tensor

Returns 1.0/0.0. All have vec-scalar variants.

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 8 | equal(a, b) | BinaryVecVecEqual | BinaryVecScalarEqual | Y |
| 9 | not_equal(a, b) | BinaryVecVecNotEqual | BinaryVecScalarNotEqual | Y |
| 10 | less(a, b) | BinaryVecVecLess | BinaryVecScalarLess | Y |
| 11 | less_equal(a, b) | BinaryVecVecLessEqual | BinaryVecScalarLessEqual | Y |
| 12 | greater(a, b) | BinaryVecVecGreater | BinaryVecScalarGreater | Y |
| 13 | greater_equal(a, b) | BinaryVecVecGreaterEqual | BinaryVecScalarGreaterEqual | Y |

## Element-wise Min/Max

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 14 | minimum(a, b) | BinaryVecVecMin | BinaryVecScalarMin | Y |
| 15 | maximum(a, b) | BinaryVecVecMax | BinaryVecScalarMax | Y |

## Element-wise Bitwise (integer types)

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 16 | bitwise_and(a, b) | BinaryVecVecBitwiseAnd | BinaryVecScalarBitwiseAnd | Y |
| 17 | bitwise_or(a, b) | BinaryVecVecBitwiseOr | BinaryVecScalarBitwiseOr | Y |
| 18 | bitwise_xor(a, b) | BinaryVecVecBitwiseXor | BinaryVecScalarBitwiseXor | Y |
| 19 | left_shift(a, b) | BinaryVecVecLeftShift | BinaryVecScalarLeftShift | Y |
| 20 | right_shift(a, b) | BinaryVecVecRightShift | BinaryVecScalarRightShift | Y |

## Element-wise Logical

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 21 | logical_and(a, b) | BinaryVecVecLogicalAnd | BinaryVecScalarLogicalAnd | Y |
| 22 | logical_or(a, b) | BinaryVecVecLogicalOr | BinaryVecScalarLogicalOr | Y |
| 23 | logical_xor(a, b) | BinaryVecVecLogicalXor | BinaryVecScalarLogicalXor | Y |

## Element-wise Advanced Math (Binary)

| # | Operation | Vec-Vec Enum | Vec-Scalar Enum | Status |
|---|-----------|-------------|-----------------|--------|
| 24 | arctan2(a, b) | BinaryVecVecAtan2 | BinaryVecScalarAtan2 | Y |
| 25 | hypot(a, b) | BinaryVecVecHypot | BinaryVecScalarHypot | Y |
| 26 | copysign(a, b) | BinaryVecVecCopysign | BinaryVecScalarCopysign | Y |
| 27 | fmod(a, b) | BinaryVecVecFmod | BinaryVecScalarFmod | Y |
| 28 | logaddexp(a, b) | BinaryVecVecLogaddexp | BinaryVecScalarLogaddexp | Y |
| 29 | logaddexp2(a, b) | BinaryVecVecLogaddexp2 | BinaryVecScalarLogaddexp2 | Y |

## Element-wise Activation (Vec-Scalar only)

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 30 | leaky_relu(a, alpha) | alpha: float (negative slope) | BinaryVecScalarLeakyRelu | Y |
| 31 | prelu(a, weight) | weight: float | BinaryVecScalarPrelu | Y |
| 32 | hardshrink(a, lambda) | lambda: float (threshold) | BinaryVecScalarHardshrink | Y |
| 33 | softshrink(a, lambda) | lambda: float (threshold) | BinaryVecScalarSoftshrink | Y |

## Unary Operations (Tensor) → Tensor

### Basic Math

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 34 | negative(a) | UnaryNeg | Y |
| 35 | abs(a) | UnaryAbs | Y |
| 36 | sqrt(a) | UnarySqrt | Y |
| 37 | square(a) | UnarySquare | Y |
| 38 | reciprocal(a) | UnaryReciprocal | Y |
| 39 | sign(a) | UnarySign | Y |
| 40 | rsqrt(a) | UnaryRsqrt | Y |

### Exponential / Logarithmic

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 41 | exp(a) | UnaryExp | Y |
| 42 | exp2(a) | UnaryExp2 | Y |
| 43 | expm1(a) | UnaryExpm1 | Y |
| 44 | log(a) | UnaryLog | Y |
| 45 | log2(a) | UnaryLog2 | Y |
| 46 | log10(a) | UnaryLog10 | Y |
| 47 | log1p(a) | UnaryLog1p | Y |

### Trigonometric

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 48 | sin(a) | UnarySin | Y |
| 49 | cos(a) | UnaryCos | Y |
| 50 | tan(a) | UnaryTan | Y |
| 51 | arcsin(a) | UnaryAsin | Y |
| 52 | arccos(a) | UnaryAcos | Y |
| 53 | arctan(a) | UnaryAtan | Y |

### Hyperbolic

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 54 | sinh(a) | UnarySinh | Y |
| 55 | cosh(a) | UnaryCosh | Y |
| 56 | tanh(a) | UnaryTanh | Y |
| 57 | arcsinh(a) | UnaryAsinh | Y |
| 58 | arccosh(a) | UnaryAcosh | Y |
| 59 | arctanh(a) | UnaryAtanh | Y |

### Rounding

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 60 | floor(a) | UnaryFloor | Y |
| 61 | ceil(a) | UnaryCeil | Y |
| 62 | round(a) | UnaryRound | Y |
| 63 | trunc(a) | UnaryTrunc | Y |
| 64 | frac(a) | UnaryFrac | Y |

### Special Math

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 65 | cbrt(a) | UnaryCbrt | Y |
| 66 | degrees(a) | UnaryDegrees | Y |
| 67 | radians(a) | UnaryRadians | Y |

### Logical / Bitwise Unary

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 68 | logical_not(a) | UnaryLogicalNot | Y |
| 69 | invert(a) | UnaryBitwiseNot | Y |

### Unary Activation Functions

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 70 | relu(a) | UnaryRelu | Y |
| 71 | relu6(a) | UnaryRelu6 | Y |
| 72 | sigmoid(a) | UnarySigmoid | Y |
| 73 | gelu(a) | UnaryGelu | Y |
| 74 | silu(a) | UnarySilu | Y |
| 75 | softplus(a) | UnarySoftplus | Y |
| 76 | elu(a) | UnaryElu | Y |
| 77 | selu(a) | UnarySelu | Y |
| 78 | celu(a) | UnaryCelu | Y |
| 79 | mish(a) | UnaryMish | Y |
| 80 | hardswish(a) | UnaryHardswish | Y |
| 81 | hardsigmoid(a) | UnaryHardsigmoid | Y |
| 82 | hardtanh(a) | UnaryHardtanh | Y |
| 83 | softsign(a) | UnarySoftsign | Y |
| 84 | logsigmoid(a) | UnaryLogSigmoid | Y |
| 85 | tanhshrink(a) | UnaryTanhshrink | Y |

### Check Operations

| # | Operation | Enum | Status |
|---|-----------|------|--------|
| 86 | isnan(a) | UnaryIsNan | Y |
| 87 | isinf(a) | UnaryIsInf | Y |
| 88 | isfinite(a) | UnaryIsFinite | Y |

## Ternary Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 89 | clamp(a, min, max) | min, max: scalar | TernaryClamp | Y |
| 90 | where(cond, x, y) | cond, x, y: Tensor (same shape) | TernarySelect | Y |

## Reduction Operations

All support optional `dim` parameter. Without dim: global reduction → shape {1}. With dim: reduces along that dimension.

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 91 | sum(a, dim=None) | dim: optional int | ReduceSum | Y |
| 92 | mean(a, dim=None) | dim: optional int | ReduceMean | Y |
| 93 | min(a, dim=None) | dim: optional int | ReduceMin | Y |
| 94 | max(a, dim=None) | dim: optional int | ReduceMax | Y |
| 95 | prod(a, dim=None) | dim: optional int | ReduceProd | Y |
| 96 | any(a, dim=None) | dim: optional int | ReduceAny | Y |
| 97 | all(a, dim=None) | dim: optional int | ReduceAll | Y |
| 98 | argmax(a, dim=None) | dim: optional int | ReduceArgmax | Y |
| 99 | argmin(a, dim=None) | dim: optional int | ReduceArgmin | Y |

## Statistical Operations

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 100 | variance(a, correction=1, dim=None) | correction: int, dim: optional int | Y |
| 101 | std(a, correction=1, dim=None) | correction: int, dim: optional int | Y (Python: sqrt(variance)) |

## Cumulative Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 102 | cumsum(a, dim=0) | dim: int | CumSum | Y |
| 103 | cumprod(a, dim=0) | dim: int | CumProd | Y |

## Prefix Scan Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 104 | prefix_scan_exclusive_sum(a) | — | PrefixScanExclusiveSum | Y |
| 105 | prefix_scan_inclusive_sum(a) | — | PrefixScanInclusiveSum | Y |

## Sort Operations (in-place)

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 106 | sort_bitonic(keys, vals) | keys, vals: Tensor | SortBitonic | Y |
| 107 | sort_radix(keys, vals) | keys, vals: Tensor | SortRadix | Y |

## Matrix / Linear Algebra Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 108 | matmul(a, b) | a: [M,K], b: [K,N] → [M,N] | MatMul | Y |
| 109 | transpose(a) | a: [M,N] → [N,M] | Transpose | Y |
| 110 | dot(a, b) | a, b: 1D → scalar | Dot | Y |
| 111 | bmm(a, b) | a: [B,M,K], b: [B,K,N] → [B,M,N] | — | N |
| 112 | mv(a, b) | a: [M,N], b: [N] → [M] | — | N |
| 113 | outer(a, b) | a: [M], b: [N] → [M,N] | — | N |
| 114 | einsum(equation, *tensors) | Einstein summation | — | N |

## Norm Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 115 | norm(a, dim=None) | L2 norm, dim: optional int | Norm / NormDim | Y |

## Softmax Operations

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 116 | softmax(a, dim=-1) | dim: int | Y |
| 117 | log_softmax(a, dim=-1) | dim: int | Y |

## Tensor Creation Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 118 | zeros(shape, dtype) | shape: tuple, dtype: DataType | Zeros | Y |
| 119 | ones(shape, dtype) | shape: tuple, dtype: DataType | Ones | Y |
| 120 | full(shape, fill_value, dtype) | shape: tuple, value: scalar | Full | Y |
| 121 | arange(start, end, step, dtype) | start, end, step: scalar | Arange | Y |
| 122 | linspace(start, end, steps, dtype) | start, end: float, steps: int | Linspace | Y |
| 123 | rand(shape, dtype) | Random uniform [0,1) | — | N |
| 124 | randn(shape, dtype) | Random normal N(0,1) | — | N |
| 125 | eye(n, dtype) | Identity matrix | — | N |

## Shape Manipulation Operations

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 126 | reshape(a, new_shape) | new_shape: list[int] (supports -1) | Y |
| 127 | squeeze(a, dim=None) | dim: optional int | Y |
| 128 | unsqueeze(a, dim) | dim: int | Y |
| 129 | flatten(a, start_dim=0, end_dim=-1) | start_dim, end_dim: int | Y |
| 130 | unflatten(a, dim, sizes) | dim: int, sizes: list[int] | Y |
| 131 | permute(a, dims) | dims: list[int] | N |
| 132 | expand(a, shape) | shape: list[int] | N |
| 133 | repeat(a, repeats) | repeats: list[int] | N |
| 134 | split(a, sections, dim=0) | sections: int/list | N |
| 135 | chunk(a, chunks, dim=0) | chunks: int | N |
| 136 | flip(a, dims) | dims: list[int] | N |
| 137 | roll(a, shifts, dims) | shifts, dims: list[int] | N |

## Tensor Joining Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 138 | cat(tensors, dim=0) | tensors: list, dim: int | Concat | Y |
| 139 | stack(tensors, dim=0) | tensors: list, dim: int | Stack | Y |

## Convolution Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 140 | conv1d(input, weight, stride=1, padding=0) | input: [N,C_in,L], weight: [C_out,C_in,kL] | Conv1D | Y |
| 141 | conv2d(input, weight, stride=1, padding=0) | input: [N,C_in,H,W], weight: [C_out,C_in,kH,kW] | Conv2D | Y |
| 142 | conv3d(...) | 3D convolution | — | N |
| 143 | conv_transpose2d(...) | Transposed 2D convolution | — | N |

## Pooling Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 144 | max_pool2d(input, kernel_size, stride, padding) | input: [N,C,H,W] | MaxPool2D | Y |
| 145 | avg_pool2d(input, kernel_size, stride, padding) | input: [N,C,H,W] | AvgPool2D | Y |
| 146 | adaptive_avg_pool2d(input, output_size) | input: [N,C,H,W], output_size: (H,W) | — | Y |

## Normalization Operations

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 147 | layer_norm(input, normalized_shape, weight, bias, eps) | input: [*,norm_shape], weight/bias: [norm_shape] | Y |
| 148 | batch_norm(input, running_mean, running_var, weight, bias, eps) | input: [N,C,*], stats: [C] | Y |
| 149 | group_norm(input, num_groups, weight, bias, eps) | input: [N,C,*] | N |
| 150 | instance_norm(input, running_mean, running_var, weight, bias, eps) | input: [N,C,*] | N |

## Embedding Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 151 | embedding(indices, weight) | indices: [*] int, weight: [V,D] | Embedding | Y |

## Padding Operations

| # | Operation | Arguments | Enum | Status |
|---|-----------|-----------|------|--------|
| 152 | pad(input, pad_widths, value=0) | pad_widths: list[int], value: float | Pad | Y |

## Indexing Operations

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 153 | index_select(input, dim, index) | dim: int, index: 1D int tensor | N |
| 154 | gather(input, dim, index) | dim: int, index: int tensor | N |
| 155 | scatter(input, dim, index, src) | dim: int, index/src: tensors | N |
| 156 | masked_fill(input, mask, value) | mask: bool tensor, value: scalar | N |
| 157 | masked_select(input, mask) | mask: bool tensor | N |

## Loss Functions (training only)

| # | Operation | Arguments | Status |
|---|-----------|-----------|--------|
| 158 | cross_entropy(input, target) | — | N |
| 159 | mse_loss(input, target) | — | N |
| 160 | nll_loss(input, target) | — | N |

## Other Missing (lower priority)

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 161 | dropout(input, p=0.5) | N | Identity at inference |
| 162 | attention(q, k, v) | N | Can compose from matmul+softmax |
| 163 | linear(input, weight, bias) | N | Can compose from matmul+add |

---

## Summary

| Category | Implemented | Total | Coverage |
|----------|------------|-------|----------|
| Element-wise binary (vec-vec) | 29 | 29 | 100% |
| Element-wise binary (vec-scalar) | 33 | 33 | 100% |
| Unary | 55 | 55 | 100% |
| Ternary | 2 | 2 | 100% |
| Reduction | 9 | 9 | 100% |
| Statistical | 2 | 2 | 100% |
| Cumulative | 2 | 2 | 100% |
| Prefix scan | 2 | 2 | 100% |
| Sort | 2 | 2 | 100% |
| Matrix/LA | 3 | 7 | 43% |
| Norm | 1 | 1 | 100% |
| Softmax | 2 | 2 | 100% |
| Creation | 5 | 8 | 63% |
| Shape | 5 | 12 | 42% |
| Joining | 2 | 2 | 100% |
| Convolution | 2 | 4 | 50% |
| Pooling | 3 | 3 | 100% |
| Normalization | 2 | 4 | 50% |
| Embedding | 1 | 1 | 100% |
| Padding | 1 | 1 | 100% |
| Indexing | 0 | 5 | 0% |
| Loss | 0 | 3 | 0% |
| **Total** | **161** | **189** | **85%** |
