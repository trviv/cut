# PyTorch vs CUT — Operator Comparison

Comprehensive comparison of PyTorch operators (with argument variations) against CUT's implementation status. Scope: all ops relevant to inference/training of common models (CNNs, transformers, MLPs).

## Status Legend

- **Y** = Implemented in CUT
- **P** = Partial (implemented but missing some PyTorch argument variations)
- **N** = Not implemented in CUT

---

## 1. Element-wise Arithmetic

All PyTorch arithmetic ops support: broadcasting, `out` parameter, `dtype` promotion, and in-place variants (`_` suffix). CUT supports vec-vec and vec-scalar variants but no in-place or `out` parameter.

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 1 | `torch.add` | `add(input, other, *, alpha=1, out=None)` | Y | No `alpha` multiplier, no `out` param |
| 2 | `torch.sub` | `sub(input, other, *, alpha=1, out=None)` | Y | No `alpha` multiplier |
| 3 | `torch.mul` | `mul(input, other, *, out=None)` | Y | — |
| 4 | `torch.div` | `div(input, other, *, rounding_mode=None, out=None)` | Y | No `rounding_mode` ('trunc', 'floor') |
| 5 | `torch.pow` | `pow(input, exponent, *, out=None)` | Y | — |
| 6 | `torch.fmod` | `fmod(input, other, *, out=None)` | Y | — |
| 7 | `torch.remainder` | `remainder(input, other, *, out=None)` | Y | (CUT calls it `mod`) |
| 8 | `torch.floor_divide` | `floor_divide(input, other, *, out=None)` | Y | — |
| 9 | `torch.neg` | `neg(input, *, out=None)` | Y | — |
| 10 | `torch.abs` | `abs(input, *, out=None)` | Y | — |
| 11 | `torch.sign` | `sign(input, *, out=None)` | Y | — |
| 12 | `torch.sqrt` | `sqrt(input, *, out=None)` | Y | — |
| 13 | `torch.rsqrt` | `rsqrt(input, *, out=None)` | Y | — |
| 14 | `torch.reciprocal` | `reciprocal(input, *, out=None)` | Y | — |
| 15 | `torch.square` | `square(input, *, out=None)` | Y | — |
| 16 | `torch.exp` | `exp(input, *, out=None)` | Y | — |
| 17 | `torch.exp2` | `exp2(input, *, out=None)` | Y | — |
| 18 | `torch.expm1` | `expm1(input, *, out=None)` | Y | — |
| 19 | `torch.log` | `log(input, *, out=None)` | Y | — |
| 20 | `torch.log2` | `log2(input, *, out=None)` | Y | — |
| 21 | `torch.log10` | `log10(input, *, out=None)` | Y | — |
| 22 | `torch.log1p` | `log1p(input, *, out=None)` | Y | — |
| 23 | `torch.ceil` | `ceil(input, *, out=None)` | Y | — |
| 24 | `torch.floor` | `floor(input, *, out=None)` | Y | — |
| 25 | `torch.round` | `round(input, *, decimals=0, out=None)` | Y | No `decimals` param |
| 26 | `torch.trunc` | `trunc(input, *, out=None)` | Y | — |
| 27 | `torch.frac` | `frac(input, *, out=None)` | Y | — |
| 28 | `torch.clamp` | `clamp(input, min=None, max=None, *, out=None)` | Y | CUT requires both min and max (no one-sided clamp) |
| 29 | `torch.lerp` | `lerp(input, end, weight, *, out=None)` | N | — |
| 30 | `torch.addcmul` | `addcmul(input, tensor1, tensor2, *, value=1, out=None)` | N | — |
| 31 | `torch.addcdiv` | `addcdiv(input, tensor1, tensor2, *, value=1, out=None)` | N | — |

**Global note:** CUT does not support in-place (`_` suffix) or `out=` parameter for any element-wise op. All PyTorch pointwise ops have these; they are omitted from the Gaps column for brevity.

---

## 2. Element-wise Comparison

PyTorch returns bool tensors. CUT returns float tensors (1.0/0.0).

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 32 | `torch.eq` | `eq(input, other, *, out=None)` | Y | Returns float, not bool |
| 33 | `torch.ne` | `ne(input, other, *, out=None)` | Y | Returns float, not bool |
| 34 | `torch.gt` | `gt(input, other, *, out=None)` | Y | Returns float, not bool |
| 35 | `torch.ge` | `ge(input, other, *, out=None)` | Y | Returns float, not bool |
| 36 | `torch.lt` | `lt(input, other, *, out=None)` | Y | Returns float, not bool |
| 37 | `torch.le` | `le(input, other, *, out=None)` | Y | Returns float, not bool |
| 38 | `torch.minimum` | `minimum(input, other, *, out=None)` | Y | — |
| 39 | `torch.maximum` | `maximum(input, other, *, out=None)` | Y | — |
| 40 | `torch.isnan` | `isnan(input)` | Y | — |
| 41 | `torch.isinf` | `isinf(input)` | Y | — |
| 42 | `torch.isfinite` | `isfinite(input)` | Y | — |
| 43 | `torch.isclose` | `isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)` | N | — |
| 44 | `torch.allclose` | `allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)` | N | — |
| 45 | `torch.isreal` | `isreal(input)` | N | — |
| 46 | `torch.isneginf` | `isneginf(input, *, out=None)` | N | — |
| 47 | `torch.isposinf` | `isposinf(input, *, out=None)` | N | — |

---

## 3. Logical & Bitwise

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 48 | `torch.logical_and` | `logical_and(input, other, *, out=None)` | Y | — |
| 49 | `torch.logical_or` | `logical_or(input, other, *, out=None)` | Y | — |
| 50 | `torch.logical_xor` | `logical_xor(input, other, *, out=None)` | Y | — |
| 51 | `torch.logical_not` | `logical_not(input, *, out=None)` | Y | — |
| 52 | `torch.bitwise_and` | `bitwise_and(input, other, *, out=None)` | Y | — |
| 53 | `torch.bitwise_or` | `bitwise_or(input, other, *, out=None)` | Y | — |
| 54 | `torch.bitwise_xor` | `bitwise_xor(input, other, *, out=None)` | Y | — |
| 55 | `torch.bitwise_not` | `bitwise_not(input, *, out=None)` | Y | — |
| 56 | `torch.bitwise_left_shift` | `bitwise_left_shift(input, other, *, out=None)` | Y | — |
| 57 | `torch.bitwise_right_shift` | `bitwise_right_shift(input, other, *, out=None)` | Y | — |

---

## 4. Trigonometric & Hyperbolic

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 58 | `torch.sin` | `sin(input, *, out=None)` | Y | — |
| 59 | `torch.cos` | `cos(input, *, out=None)` | Y | — |
| 60 | `torch.tan` | `tan(input, *, out=None)` | Y | — |
| 61 | `torch.asin` | `asin(input, *, out=None)` | Y | — |
| 62 | `torch.acos` | `acos(input, *, out=None)` | Y | — |
| 63 | `torch.atan` | `atan(input, *, out=None)` | Y | — |
| 64 | `torch.atan2` | `atan2(input, other, *, out=None)` | Y | — |
| 65 | `torch.sinh` | `sinh(input, *, out=None)` | Y | — |
| 66 | `torch.cosh` | `cosh(input, *, out=None)` | Y | — |
| 67 | `torch.tanh` | `tanh(input, *, out=None)` | Y | — |
| 68 | `torch.asinh` | `asinh(input, *, out=None)` | Y | — |
| 69 | `torch.acosh` | `acosh(input, *, out=None)` | Y | — |
| 70 | `torch.atanh` | `atanh(input, *, out=None)` | Y | — |
| 71 | `torch.hypot` | `hypot(input, other, *, out=None)` | Y | — |
| 72 | `torch.sinc` | `sinc(input, *, out=None)` | N | — |

---

## 5. Special Math

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 73 | `torch.copysign` | `copysign(input, other, *, out=None)` | Y | — |
| 74 | `torch.logaddexp` | `logaddexp(input, other, *, out=None)` | Y | — |
| 75 | `torch.logaddexp2` | `logaddexp2(input, other, *, out=None)` | Y | — |
| 76 | `torch.deg2rad` | `deg2rad(input, *, out=None)` | Y | (CUT: `radians`) |
| 77 | `torch.rad2deg` | `rad2deg(input, *, out=None)` | Y | (CUT: `degrees`) |
| 78 | (no PyTorch builtin) | — | Y | CUT has `cbrt` (cube root), PyTorch uses `pow(x, 1/3)` |
| 79 | `torch.special.erf` | `erf(input, *, out=None)` | N | — |
| 80 | `torch.special.erfc` | `erfc(input, *, out=None)` | N | — |
| 81 | `torch.special.erfinv` | `erfinv(input, *, out=None)` | N | — |
| 82 | `torch.lgamma` | `lgamma(input, *, out=None)` | N | — |
| 83 | `torch.digamma` | `digamma(input, *, out=None)` | N | — |
| 84 | `torch.special.i0` | `i0(input, *, out=None)` | N | — |
| 85 | `torch.special.i1` | `i1(input, *, out=None)` | N | — |

---

## 6. Activation Functions

All PyTorch activations are available as both `torch.nn.functional.*` (functional) and `torch.nn.*` (module). CUT implements functional-style only.

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 86 | `relu` | `relu(input, inplace=False)` | Y | No `inplace` |
| 87 | `relu6` | `relu6(input, inplace=False)` | Y | No `inplace` |
| 88 | `leaky_relu` | `leaky_relu(input, negative_slope=0.01, inplace=False)` | Y | No `inplace` |
| 89 | `prelu` | `prelu(input, weight)` | Y | CUT takes scalar weight only (PyTorch supports per-channel weight tensor) |
| 90 | `elu` | `elu(input, alpha=1.0, inplace=False)` | Y | CUT uses fixed alpha=1.0, no configurable alpha |
| 91 | `selu` | `selu(input, inplace=False)` | Y | — |
| 92 | `celu` | `celu(input, alpha=1.0, inplace=False)` | Y | CUT uses fixed alpha=1.0, no configurable alpha |
| 93 | `gelu` | `gelu(input, approximate='none')` | Y | No `approximate='tanh'` option |
| 94 | `silu` / `swish` | `silu(input, inplace=False)` | Y | — |
| 95 | `mish` | `mish(input, inplace=False)` | Y | — |
| 96 | `sigmoid` | `sigmoid(input)` | Y | — |
| 97 | `hardsigmoid` | `hardsigmoid(input, inplace=False)` | Y | — |
| 98 | `hardswish` | `hardswish(input, inplace=False)` | Y | — |
| 99 | `hardtanh` | `hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)` | Y | CUT uses fixed min=-1, max=1 |
| 100 | `softplus` | `softplus(input, beta=1, threshold=20)` | Y | No configurable `beta`/`threshold` |
| 101 | `softsign` | `softsign(input)` | Y | — |
| 102 | `softshrink` | `softshrink(input, lambd=0.5)` | Y | — |
| 103 | `hardshrink` | `hardshrink(input, lambd=0.5)` | Y | — |
| 104 | `tanhshrink` | `tanhshrink(input)` | Y | — |
| 105 | `logsigmoid` | `logsigmoid(input)` | Y | — |
| 106 | `threshold` | `threshold(input, threshold, value, inplace=False)` | N | — |
| 107 | `rrelu` | `rrelu(input, lower=1/8, upper=1/3, training=False, inplace=False)` | N | — |
| 108 | `glu` | `glu(input, dim=-1)` | N | — |

---

## 7. Reduction Operations

PyTorch reductions support `dim` (int or tuple of ints), `keepdim`, and `dtype`. CUT supports single `dim` (int) and no `keepdim` or `dtype` args.

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 109 | `torch.sum` | `sum(input, dim=None, keepdim=False, *, dtype=None, out=None)` | Y | No `keepdim`, no multi-dim, no `dtype` |
| 110 | `torch.mean` | `mean(input, dim=None, keepdim=False, *, dtype=None, out=None)` | Y | No `keepdim`, no multi-dim, no `dtype` |
| 111 | `torch.max` | `max(input, dim=None, keepdim=False, *, out=None)` → (values, indices) | P | No `keepdim`; global max returns value only (no indices); dim-wise returns value only (no indices) |
| 112 | `torch.min` | `min(input, dim=None, keepdim=False, *, out=None)` → (values, indices) | P | Same as max |
| 113 | `torch.prod` | `prod(input, dim=None, keepdim=False, *, dtype=None, out=None)` | Y | No `keepdim`, no multi-dim |
| 114 | `torch.any` | `any(input, dim=None, keepdim=False, *, out=None)` | Y | No `keepdim` |
| 115 | `torch.all` | `all(input, dim=None, keepdim=False, *, out=None)` | Y | No `keepdim` |
| 116 | `torch.argmax` | `argmax(input, dim=None, keepdim=False)` | Y | No `keepdim` |
| 117 | `torch.argmin` | `argmin(input, dim=None, keepdim=False)` | Y | No `keepdim` |
| 118 | `torch.logsumexp` | `logsumexp(input, dim, keepdim=False, *, out=None)` | N | — |
| 119 | `torch.count_nonzero` | `count_nonzero(input, dim=None)` | N | — |
| 120 | `torch.median` | `median(input, dim=-1, keepdim=False, *, out=None)` | N | — |
| 121 | `torch.nanmean` | `nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None)` | N | — |
| 122 | `torch.nansum` | `nansum(input, dim=None, keepdim=False, *, dtype=None, out=None)` | N | — |
| 123 | `torch.amax` | `amax(input, dim=None, keepdim=False, *, out=None)` | N | (max without indices) |
| 124 | `torch.amin` | `amin(input, dim=None, keepdim=False, *, out=None)` | N | (min without indices) |
| 125 | `torch.aminmax` | `aminmax(input, *, dim=None, keepdim=False, out=None)` | N | — |

---

## 8. Statistical Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 126 | `torch.var` | `var(input, dim=None, unbiased=True, keepdim=False, *, correction=1, out=None)` | Y | No `keepdim`; CUT uses `correction` param (matches PyTorch) |
| 127 | `torch.std` | `std(input, dim=None, unbiased=True, keepdim=False, *, correction=1, out=None)` | Y | No `keepdim`; Python-side impl (sqrt of variance) |
| 128 | `torch.norm` | `norm(input, p='fro', dim=None, keepdim=False, *, dtype=None, out=None)` | P | L2 only (no p-norm), no `keepdim` |
| 129 | `torch.var_mean` | `var_mean(input, dim=None, unbiased=True, keepdim=False, *, correction=1)` | N | — |
| 130 | `torch.std_mean` | `std_mean(input, dim=None, unbiased=True, keepdim=False, *, correction=1)` | N | — |
| 131 | `torch.histc` | `histc(input, bins=100, min=0, max=0, *, out=None)` | N | — |
| 132 | `torch.histogram` | `histogram(input, bins, *, range=None, weight=None, density=False, out=None)` | N | — |

---

## 9. Cumulative / Scan Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 133 | `torch.cumsum` | `cumsum(input, dim, *, dtype=None, out=None)` | Y | No `dtype` |
| 134 | `torch.cumprod` | `cumprod(input, dim, *, dtype=None, out=None)` | Y | No `dtype` |
| 135 | `torch.cummax` | `cummax(input, dim)` → (values, indices) | N | — |
| 136 | `torch.cummin` | `cummin(input, dim)` → (values, indices) | N | — |
| 137 | `torch.logcumsumexp` | `logcumsumexp(input, dim, *, out=None)` | N | — |

CUT also has `prefix_scan_exclusive_sum` and `prefix_scan_inclusive_sum` (no PyTorch equivalent — these are GPU-specific parallel scan primitives).

---

## 10. Sort Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 138 | `torch.sort` | `sort(input, dim=-1, descending=False, stable=False)` → (values, indices) | P | CUT has `sort_bitonic` and `sort_radix` (in-place, key-value pair); no `dim`, no `descending` flag, no `stable` flag |
| 139 | `torch.argsort` | `argsort(input, dim=-1, descending=False, stable=False)` | N | — |
| 140 | `torch.topk` | `topk(input, k, dim=-1, largest=True, sorted=True)` → (values, indices) | N | — |
| 141 | `torch.kthvalue` | `kthvalue(input, k, dim=-1, keepdim=False)` → (values, indices) | N | — |
| 142 | `torch.msort` | `msort(input)` | N | — |

---

## 11. Softmax

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 143 | `F.softmax` | `softmax(input, dim=None, *, dtype=None)` | Y | No `dtype` |
| 144 | `F.log_softmax` | `log_softmax(input, dim=None, *, dtype=None)` | Y | No `dtype` |
| 145 | `F.gumbel_softmax` | `gumbel_softmax(logits, tau=1, hard=False, dim=-1)` | N | — |

---

## 12. Matrix / Linear Algebra

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 146 | `torch.matmul` | `matmul(input, other, *, out=None)` — supports 1D-4D with broadcasting | P | 2D only (`[M,K] @ [K,N]`); no broadcasting, no batched matmul for N>2D |
| 147 | `torch.mm` | `mm(input, mat2, *, out=None)` — strict 2D | Y | — (equivalent to CUT matmul) |
| 148 | `torch.bmm` | `bmm(input, mat2, *, out=None)` — batched 3D | N | — |
| 149 | `torch.dot` | `dot(input, other, *, out=None)` — 1D | Y | — |
| 150 | `torch.mv` | `mv(input, vec, *, out=None)` — matrix-vector | N | — |
| 151 | `torch.outer` | `outer(input, vec2, *, out=None)` | N | — |
| 152 | `torch.addmm` | `addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)` | N | — |
| 153 | `torch.baddbmm` | `baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)` | N | — |
| 154 | `torch.einsum` | `einsum(equation, *operands)` | N | — |
| 155 | `torch.transpose` | `transpose(input, dim0, dim1)` | P | CUT only supports 2D transpose (`[M,N]→[N,M]`); no arbitrary dimension swap |
| 156 | `torch.t` | `t(input)` — 2D transpose shorthand | Y | — (equivalent to CUT transpose) |
| 157 | `torch.linalg.cross` | `cross(input, other, *, dim=-1)` | N | — |
| 158 | `torch.trace` | `trace(input)` | N | — |
| 159 | `torch.diag` | `diag(input, diagonal=0)` | N | — |
| 160 | `torch.tril` | `tril(input, diagonal=0, *, out=None)` | N | — |
| 161 | `torch.triu` | `triu(input, diagonal=0, *, out=None)` | N | — |

---

## 13. Convolution Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 162 | `F.conv1d` | `conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)` | P | No `bias`, no `dilation`, no `groups` |
| 163 | `F.conv2d` | `conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)` | P | No `bias`, no `dilation`, no `groups` |
| 164 | `F.conv3d` | `conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)` | N | — |
| 165 | `F.conv_transpose1d` | `conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)` | N | — |
| 166 | `F.conv_transpose2d` | `conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)` | N | — |
| 167 | `F.conv_transpose3d` | `conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)` | N | — |

**Padding modes** (PyTorch): `'zeros'` (default), `'reflect'`, `'replicate'`, `'circular'`. CUT: zero-padding only.

---

## 14. Pooling Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 168 | `F.max_pool1d` | `max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)` | N | — |
| 169 | `F.max_pool2d` | `max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)` | P | No `dilation`, no `ceil_mode`, no `return_indices` |
| 170 | `F.max_pool3d` | `max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)` | N | — |
| 171 | `F.avg_pool1d` | `avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)` | N | — |
| 172 | `F.avg_pool2d` | `avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)` | P | No `ceil_mode`, no `count_include_pad`, no `divisor_override` |
| 173 | `F.avg_pool3d` | `avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)` | N | — |
| 174 | `F.adaptive_max_pool1d` | `adaptive_max_pool1d(input, output_size, return_indices=False)` | N | — |
| 175 | `F.adaptive_max_pool2d` | `adaptive_max_pool2d(input, output_size, return_indices=False)` | N | — |
| 176 | `F.adaptive_max_pool3d` | `adaptive_max_pool3d(input, output_size, return_indices=False)` | N | — |
| 177 | `F.adaptive_avg_pool1d` | `adaptive_avg_pool1d(input, output_size)` | N | — |
| 178 | `F.adaptive_avg_pool2d` | `adaptive_avg_pool2d(input, output_size)` | Y | — |
| 179 | `F.adaptive_avg_pool3d` | `adaptive_avg_pool3d(input, output_size)` | N | — |
| 180 | `F.lp_pool1d` | `lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)` | N | — |
| 181 | `F.lp_pool2d` | `lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)` | N | — |
| 182 | `F.max_unpool2d` | `max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)` | N | — |

---

## 15. Normalization Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 183 | `F.batch_norm` | `batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)` | P | No `training` mode (inference only), no `momentum` |
| 184 | `F.layer_norm` | `layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)` | Y | — |
| 185 | `F.group_norm` | `group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)` | N | — |
| 186 | `F.instance_norm` | `instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)` | N | — |
| 187 | `F.local_response_norm` | `local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0)` | N | — |
| 188 | `torch.nn.RMSNorm` | `RMSNorm(normalized_shape, eps=1e-5, elementwise_affine=True)` | N | — |

---

## 16. Loss Functions

All PyTorch losses support `reduction` parameter: `'none'` | `'mean'` | `'sum'`.

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 189 | `F.cross_entropy` | `cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` | N | — |
| 190 | `F.mse_loss` | `mse_loss(input, target, reduction='mean')` | N | — |
| 191 | `F.nll_loss` | `nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean')` | N | — |
| 192 | `F.binary_cross_entropy` | `binary_cross_entropy(input, target, weight=None, reduction='mean')` | N | — |
| 193 | `F.binary_cross_entropy_with_logits` | `binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None)` | N | — |
| 194 | `F.smooth_l1_loss` | `smooth_l1_loss(input, target, reduction='mean', beta=1.0)` | N | — |
| 195 | `F.huber_loss` | `huber_loss(input, target, reduction='mean', delta=1.0)` | N | — |
| 196 | `F.l1_loss` | `l1_loss(input, target, reduction='mean')` | N | — |
| 197 | `F.kl_div` | `kl_div(input, target, reduction='mean', log_target=False)` | N | — |
| 198 | `F.cosine_embedding_loss` | `cosine_embedding_loss(input1, input2, target, margin=0.0, reduction='mean')` | N | — |
| 199 | `F.hinge_embedding_loss` | `hinge_embedding_loss(input, target, margin=1.0, reduction='mean')` | N | — |
| 200 | `F.triplet_margin_loss` | `triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, reduction='mean')` | N | — |

---

## 17. Tensor Creation

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 201 | `torch.zeros` | `zeros(*size, *, dtype=None, device=None, requires_grad=False)` | Y | — |
| 202 | `torch.ones` | `ones(*size, *, dtype=None, device=None, requires_grad=False)` | Y | — |
| 203 | `torch.full` | `full(size, fill_value, *, dtype=None, device=None, requires_grad=False)` | Y | — |
| 204 | `torch.arange` | `arange(start=0, end, step=1, *, dtype=None, device=None, requires_grad=False)` | Y | — |
| 205 | `torch.linspace` | `linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False)` | Y | — |
| 206 | `torch.logspace` | `logspace(start, end, steps, base=10.0, *, dtype=None, device=None)` | N | — |
| 207 | `torch.eye` | `eye(n, m=None, *, dtype=None, device=None, requires_grad=False)` | N | — |
| 208 | `torch.rand` | `rand(*size, *, dtype=None, device=None, requires_grad=False)` | N | — |
| 209 | `torch.randn` | `randn(*size, *, dtype=None, device=None, requires_grad=False)` | N | — |
| 210 | `torch.randint` | `randint(low=0, high, size, *, dtype=torch.int64, device=None)` | N | — |
| 211 | `torch.empty` | `empty(*size, *, dtype=None, device=None, requires_grad=False)` | N | — |
| 212 | `torch.zeros_like` | `zeros_like(input, *, dtype=None, device=None)` | N | — |
| 213 | `torch.ones_like` | `ones_like(input, *, dtype=None, device=None)` | N | — |
| 214 | `torch.full_like` | `full_like(input, fill_value, *, dtype=None, device=None)` | N | — |
| 215 | `torch.rand_like` | `rand_like(input, *, dtype=None, device=None)` | N | — |
| 216 | `torch.randn_like` | `randn_like(input, *, dtype=None, device=None)` | N | — |

---

## 18. Shape Manipulation

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 217 | `torch.reshape` | `reshape(input, shape)` | Y | — |
| 218 | `Tensor.view` | `view(*shape)` — requires contiguous | Y | CUT reshape is equivalent (always copies) |
| 219 | `torch.squeeze` | `squeeze(input, dim=None)` | Y | — |
| 220 | `torch.unsqueeze` | `unsqueeze(input, dim)` | Y | — |
| 221 | `torch.flatten` | `flatten(input, start_dim=0, end_dim=-1)` | Y | — |
| 222 | `torch.unflatten` | `unflatten(input, dim, sizes)` | Y | — |
| 223 | `torch.permute` | `permute(input, dims)` | N | — |
| 224 | `torch.expand` | `Tensor.expand(*sizes)` — no data copy | N | — |
| 225 | `torch.repeat` | `Tensor.repeat(*sizes)` — copies data | N | — |
| 226 | `torch.split` | `split(tensor, split_size_or_sections, dim=0)` | N | — |
| 227 | `torch.chunk` | `chunk(input, chunks, dim=0)` | N | — |
| 228 | `torch.flip` | `flip(input, dims)` | N | — |
| 229 | `torch.roll` | `roll(input, shifts, dims=None)` | N | — |
| 230 | `torch.narrow` | `narrow(input, dim, start, length)` | N | — |
| 231 | `torch.select` | `select(input, dim, index)` | N | — |
| 232 | `Tensor.contiguous` | `contiguous(memory_format=torch.contiguous_format)` | N | CUT tensors are always contiguous |
| 233 | `torch.rot90` | `rot90(input, k=1, dims=[0,1])` | N | — |
| 234 | `torch.tile` | `tile(input, dims)` | N | — |
| 235 | `torch.movedim` | `movedim(input, source, destination)` | N | — |

---

## 19. Tensor Joining

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 236 | `torch.cat` | `cat(tensors, dim=0, *, out=None)` | Y | — |
| 237 | `torch.stack` | `stack(tensors, dim=0, *, out=None)` | Y | — |
| 238 | `torch.hstack` | `hstack(tensors, *, out=None)` | N | — |
| 239 | `torch.vstack` | `vstack(tensors, *, out=None)` | N | — |
| 240 | `torch.dstack` | `dstack(tensors, *, out=None)` | N | — |
| 241 | `torch.column_stack` | `column_stack(tensors, *, out=None)` | N | — |
| 242 | `torch.row_stack` | `row_stack(tensors, *, out=None)` | N | — |

---

## 20. Indexing Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 243 | `torch.index_select` | `index_select(input, dim, index, *, out=None)` | N | — |
| 244 | `torch.gather` | `gather(input, dim, index, *, sparse_grad=False, out=None)` | N | — |
| 245 | `Tensor.scatter_` | `scatter_(dim, index, src, *, reduce=None)` | N | — |
| 246 | `Tensor.scatter_add_` | `scatter_add_(dim, index, src)` | N | — |
| 247 | `torch.masked_fill` | `Tensor.masked_fill_(mask, value)` | N | — |
| 248 | `torch.masked_select` | `masked_select(input, mask, *, out=None)` | N | — |
| 249 | `torch.where` | `where(condition, input, other)` | Y | (CUT: `TernarySelect`) |
| 250 | `torch.nonzero` | `nonzero(input, *, as_tuple=False, out=None)` | N | — |
| 251 | `torch.index_put` | `Tensor.index_put_(indices, values, accumulate=False)` | N | — |
| 252 | `torch.take` | `take(input, index)` | N | — |
| 253 | `torch.take_along_dim` | `take_along_dim(input, indices, dim=None, *, out=None)` | N | — |
| 254 | `torch.index_copy` | `Tensor.index_copy_(dim, index, tensor)` | N | — |

---

## 21. Padding & Embedding

### Padding

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 255 | `F.pad` | `pad(input, pad, mode='constant', value=0.0)` | P | CUT: constant mode only. No `'reflect'`, `'replicate'`, `'circular'` modes |

### Embedding

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 256 | `F.embedding` | `embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)` | P | No `padding_idx`, no `max_norm`, no `scale_grad_by_freq`, no `sparse` |
| 257 | `F.embedding_bag` | `embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None)` | N | — |
| 258 | `F.one_hot` | `one_hot(tensor, num_classes=-1)` | N | — |

---

## 22. Other NN Operations

| # | PyTorch Operation | PyTorch Signature | CUT | CUT Gaps |
|---|---|---|---|---|
| 259 | `F.dropout` | `dropout(input, p=0.5, training=True, inplace=False)` | N | Identity at inference; not needed for inference-only use |
| 260 | `F.dropout2d` | `dropout2d(input, p=0.5, training=True, inplace=False)` | N | — |
| 261 | `F.linear` | `linear(input, weight, bias=None)` | N | Can compose from matmul + add |
| 262 | `F.bilinear` | `bilinear(input1, input2, weight, bias=None)` | N | — |
| 263 | `F.scaled_dot_product_attention` | `scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)` | N | Can compose from matmul + softmax + matmul |
| 264 | `F.interpolate` | `interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)` | N | — |
| 265 | `F.upsample` | `upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)` | N | — |
| 266 | `F.pixel_shuffle` | `pixel_shuffle(input, upscale_factor)` | N | — |
| 267 | `F.pixel_unshuffle` | `pixel_unshuffle(input, downscale_factor)` | N | — |
| 268 | `F.unfold` | `unfold(input, kernel_size, dilation=1, padding=0, stride=1)` | N | — |
| 269 | `F.fold` | `fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)` | N | — |
| 270 | `F.grid_sample` | `grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)` | N | — |

---

## Summary

### Coverage by Category

| Category | CUT (Y+P) | PyTorch Total | Coverage |
|---|---|---|---|
| Element-wise Arithmetic | 28 | 31 | 90% |
| Element-wise Comparison | 11 | 16 | 69% |
| Logical & Bitwise | 10 | 10 | 100% |
| Trigonometric & Hyperbolic | 14 | 15 | 93% |
| Special Math | 6 | 13 | 46% |
| Activation Functions | 20 | 23 | 87% |
| Reduction Ops | 9 | 17 | 53% |
| Statistical Ops | 3 | 7 | 43% |
| Cumulative / Scan | 2 | 5 | 40% |
| Sort Ops | 1 | 5 | 20% |
| Softmax | 2 | 3 | 67% |
| Matrix / Linear Algebra | 5 | 16 | 31% |
| Convolution | 2 | 6 | 33% |
| Pooling | 3 | 15 | 20% |
| Normalization | 2 | 6 | 33% |
| Loss Functions | 0 | 12 | 0% |
| Tensor Creation | 5 | 16 | 31% |
| Shape Manipulation | 6 | 19 | 32% |
| Tensor Joining | 2 | 7 | 29% |
| Indexing | 1 | 12 | 8% |
| Padding & Embedding | 2 | 4 | 50% |
| Other NN | 0 | 12 | 0% |
| **Total** | **134** | **270** | **50%** |

### Key Argument Gaps in "Implemented" Ops

Even where CUT implements an operator, these PyTorch features are commonly missing:

| Gap | Affected Ops | Impact |
|---|---|---|
| No `keepdim` | All reductions, var, std, norm | Users must manually unsqueeze after reducing |
| No multi-dim reduce | sum, mean, prod, etc. | Can't reduce over multiple dims at once |
| No `dilation` | conv1d, conv2d | Can't use dilated/atrous convolutions |
| No `groups` | conv1d, conv2d | Can't do depthwise-separable or grouped convolution |
| No `bias` in conv | conv1d, conv2d | Must manually add bias after conv |
| No in-place ops (`_`) | All ops | No memory-saving in-place variants |
| No `out` parameter | All ops | Can't write into pre-allocated output |
| No `alpha` in add/sub | add, sub | Can't do `a + alpha * b` in one op |
| L2 norm only | norm | No L1, Linf, or p-norm |
| 2D transpose only | transpose | Can't swap arbitrary dimensions |
| 2D matmul only | matmul | No batched matmul, no broadcasting rules |
| No per-channel PReLU | prelu | Weight must be scalar, not per-channel tensor |
| No `approximate` in GELU | gelu | Can't use faster tanh approximation |
| Constant pad only | pad | No reflect/replicate/circular padding |
| Float comparisons | eq, ne, gt, etc. | Returns 1.0/0.0 instead of bool tensor |
