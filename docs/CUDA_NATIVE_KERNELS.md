# Native CUDA Kernels: Architecture & Migration Plan

Status: COMPLETE (2026-07-20) — all phases landed and verified.
Verification (RTX 3090, 0f5d608): C++ suite 442/442 on `CUT_TEST_BACKEND=cuda`;
SmolLM2 token IDs identical to Vulkan; LTX 2B 10-step latents within
cross-backend tolerance of Vulkan (mean |diff| 0.003) at **1.72 s/step native
(WMMA tensor cores) vs 2.37 transpiled** — `CUT_CUDA_KERNELS=transpiled` is
bit-identical to the pre-migration baseline. Remaining perf gap vs Vulkan
fp16+coopmat (1.20 s/step): CUDA has no autotune data and the WMMA/attention
kernels are untuned first ports.

> Hard-won lesson: every embedded `.cuh` is passed to `nvrtcCreateProgram` by
> basename, and NVRTC rejects duplicate header names — which disables EVERY
> kernel at runtime. Header basenames must be unique repo-wide; the embedder
> now fails the build on collisions (see 0f5d608).

## Goal

Give every compute shader a first-class **native CUDA implementation** so the
CUDA backend runs hand-authored `.cu` kernels instead of auto-transpiled HLSL,
and can use CUDA-only hardware capability (tensor cores / WMMA, `__shfl_sync`
warp intrinsics, `__dp4a`). Vulkan (`.shader` HLSL / `.comp` GLSL) remains the
portable path; CUDA gets equal-citizen sources living **next to their
counterpart shader file**.

## Current state (before this work)

- `operators/impl/<group>/X.shader` — templated HLSL (`%VEC_DTYPE_SLOT%`
  placeholders, `#ifdef DTYPE_SLOT_IS_*` branches, `.shaderh` includes),
  expanded per dtype combo by `scripts/codegen/generate_shader_variants.py` from
  `shaders.json`, compiled to SPIR-V (DXC), embedded in `CompiledShaders.cpp`.
- 6 GLSL `.comp` cooperative-matrix matmul kernels (Vulkan-only).
- CUDA backend: `scripts/codegen/transpile_cuda_kernels.py` textually converts the
  dtype-preprocessed HLSL to CUDA C++; `scripts/codegen/embed_cuda_kernels.py` embeds
  the sources into `operators/runtime/cuda/CompiledCudaKernels.cpp` keyed by
  **normalized SPIR-V hash** (spec-constant literals zeroed). At runtime
  `CudaShaderContainer::createShader(spirv)` hashes the dispatched SPIR-V,
  finds the CUDA source, NVRTC-compiles it with `-DCUT_SPEC_<id>=<value>`,
  and launches with grid dims from the op-layer dispatch and block dims from
  SPIR-V reflection.
- Coverage 372/374 variants; the 2 `Fusion*` callback shaders and all 6
  coopmat `.comp` kernels have **no CUDA implementation**. `caps.cooperativeMatrix
  = false` on CUDA ⇒ LTX runs 2.38 s/step vs 1.21 on Vulkan fp16+coopmat.

## Target architecture

### File layout (one native source next to each shader)

```
operators/impl/matmul/
  MatMulTiledReg.shader      # HLSL template (Vulkan)
  MatMulTiledReg.cu          # native CUDA counterpart  <-- NEW
  MatMulCoopMatTiled.comp    # GLSL coopmat (Vulkan)
  MatMulCoopMatTiled.cu      # native WMMA counterpart  <-- NEW
  MatMulCommon.shaderh       # shared HLSL
  MatMulCommon.cuh           # shared CUDA              <-- NEW (as needed)
  shaders.json               # unchanged; drives BOTH paths
```

### Native kernel contract (must match the Vulkan counterpart)

1. Entry point: `extern "C" __global__ void cut_main(...)`.
2. Parameters: one pointer per storage buffer **in `[[vk::binding(N,0)]]`
   order** (`const T*` for `StructuredBuffer`, `T*` for `RWStructuredBuffer`),
   followed by the push-constant struct **by value** (same layout as HLSL).
3. Block geometry: identical to the shader's `[numthreads(...)]` (launch block
   dims come from SPIR-V reflection; grid dims from the op-layer dispatch).
   Same per-block tile coverage. Freedom *within* the block (warp shuffles,
   different shared-memory strategy) is encouraged.
4. Spec constants: `CUT_SPEC_<id>` macros with `#ifndef` defaults; runtime
   passes actual values from the dispatched SPIR-V.
5. Dtype selection: **no `%PLACEHOLDER%` preprocessing for .cu files.** One
   source per kernel; per-variant NVRTC defines (emitted by the generator into
   the native manifest) mirror the shader scheme:
   - `CUT_VEC_DTYPE_<SLOT>` (e.g. `float4`/`half4`/`int4`/`uint4`)
   - `CUT_SCALAR_DTYPE_<SLOT>`, `CUT_DTYPE_SIZE_<SLOT>`
   - `CUT_DTYPE_<SLOT>_IS_FLOAT / _IS_HALF / _IS_INT / _IS_UINT / _IS_INT8`
   - `shaders.json` variant `defines` are passed verbatim (`-DKEY=VAL`);
     kernels should provide `#ifndef KEY` fallbacks for standalone reading.
6. `cut_cuda_prelude.cuh` and `ComputeOpsShared.h` are always available as
   NVRTC headers, plus every `*.cuh` in the kernel's impl directory.
7. Innermost-dimension alignment: the least significant dimension is padded to
   a multiple of 4 elements (project-wide `shape_` convention) — identical to
   the HLSL kernels.

### Keying and lookup (runtime unchanged for ops)

The normalized-SPIR-V-hash keying stays: ops still dispatch SPIR-V; the CUDA
backend resolves hash → kernel. The registry entry grows:

```cpp
struct CudaKernelEntry {
  uint64_t hash;        // normalized SPIR-V hash
  const char *name;     // "{Func}_{Dtype...}" variant stem
  const char *source;   // CUDA C++ source
  const char *entry;    // "cut_main"
  const char *defines;  // space-separated KEY=VAL pairs (dtype + variant defines)
  bool native;          // hand-authored (.cu) vs transpiled fallback
};
const CudaKernelEntry *lookupCudaKernelByHash(uint64_t hash);
const CudaKernelEntry *lookupCudaKernelByName(const char *name);  // NEW
```

- **Native wins over transpiled** when both exist for a hash. The transpiled
  path remains as fallback for any `.shader` without a `.cu` yet (build prints
  a coverage report), so new Vulkan shaders never silently lose CUDA support.
- `CUT_CUDA_KERNELS=transpiled` env var forces the transpiled table (A/B
  debugging / numeric bisection). Default: native.
- Embedded `.cuh` headers are passed to every NVRTC compile alongside the
  prelude and enums.

### Capability flips (the actual perf win)

In `CudaCompute` constructor, gated on compute capability **and** the presence
of the corresponding native kernels (via `lookupCudaKernelByName`):

- `caps_.cooperativeMatrix = true` (CC ≥ 7.0) once the WMMA ports of the 6
  `.comp` coopmat kernels are registered — the op layer then selects coopmat
  variants on CUDA and the LTX runner auto-enables fp16 activations
  (`subgroupSize` is already 32).
- `caps_.integerDotProduct = true` (CC ≥ 6.1) once the `__dp4a` ports of the
  Q8 dot kernels are registered.

### Build pipeline changes

- `generate_shader_variants.py`: for each group whose dir contains `X.cu`
  matching a variant's template/source shader, emit
  `generated_cuda/native_manifest.json`: stem → {cu path, entry, defines list,
  cuh headers}. No preprocessing of `.cu` contents.
- `embed_cuda_kernels.py`: consume the native manifest + transpiler manifest;
  embed native sources (deduped — one string per `.cu`, shared by its
  variants) and transpiled fallbacks; emit both tables + header table into
  `CompiledCudaKernels.cpp`; print coverage (native / transpiled-only / none).
- `cmake/shader_loader.cmake`: add `.cu`/`.cuh` files to the embed step's
  DEPENDS so edits retrigger embedding.
- `scripts/codegen/check_cuda_kernels.py` (NEW): offline gate that NVRTC-compiles
  every embedded entry (default spec values + its defines) using the venv's
  `cuda.bindings.nvrtc`; no GPU memory needed. Run after any kernel change —
  this kills the "NVRTC failure ⇒ silent dispatch skip ⇒ zero outputs"
  failure mode at build time instead of runtime.

## Execution phases

- **Phase 0 — infrastructure** (sequential): generator + embedder + registry +
  `CudaContainers` defines/header plumbing + env knob + compile-check tool +
  CMake deps. Gate: existing transpiled path still passes the C++ suite.
- **Phase 1 — per-group native kernels** (parallel agents, ~10 batches):
  element-wise families first (binary/unary/cast/creation/ternary/memory/
  transpose), then reductions/softmax/rmsnorm/scan/sort (warp-shuffle
  rewrites), conv/pool/dequant, matmul scalar+gemv, matmul quantized,
  attention+rope. Gate per batch: `check_cuda_kernels.py` clean for the
  group's variants + review against the transpiled reference.
- **Phase 2 — CUDA-only capability**: WMMA ports of the 6 coopmat `.comp`
  kernels (64×64 block, BK=32, 4 warps × 2×2 `wmma` 16×16 tiles, double-
  buffered shared staging — mirrors the tuned Vulkan design), native
  `FusionBinaryOp`/`FusionUnary`, `__dp4a` Q8 dot kernels, caps flips.
- **Phase 3 — verification**: stop Ollama (VRAM), Release rebuild
  (`build-cuda-rel`), C++ suite on `CUT_DEVICES=cuda:0`, cross-backend LTX
  latent parity vs Vulkan baseline, SmolLM2 token-ID diff, LTX s/step
  benchmark (target: ≤ 1.21 s/step Vulkan number once coopmat caps flip).

## Risks / notes

- devstral (~15 GB VRAM) and GPU test runs conflict on the 3090 — codegen and
  runtime verification are separate phases (`ollama stop devstral-small-2:24b`
  before benchmarks).
- Kernels the transpiler already handles are memory-bound element-wise ops;
  native ports there are about **maintainability + dropping the transpiler
  long-term**, not speed. The measured wins are coopmat/WMMA, attention, and
  warp-level reductions.
- Native kernels keyed by SPIR-V hash inherit the Vulkan launch geometry; a
  kernel wanting different geometry must be added as a new variant in
  `shaders.json` (dispatch tables handle selection), not by diverging from
  the contract.
- `build-cuda/` is stale Debug+ASAN; use `build-cuda-rel/` for anything
  timed.
