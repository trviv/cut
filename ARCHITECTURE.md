# CUT — Architecture & Structural Reference

CUT (Compute Unified Toolkit) is a Vulkan-based GPU compute toolkit with a PyTorch-like Python API. It ships 163+ operators across element-wise math, reductions, linear algebra, activations, sorting, and attention — backed by precompiled HLSL/GLSL kernels and a graph-execution runtime, with LLM inference (GGUF, llama-family) as the primary showcase application.

---

## Layered Architecture

```
Python API  (interface/python/cut/)
    │  pybind11
    ▼
Operations  (runtime/Operations.h/cpp)     ── high-level tensor ops
    │
    ▼
Runtime     (runtime/Runtime.h/cpp)         ── lifecycle, caching, tensor management
    │
    ▼
Graph       (runtime/graph/)                ── op graph, optimizer, memory planner, executor
    │
    ▼
Dispatcher  (runtime/Dispatcher.h/cpp)      ── GPU command encoding
    │
    ▼
Operators   (operators/impl/ + operators/runtime/)  ── per-op classes + shader registry
    │
    ▼
ComputeInterface  (core/api/)               ── abstract compute API
    │
    ▼
VulkanCompute     (core/backends/vulkan/)   ── Vulkan implementation
    │
    ▼
GPU  (Vulkan; MoltenVK on macOS, native on Linux)
```

---

## Directory Layout

```
cut/
├── ARCHITECTURE.md           # This file
├── README.md
├── LICENSE
├── CMakeLists.txt            # Root build configuration
├── .clang-format             # C/C++ style config
├── cmake/                    # Build helpers
│   ├── shader_loader.cmake   # Shader compilation pipeline (DXC + glslc)
│   └── lsan.supp             # LeakSanitizer suppressions
├── core/
│   ├── api/
│   │   ├── include/          # Public headers (ComputeInterface, Handle, Container, Structs)
│   │   └── src/              # Implementations
│   └── backends/
│       └── vulkan/           # Vulkan backend (Compute, CommandBuffer, Containers, Structs, Staging)
├── runtime/
│   ├── Runtime.h/cpp         # Central coordinator
│   ├── Operations.h/cpp      # High-level tensor operations (consumed by Python)
│   ├── Dispatcher.h/cpp      # GPU command encoding
│   ├── TensorStore.h/cpp     # Tensor allocation + buffer-view recycling
│   ├── ShapeUtils.h/cpp      # Shape arithmetic helpers
│   ├── VariantSelector.h/cpp # Picks best shader variant for dispatch size
│   └── graph/                # Graph IR
│       ├── Graph.h/cpp
│       ├── GraphBuilder.h/cpp
│       ├── GraphExecutor.h/cpp
│       ├── GraphOptimizer.h/cpp  # Fusion rules (matmul+silu, fused binary, etc.)
│       ├── MemoryPlanner.h/cpp   # Buffer reuse across graph nodes
│       └── GraphReport.h/cpp     # Diagnostic dump
├── operators/
│   ├── runtime/              # Shader infrastructure (used by all op families)
│   │   ├── Shaders.h/cpp     # Shader registry + lookup API
│   │   ├── ShaderUtils.h/cpp # Compilation helpers
│   │   ├── CompiledShaders.cpp  # Generated SPIR-V blobs (build artifact)
│   │   ├── ComputeOps.h/cpp     # OperatorEnum and Operator metadata
│   │   ├── ComputeOpsShared.h   # Shared by C++ and HLSL/GLSL shaders
│   │   └── OpNode.h/cpp      # Base class for all operator nodes
│   └── impl/                 # Per-op-family implementations
│       ├── matmul/           # MatMulOp + 20+ shader variants (Q4, Q8, CoopMat, GEMV, …)
│       ├── binary/           # BinaryOp + fused binary ops + shader variants
│       ├── unary/            # UnaryOp + 60+ unary ops in one shader
│       ├── reduce/           # ReduceOp + ReduceRMS/Variance/LogSumExp variants
│       ├── reducedim/        # ReduceDimOp (per-dimension reduction)
│       ├── conv1d/, conv2d/  # Convolution ops with tiled variants
│       ├── attention/        # Attention + FusedAttention + BatchedFusedAttention
│       ├── rmsnorm/          # RMSNorm + ExtendedRMSNorm
│       ├── softmax/          # Softmax + LogSoftmax
│       ├── dequant/          # GPU dequant for BF16, Q4_K, Q5_K, Q6_K
│       ├── memory/           # Copy, Pad, Embedding, Expand
│       ├── transpose/        # Naive, tiled, vec4 variants
│       ├── sort/             # Bitonic + radix
│       ├── scan/             # Per-wg + propagate prefix scan
│       ├── ternary/          # Clamp, Select
│       ├── cast/             # Tensor dtype conversion
│       ├── creation/         # Arange, Fill
│       ├── rope/             # Rotary position embedding
│       ├── sampling/         # Repetition penalty
│       ├── q4transpose/      # Nibble-aware Q4 transpose
│       └── pool/, avgpool2d/, maxpool2d/, scan/
├── interface/
│   ├── loader/
│   │   ├── gguf/             # GGUF model reader + dequant
│   │   └── safetensor/       # SafeTensor model reader
│   ├── runner/llama/         # Llama-family model runner (example.cpp, server, client)
│   └── python/
│       ├── cut/              # Python package (compute.py, _ops.py, _compute_bindings.cpp)
│       ├── tests/            # pytest suite
│       └── benchmarks/       # Performance benchmarks vs PyTorch/NumPy/JAX/CuPy
├── examples/                 # Standalone inference scripts (gguf, safetensor, llm)
├── tests/                    # C++ Google Test suite
│   ├── api/, runtime/, vulkan/, combined/, graph/
├── benchmarks/               # C++ benchmarks (end-to-end, per-operator)
│   ├── autotune/             # Variant sweeps that build the dispatch table
│   └── vendor/               # CUT vs cuBLAS/rocBLAS on crucial operations
├── docs/                     # Reference docs (operators.md, pytorch_comparison.md)
└── scripts/                  # Build, format, benchmark helpers
```

---

## Core (`core/api/`)

The abstract compute foundation. All types here are backend-agnostic.

### ComputeInterface (`core/api/include/ComputeInterface.h`)

Abstract base class defining the GPU compute contract:

- `createBuffer(shape, dtype, data, isUniform)` — allocate a GPU buffer
- `copyDataToBuffer()` / `copyDataFromBuffer()` — host ↔ GPU transfers
- `createShaderModule(spirv)` — load compiled SPIR-V
- `encode(ComputeDispatch)` — record a dispatch to the active command buffer
- `submit()` — end recording and submit to GPU
- `wait(handle)` — block until completion

The interface manages an `activeCommandBuffer_` internally. Calling `encode()` lazily creates one; `submit()` finalises and submits it, returning a handle for `wait()`.

### ComputeHandle (`core/api/include/ComputeHandle.h`)

Opaque, ref-counted handle for accessing GPU objects.

- Stores a `ComputeContainer*` and a `size_t id_`
- Copy increments refcount; move transfers ownership; destructor decrements
- `as<T>()` provides type-safe casting with runtime container verification
- When refcount reaches 0, the container calls `destroy()` and pushes the id to a free-list for reuse

### ComputeContainer (`core/api/include/ComputeContainer.h`)

Template base managing handle lifecycle and object storage.

- `ComputeContainer` — base with `refCounts_` vector and `freeHandles_` stack
- `ComputeDataContainer<T>` — typed extension storing objects in a vector
  - `create(T&&)` — allocates a slot (reusing freed ids) and stores the object
  - `get(handle)` — retrieves by handle with verification
  - `destroyAPIObject()` — virtual hook for GPU resource cleanup

### ComputeBuffer (`core/api/include/ComputeStructs.h`)

Tensor-like buffer metadata:

- `shape_` — up to 4 dimensions (padded to 4D)
- `dtype_` — `Float32`, `Float16`, `UInt32`, `Int32`
- `size_` — aligned buffer size in bytes
- `executionElementCount_` — padded element count for dispatch sizing

**Alignment strategy:** The innermost dimension is rounded up to a multiple of 4 (for vec4 GPU access). Total size is then 16-byte aligned. The runtime handles transparent padding insertion/removal during host copies via `copyActualToAligned()` and `copyAlignedToActual()`.

### ComputeBinding (`core/api/include/ComputeStructs.h`)

Wrapper for shader resource binding. Can hold either:

- A `ComputeHandle` (reference to a GPU buffer/texture)
- A `std::vector<uint8_t>` (owned inline data for push constants)

### ComputeDispatch (`core/api/include/ComputeStructs.h`)

Represents a single compute shader invocation:

- `shader_` — handle to compiled shader
- `wgSize_` — workgroup dimensions `{x, y, z}`
- `bindings_` — vector of `ComputeBinding` objects
- Methods: `bindShader()`, `bindResource()`, `bindData()`, `bindValue<T>()`
- Static `createBarrier()` — inserts a memory barrier between dispatches

### CommandBuffer (`core/api/include/ComputeStructs.h`)

Abstract base for recording dispatches:

- `begin()` / `end()` — recording lifecycle
- `encode(ComputeDispatch&&)` — appends a dispatch
- `submit()` / `wait()` — execution control

### OperatorEnum (`core/api/include/ComputeOps.h`)

Comprehensive enum of 100+ operator types, grouped by category:

| Range   | Category                |
|---------|-------------------------|
| 0–29    | Binary vec-vec          |
| 30–59   | Binary vec-scalar       |
| 60–96   | Unary                   |
| 100–109 | Ternary                 |
| 110–119 | Reduction (scalar)      |
| 120–139 | Matrix / tensor         |
| 160–199 | Extended unary          |
| 220–223 | Argmax / argmin         |
| 240–261 | Cumulative / prefix scan|
| 270–271 | Sort (bitonic / radix)  |
| 280–299 | Internal (multi-pass helpers) |

`ComputeOpsShared.h` provides parallel `#define` declarations so the same constants are available inside GLSL shaders.

### Shader Compilation & Reflection (`core/api/src/ComputeCommon.cpp`)

**Compilation:** GLSL source → SPIR-V via shaderc (`compileShaderToSpirv()`).

**Reflection:** Two-pass SPIR-V parser (`reflectSpirvBindings()`):

1. **Pass 1** — Collect decorations: binding indices, descriptor sets, access qualifiers, type info, specialisation constants, workgroup size from `OpExecutionMode`
2. **Pass 2** — Match `OpVariable` instructions to bindings, infer binding type from storage class (`StorageBuffer`, `UniformBuffer`, `SampledImage`, `StorageImage`, `Sampler`)

Output: `ShaderReflection` containing all `BindingInfo` entries, workgroup size, push constant size, and dtype vec size.

---

## Vulkan Backend (`core/backends/vulkan/`)

### VulkanCompute (`VulkanCompute.h/cpp`)

Concrete `ComputeInterface` implementation:

1. Picks physical device by type
2. Creates logical device with a compute queue
3. Queries device properties and memory types
4. Initialises VMA allocator (if enabled)
5. Creates typed containers for all GPU resources
6. Creates the command buffer container

`VulkanInstance` wraps the `VkInstance` and debug messenger, and provides `createInterface()` as a factory.

### VulkanCommandBuffer (`VulkanCommandBuffer.h/cpp`)

Implements `CommandBuffer` for Vulkan:

- `begin()` — `vkBeginCommandBuffer` with `ONE_TIME_SUBMIT`
- `end()` — creates descriptor sets and pipelines from recorded dispatches, records `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdDispatch`, then `vkEndCommandBuffer`
- `submit()` — `vkQueueSubmit` with fence
- `wait()` — `vkWaitForFences`

Pipeline and descriptor creation is **deferred to `end()`**, enabling batch optimisation before submission.

### VulkanContainers (`VulkanContainers.h/cpp`)

Typed containers with built-in caching:

| Container | Caches by |
|-----------|-----------|
| `VulkanDescriptorSetLayoutContainer` | Binding signature |
| `VulkanDescriptorContainer` | Layout handle |
| `VulkanPipelineContainer` | Shader + layout pair (plus Vulkan pipeline cache) |
| `VulkanBufferContainer` | — (no caching, direct allocation) |
| `VulkanShaderContainer` | — (caching done at Runtime level) |

### VulkanBufferStruct (`VulkanStructs.h`)

Extends `ComputeBuffer` with:

- `VkBuffer buffer` / `VkDeviceMemory memory` (or `VmaAllocation`)
- `offset` — buffer sub-allocation offset
- `isCoherent` — memory coherency flag
- `data` — host-visible pointer (`nullptr` for device-only buffers)

---

## Operators (`operators/`)

Shaders are authored as standalone **HLSL** (`*.shader`) or **GLSL** (`*.comp`)
files under `operators/impl/<family>/` and compiled to SPIR-V **at build time**.
The build system pre-generates per-datatype variants, embeds the SPIR-V into
`operators/runtime/CompiledShaders.cpp`, and the runtime selects a variant by
key at dispatch time.

### `operators/runtime/` — Shader Infrastructure

| File | Purpose |
|------|---------|
| `Shaders.h/cpp` | Shader registry; `getShader(OperatorEnum, DataType, variant)` returns SPIR-V |
| `CompiledShaders.cpp` | Build-generated; embeds every (op, dtype, variant) SPIR-V blob |
| `ShaderUtils.h/cpp` | Helpers used by `OpNode` subclasses (dispatch sizing, push-constant packing) |
| `ComputeOps.h` | `OperatorEnum` and operator metadata table |
| `ComputeOpsShared.h` | Constants shared between C++ and shader source (`#include`d from both) |
| `OpNode.h/cpp` | Abstract base class. Each per-op subclass returns its shader, push constants, and execution size |

### `operators/impl/<family>/` — Per-Op Implementations

Each operator family lives in its own subdirectory containing:

- `<Name>Op.h/cpp`: the C++ class derived from `OpNode`
- `<Name>.shader` / `<Name>.comp`: one or more compute shader source files
- `<Name>Common.shaderh`: shader-side header (shared snippets)
- `<Name>Variants.generated.h`: build-generated variant registry
- `<Name>Shaders.generated.h`: build-generated SPIR-V binding helper
- `shaders.json`: variant manifest (dtypes, function names, slot names)

### Build-Time Shader Pipeline

```
operators/impl/<family>/<Name>.shader  (HLSL with #include "ComputeOpsShared.h")
         │
         ▼  scripts/codegen/generate_shader_variants.py  ── reads shaders.json
         │   produces per-datatype variants:  <Name>_Float32.shader, <Name>_UInt32.shader, …
         ▼
   <build>/generated_shaders/<Name>_<Dtype>.shader
         │
         ▼  DXC  (HLSL → SPIR-V)            or  glslc  (GLSL → SPIR-V; cooperative-matrix shaders)
         │
         ▼
   <build>/operators/<Name>_<Dtype>.spv
         │
         ▼  CMake script (in cmake/shader_loader.cmake)
         │   reads every .spv, emits one `compiled<Name>(…)` function returning
         │   `std::optional<std::vector<uint32_t>>` selecting on dtype.
         ▼
operators/runtime/CompiledShaders.cpp   ← linked into CUTLib
```

A `.shader_cache/` keyed by source hash skips recompilation on clean builds.

### Dispatch Flow

```
runtime/Operations.cpp     :  caller picks an OpNode subclass (e.g. MatMulOp)
        │
        ▼
operators/impl/matmul/MatMulOp.cpp  : picks a *variant* using VariantSelector
        │                              (dispatch size + dtype + alignment)
        ▼
operators/runtime/Shaders.cpp       : looks up the compiled SPIR-V for that variant
        │
        ▼
runtime/Dispatcher.cpp              : encodes ComputeDispatch (shader, push constants,
        │                              bindings, workgroup size) into the active
        │                              command buffer
        ▼
core/backends/vulkan/VulkanCommandBuffer.cpp  :  vkCmdDispatch on submit
```

### Vectorisation

Each thread processes a **vec4** (4 elements) for element-wise ops. Supported dtype mappings:

| DataType | HLSL Vector | GLSL Vector | Scalar |
|----------|-------------|-------------|--------|
| Float32  | `float4`    | `vec4`      | `float`     |
| Float16  | `half4`     | `f16vec4`   | `float16_t` |
| UInt32   | `uint4`     | `uvec4`     | `uint`      |
| Int32    | `int4`      | `ivec4`     | `int`       |

### Adding a New Operator

1. Create `operators/impl/<family>/` with `<Name>Op.h/cpp`, one or more `.shader` files, and `shaders.json`.
2. Add the `OperatorEnum` value in `operators/runtime/ComputeOps.h`.
3. List the new sources in `CMakeLists.txt`.
4. Wire it through `runtime/Operations.cpp` and (optionally) the Python bindings.

The build system picks up the new `shaders.json` automatically and emits the per-dtype variant binaries.

---

## Runtime (`runtime/`)

### Runtime (`Runtime.h/cpp`)

Central coordinator managing lifecycle and dispatch:

- **Initialisation:** `init(BackendType::Vulkan)` → creates `VulkanInstance`, `VulkanCompute`, `Dispatcher`, and `Operations`
- **Tensor management:** `createTensor()`, `copyToTensor()`, `copyFromTensor()`
- **Shader caching:** `getOrCreateShader(op, dtype)` — keyed by `(OperatorEnum, DataType)`, compiles on first use
- **Execution size inference:** Reduction ops use actual element count (avoids padding artifacts); element-wise ops use aligned sizes
- **Lazy flush:** `pendingCommands_` flag; actual `submit()` + `wait()` deferred until data is read back via `copyFromTensor()` or explicit `flushPendingCommands()`
- **Shutdown:** Ordered teardown: operations → shader cache → dispatcher → interface → Vulkan instance

### Operations (`Operations.h/cpp`)

High-level tensor API consumed by the Python bindings:

| Category | Methods | Notes |
|----------|---------|-------|
| Element-wise | `binaryOp()`, `unaryOp()`, `binaryOp()` | Auto-allocates output buffer with matching shape/dtype |
| Reduction | `reduce(op, a, dim={})` | Returns Tensor; global (no dim) outputs shape `{1}`, dim-wise removes that dim |
| Matrix | `matmul(a,b)`, `transpose(a)`, `dot(a,b)` | Shape validation; push constants carry `{M, K, N}` |
| Ternary | `clamp(a, min, max)`, `where(cond, x, y)` | — |
| Cumulative | `cumOp(a, dim, op)` | Cumsum, cumprod along a dimension |
| Statistical | `varianceScalar()`, `varianceDim()` | Hybrid: GPU mean reduction + CPU variance |
| Softmax | `softmax(a, dim)`, `logSoftmax(a, dim)` | Hybrid: GPU max per dim + CPU normalisation for numerical stability |
| Creation | `arange()`, `linspace()`, `full()` | GPU-side tensor generation |
| Shape | `reshape()`, `squeeze()`, `unsqueeze()`, `flatten()`, `unflatten()`, `concat()` | Via copy shader with layout metadata |
| Sort | `sortBitonic()`, `sortRadix()` | Multi-pass GPU algorithms |
| Scan | `prefixScan(a, op)` | 1–3 pass parallel scan |

Helper utilities: `getShape()`, `getDtype()`, `shapeProduct()`, `computeDimParams(shape, dim)` decomposing a dimension into `{outerSize, reduceSize, innerSize}`.

### Dispatcher (`Dispatcher.h/cpp`)

GPU command encoding engine — the most complex runtime component.

#### Core `encode()` Method

```
encode(OperatorEnum, bindings, shader, executionSize, dtype)
    │
    ├─ Validate binding count per op category
    ├─ Extract data bindings → push constants
    ├─ Compute workgroup size
    ├─ Create ComputeDispatch(shader, wgSize, handleBindings)
    ├─ Bind push constant data at index = len(handleBindings)
    └─ iface_->encode(dispatch)
```

#### Push Constant Layouts

| Op Type | Push Constants |
|---------|---------------|
| Binary vec-scalar | `{scalar(4B), numElements(4B)}` |
| Ternary clamp | `{minVal(4B), maxVal(4B), numElements(4B)}` |
| Ternary select | `{numElements(4B)}` |
| Dimension reduction | `{outerSize, reduceSize, innerSize, inOuterStride, inReduceStride}` |
| MatMul | `{M, K, N}` |
| Transpose | `{M, N, strideIn, strideOut}` |
| Dot | `{count}` |
| Copy | `{srcAlignedInner, dstAlignedInner, actualInnerDim, numRows}` |

#### Workgroup Sizing

| Op Type | Workgroup Config |
|---------|-----------------|
| Element-wise | `{executionSize, 1, 1}` |
| Reduction | `{256, 1, 1}` fixed |
| MatMul | `{ceil(N/16), ceil(M/16), 1}` |
| Transpose | `{ceil(N/16), ceil(M/16), 1}` |
| Dot | `{ceil(count/256), 1, 1}` |

#### Multi-Pass Algorithms

**Multi-workgroup reduction** (triggered when elements > 65,536):

1. **Phase 1 — Partial reduce:** `256 × N` threads each process ~1024 elements → per-workgroup partial results in a temp buffer
2. **Barrier**
3. **Phase 2 — Final reduce:** Single 256-thread workgroup reduces partials → scalar output

Template placeholders (`%SCALAR_DTYPE%`, `%IDENTITY%`, `%REDUCE_OP%`) are substituted per reduction type.

**Prefix scan** (exclusive/inclusive):

| Elements | Passes |
|----------|--------|
| ≤ 256 | 1 — direct scan |
| > 256 | 3 — per-workgroup scan → scan partial sums → propagate |

**Bitonic sort:**

1. Pad to next power-of-2 with sentinel values (`FLT_MAX` / `0xFFFFFFFF`)
2. `O(log²N)` compare-and-swap passes with barriers between each step
3. Copy-back phase strips padding

**Radix sort:**

1. 8 passes (4 bits per pass × 32-bit keys)
2. Per pass: histogram → exclusive prefix scan → scatter
3. Ping-pong temp buffers; result lands in original buffers after 8 (even) passes

#### Temporary Buffer Pool

Multi-pass operations need working memory:

- `acquireTempBuffer(numElements, dtype)` — checks `tempBufferPool_` for a sufficiently sized buffer; allocates if none available
- `releaseTempBuffers()` — returns active buffers to pool for reuse
- Avoids repeated GPU allocation/deallocation overhead

---

## Data Flow Example

A binary add operation (`c = a + b`) traces through the full stack:

```
Python: c = a + b
  │
  ▼  pybind11
Operations::binaryOp(BinaryVecVecAdd, a_handle, b_handle)
  │  Creates output buffer with same shape/dtype
  ▼
Runtime::encodeOperator(BinaryVecVecAdd, [a, b, output], dtype)
  │  Resolves shader: BinaryOp::resolve(variant) → operators/runtime/Shaders.cpp
  │    → returns precompiled SPIR-V from CompiledShaders.cpp for (op, dtype)
  │  Computes executionSize = a.executionSize()
  ▼
Dispatcher::encode(BinaryVecVecAdd, bindings, shader, size, Float32)
  │  Validates: 3 bindings (input A, input B, output)
  │  Workgroup: {size, 1, 1}
  │  Push constants: {numElements}
  │  Creates ComputeDispatch and calls iface_->encode()
  ▼
VulkanCompute::encode(dispatch)
  │  Appends to activeCommandBuffer_
  ▼
  (pendingCommands_ = true)

  ...later, on c.tolist()...

Runtime::flushPendingCommands()
  │  CommandBuffer::end()   ← creates Vulkan descriptor sets + pipelines
  │  CommandBuffer::submit() ← vkQueueSubmit
  │  CommandBuffer::wait()   ← vkWaitForFences
  ▼
Runtime::copyFromTensor(output, hostPtr)
  │  copyAlignedToActual() strips padding
  ▼
Python receives result
```

---

## Key Design Decisions

### 1. Build-Time Shader Compilation with Per-Op Variants

Shaders live as standalone HLSL/GLSL files under `operators/impl/<family>/`. At build time, `scripts/codegen/generate_shader_variants.py` produces per-dtype variants, DXC/glslc compiles each to SPIR-V, and CMake emits `operators/runtime/CompiledShaders.cpp` with the embedded blobs. The runtime then performs a constant-time lookup by `(op, dtype, variant)` — no compilation on the hot path. A persistent `.shader_cache/` keyed by source hash skips rebuilds.

### 2. Handle-Based Resource Management

All GPU objects are accessed through opaque `ComputeHandle` values with automatic reference counting. When the last handle is released, the container calls `destroy()` and recycles the slot id via a free-list. This prevents dangling GPU references and avoids id fragmentation.

### 3. Deferred Pipeline Creation

Vulkan descriptor sets and compute pipelines are not created during dispatch recording. Instead, they are built in `CommandBuffer::end()`, just before submission. This enables batch optimisation and avoids creating pipelines that might never execute.

### 4. Multi-Level Caching

| Level | What | Keyed By |
|-------|------|----------|
| Build | Compiled SPIR-V on disk | Source-file hash (`.shader_cache/`) |
| Runtime | SPIR-V → `VkShaderModule` | `(OperatorEnum, DataType, Variant)` |
| VulkanContainers | Descriptor set layouts | Binding signature |
| VulkanContainers | Compute pipelines | Shader + layout pair |
| Vulkan | Pipeline cache | Vulkan-internal |
| Graph | Execution plan (topological order + memory plan) | Graph identity |

### 5. Vec4 Vectorisation with Transparent Alignment

The innermost tensor dimension is padded to a multiple of 4 for vec4 GPU access. Padding is inserted during `copyActualToAligned()` (host → GPU) and stripped during `copyAlignedToActual()` (GPU → host). User code never sees the padding.

### 6. Backend Abstraction

`ComputeInterface` defines the full GPU contract. Vulkan is the only backend today, but the abstraction supports adding Metal, CUDA, or WebGPU without changing the runtime, shader, or operations layers.

### 7. Hybrid GPU/CPU for Numerical Stability

Variance and softmax use a hybrid approach: GPU computes reductions (mean, max per dimension), then CPU performs the final normalisation. This avoids numerical overflow/underflow without sacrificing GPU parallelism for the bulk of the work.

---

## Build System

- **C++ build:** CMake 3.16+, C++17. Targets Linux (native Vulkan) and macOS (MoltenVK).
- **Dependencies:** Vulkan SDK (with shaderc + SPIRV-Tools + glslang), DXC (DirectX Shader Compiler) for HLSL, optional glslc for GLSL compute, pybind11, Google Test (auto-fetched).
- **Shader pipeline:** `cmake/shader_loader.cmake` runs `scripts/codegen/generate_shader_variants.py` to produce per-dtype variants, compiles them with DXC/glslc, and emits `operators/runtime/CompiledShaders.cpp` containing embedded SPIR-V.
- **Python build:** scikit-build-core + pybind11 → `_cut_compute` extension module.
- **Sanitisers:** AddressSanitizer / LeakSanitizer suppressions in `cmake/lsan.supp`.
- **Linux setup:** `scripts/setup/setup_linux.sh` installs apt packages, DXC, and the LunarG Vulkan SDK.

### Supported Data Types

| Enum | Size | GLSL Vec | GLSL Scalar |
|------|------|----------|-------------|
| `Float32` | 4B | `vec4` | `float` |
| `Float16` | 2B | `f16vec4` | `float16_t` |
| `UInt32` | 4B | `uvec4` | `uint` |
| `Int32` | 4B | `ivec4` | `int` |
