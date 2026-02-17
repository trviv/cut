# CUT — Architecture & Structural Reference

CUT is a GPU-accelerated tensor compute library with a Vulkan backend and a PyTorch-like Python API. It supports 163+ operations across element-wise math, reductions, linear algebra, activations, and sorting — all driven by runtime-generated GLSL compute shaders compiled to SPIR-V.

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
Dispatcher  (runtime/Dispatcher.h/cpp)      ── GPU command encoding & multi-pass algorithms
    │
    ▼
ComputeInterface  (core/api/)              ── abstract compute API
    │
    ▼
VulkanCompute     (core/backends/vulkan/)  ── Vulkan implementation
    │
    ▼
Shader Generation (shaders/)               ── template-based GLSL → SPIR-V
    │
    ▼
GPU (Vulkan / MoltenVK on macOS)
```

---

## Directory Layout

```
cut/
├── core/
│   ├── api/
│   │   ├── include/          # Public headers (ComputeInterface, Handle, Container, Structs, Ops)
│   │   └── src/              # Implementations
│   └── backends/
│       └── vulkan/           # Vulkan backend (Compute, CommandBuffer, Containers, Structs)
├── runtime/
│   ├── Runtime.h/cpp         # Central coordinator
│   ├── Operations.h/cpp      # High-level tensor operations
│   └── Dispatcher.h/cpp      # GPU command encoding engine
├── shaders/
│   ├── ShaderUtils.h/cpp     # Template engine & shader templates
│   ├── Shaders.h/cpp         # Public shader API & validation
│   ├── ShadersGenerated.cpp  # Entry point with SPIR-V cache
│   ├── ShadersBasicOps.cpp   # Basic op generators (~30 ops)
│   ├── ShadersAdvancedOps.cpp# Advanced op generators (~40 ops)
│   └── CompiledShaders.cpp   # Pre-compiled SPIR-V cache
├── interface/
│   ├── python/
│   │   ├── cut/              # Python package (compute.py, _ops.py, _compute_bindings.cpp)
│   │   ├── tests/            # pytest suite
│   │   └── benchmarks/       # Performance benchmarks
│   └── loader/
│       ├── gguf/             # GGUF model loader
│       └── safetensor/       # SafeTensor model loader
├── tests/                    # C++ Google Test suite
├── CMakeLists.txt            # Root build configuration
└── setup.sh                  # macOS setup script
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

## Shader Generation (`shaders/`)

All shaders are **GLSL 4.50 compute shaders**, compiled to SPIR-V at runtime.

### The `opFunc` Pattern

Every element-wise shader follows the same template structure:

```
┌──────────────────────────────────────────────┐
│ 1. HEADER                                    │
│    #version 450                              │
│    layout(local_size_x = 256) in;            │
│    layout(constant_id = 0) const uint        │
│         dtype_vec_size = 4;                  │
├──────────────────────────────────────────────┤
│ 2. PUSH CONSTANTS                            │
│    Varies by op type:                        │
│    - {numElements}                           │
│    - {scalar, numElements}                   │
│    - {minVal, maxVal, numElements}            │
├──────────────────────────────────────────────┤
│ 3. opFunc()  ← ONLY THIS CHANGES            │
│    e.g. return a + b;                        │
│    e.g. return sin(a);                       │
├──────────────────────────────────────────────┤
│ 4. BUFFER LAYOUTS (standardised per type)    │
│    layout(binding=0) readonly  buffer A {};  │
│    layout(binding=1) readonly  buffer B {};  │
│    layout(binding=2) writeonly buffer Out {}; │
├──────────────────────────────────────────────┤
│ 5. main()  — read → opFunc → write          │
└──────────────────────────────────────────────┘
```

Data flow is **static** (defined in templates). Only the computation in `opFunc()` changes per operation.

### Shader Templates (`ShaderUtils.cpp`)

| Template | Workgroup | Shared Memory | Use |
|----------|-----------|---------------|-----|
| `templateBinaryVecVec` | 256×1×1 | None | `a ⊕ b` element-wise |
| `templateBinaryVecScalar` | 256×1×1 | None | `a ⊕ scalar` |
| `templateUnary` | 256×1×1 | None | `f(a)` |
| `templateTernaryClamp` | 256×1×1 | None | `clamp(a, min, max)` |
| `matmulShaderTemplate` | 16×16 | `float tileA/B[16][16]` | Tiled matrix multiply |
| `transposeShaderTemplate` | 16×16 | None | Dimension swap |
| `dotShaderTemplate` | 256×1×1 | `sharedData[256]` | Dot product with atomics |
| `reductionShaderTemplate` | 256×1×1 | `sharedData[256]` | Tree reduction |
| `reductionDimShaderTemplate` | 256×1×1 | `sharedData[256]` | Per-dimension reduction |
| `kPartialReduceTemplate` | 256×1×1 | `sharedData[256]` | Multi-workgroup phase 1 |
| `kFinalReduceTemplate` | 256×1×1 | `sharedData[256]` | Multi-workgroup phase 2 |
| `kScanPerWgTemplate` | 256×1×1 | `sharedData[256]` | Prefix scan per workgroup |
| `kScanPartialSumsTemplate` | 1×1×1 | None | Scan partial sums |
| `kScanPropagateTemplate` | 256×1×1 | None | Propagate prefix |
| `kBitonicStepTemplate` | 256×1×1 | None | Bitonic sort compare-swap |
| `kRadixHistogramTemplate` | 256×1×1 | `sharedHist[16]` | Radix sort histogram |
| `kRadixScatterTemplate` | 256×1×1 | None | Radix sort scatter |

### Vectorisation

Each thread processes a **vec4** (4 elements). Supported dtype mappings:

| DataType | Vector Type | Scalar Type |
|----------|-------------|-------------|
| Float32  | `vec4`      | `float`     |
| Float16  | `f16vec4`   | `float16_t` |
| UInt32   | `uvec4`     | `uint`      |
| Int32    | `ivec4`     | `int`       |

### Generation Flow

```
getGeneratedShader(OperatorEnum, DataType)
    │
    ├─ Cache hit? → return cached SPIR-V
    │
    ├─ Try generateBasicOpShader()     [ShadersBasicOps.cpp]
    │   └─ Lookup in OpEntry table → generator(op, arg, dtype) → GLSL
    │
    ├─ Try generateAdvancedOpShader()  [ShadersAdvancedOps.cpp]
    │   └─ Lookup in OpEntry table or special-case logic → GLSL
    │
    ├─ compileShaderToSpirv(glsl) → SPIR-V
    ├─ Cache result: shaderCache[key] = spirv
    └─ Return SPIR-V
```

Cache key: `makeCacheKey(OperatorEnum, DataType)` — a combined 64-bit value.

### Operation Dispatch Tables

Operations are registered as `OpEntry` structs:

```cpp
struct OpEntry {
    ShaderGenFn generator;  // e.g. generateBinaryVecVecOpShader
    const char *arg;        // e.g. "+" or "sin(a)"
    const char *name;       // e.g. "binary_vec_vec_add"
};
```

Adding a new element-wise op requires one table entry.

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
| Element-wise | `binaryOp()`, `unaryOp()`, `vecScalarOp()` | Auto-allocates output buffer with matching shape/dtype |
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
  │  Resolves shader: getOrCreateShader(BinaryVecVecAdd, Float32)
  │    → ShadersGenerated: header + "return a + b" opFunc + vec-vec template → SPIR-V
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

### 1. Runtime Shader Generation

Shaders are generated as GLSL and compiled to SPIR-V at runtime via shaderc, then cached. This enables 163+ ops × 4 dtypes without shipping thousands of precompiled binaries. The `opFunc` template pattern means adding a new element-wise op is a single table entry.

### 2. Handle-Based Resource Management

All GPU objects are accessed through opaque `ComputeHandle` values with automatic reference counting. When the last handle is released, the container calls `destroy()` and recycles the slot id via a free-list. This prevents dangling GPU references and avoids id fragmentation.

### 3. Deferred Pipeline Creation

Vulkan descriptor sets and compute pipelines are not created during dispatch recording. Instead, they are built in `CommandBuffer::end()`, just before submission. This enables batch optimisation and avoids creating pipelines that might never execute.

### 4. Multi-Level Caching

| Level | What | Keyed By |
|-------|------|----------|
| Runtime | SPIR-V bytecode | `(OperatorEnum, DataType)` |
| Dispatcher | Internal shader SPIR-V | GLSL source hash or op enum |
| VulkanContainers | Descriptor set layouts | Binding signature |
| VulkanContainers | Compute pipelines | Shader + layout pair |
| Vulkan | Pipeline cache | Vulkan-internal |

### 5. Vec4 Vectorisation with Transparent Alignment

The innermost tensor dimension is padded to a multiple of 4 for vec4 GPU access. Padding is inserted during `copyActualToAligned()` (host → GPU) and stripped during `copyAlignedToActual()` (GPU → host). User code never sees the padding.

### 6. Backend Abstraction

`ComputeInterface` defines the full GPU contract. Vulkan is the only backend today, but the abstraction supports adding Metal, CUDA, or WebGPU without changing the runtime, shader, or operations layers.

### 7. Hybrid GPU/CPU for Numerical Stability

Variance and softmax use a hybrid approach: GPU computes reductions (mean, max per dimension), then CPU performs the final normalisation. This avoids numerical overflow/underflow without sacrificing GPU parallelism for the bulk of the work.

---

## Build System

- **C++ build:** CMake 3.16+, C++17, targeting macOS with MoltenVK
- **Dependencies:** Vulkan SDK (with shaderc), pybind11, Google Test (auto-fetched)
- **Python build:** scikit-build-core + pybind11 → `_cut_compute` extension module
- **Sanitisers:** Optional AddressSanitizer and UBSan via `-DENABLE_ASAN=ON` / `-DENABLE_UBSAN=ON`

### Supported Data Types

| Enum | Size | GLSL Vec | GLSL Scalar |
|------|------|----------|-------------|
| `Float32` | 4B | `vec4` | `float` |
| `Float16` | 2B | `f16vec4` | `float16_t` |
| `UInt32` | 4B | `uvec4` | `uint` |
| `Int32` | 4B | `ivec4` | `int` |
