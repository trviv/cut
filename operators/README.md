# CUT Shader Generation Architecture

This directory contains the modular shader generation system for CUT's Vulkan compute backend.

## File Organization

### Core Files

**[ShaderUtils.h](ShaderUtils.h)** / **[ShaderUtils.cpp](ShaderUtils.cpp)** - Template-based shader generation system
- GLSL shader templates that call `opFunc()` for operations
- Operation function templates (`opFuncBinaryOp`, `opFuncUnary`, etc.)
- String manipulation utilities (`replaceAll`, `applyDatatypeSubstitutions`)
- Datatype conversion functions (`getGLSLType`, `getGLSLScalarType`)
- opFunc generators (`getOpFuncBinaryOp`, `getOpFuncUnary`, etc.)
- Shader assembly functions (compose header + opFunc + template)
- High-level shader generators (call opFunc generator + assembler)
- Helper functions to reduce boilerplate (`generateBitwiseVecVec`, etc.)

**[ShadersGenerated.cpp](ShadersGenerated.cpp)** - Main entry point
- Coordinates shader generation across specialized files
- Implements SPIR-V shader caching
- Delegates to BasicOps and AdvancedOps

### Operator Implementation Files

**[ShadersBasicOps.cpp](ShadersBasicOps.cpp)** - Binary and unary operators (~391 lines)
- Binary vec-vec operations (arithmetic, comparison, min/max)
- Binary vec-scalar operations (arithmetic, comparison, min/max)
- Unary operations (math, trigonometric, rounding, activation functions)
- Ternary clamp operation
- Uses high-level generators from ShaderUtils for simple, readable code

**[ShadersAdvancedOps.cpp](ShadersAdvancedOps.cpp)** - Advanced operators (~560 lines)
- Extended binary operations (bitwise, logical, special math)
- Reduction operations (sum, mean, min, max, prod, any, all)
- Matrix operations (matmul, transpose, dot)
- Conditional selection (where/select)
- Tensor creation (arange, linspace, zeros, ones, full)
- Norm operations
- Extended unary operations (isnan, isinf)
- Uses helper functions to reduce boilerplate (90+ lines saved)

### Pre-compiled Shaders

**[CompiledShaders.cpp](CompiledShaders.cpp)** - Pre-compiled SPIR-V shaders
- Contains frequently-used shaders compiled at build time
- Reduces runtime compilation overhead

**[Shaders.cpp](Shaders.cpp)** / **[Shaders.h](Shaders.h)** - Shader API
- Public interface for shader compilation and retrieval
- SPIR-V compilation using shaderc

## Architecture Benefits

### Before Refactoring
- **Single monolithic file**: 1,414 lines in ShadersGenerated.cpp
- **Hard to navigate**: All operators in one giant switch statement
- **Code duplication**: Template code repeated throughout
- **Difficult maintenance**: Finding specific operators was time-consuming

### After Template-based Refactoring
- **Modular structure**: 4 focused files with clean separation
  - ShaderUtils.h: 83 lines (public API declarations)
  - ShaderUtils.cpp: 578 lines (template-based generation engine)
  - ShadersGenerated.cpp: 81 lines (coordination & caching)
  - ShadersBasicOps.cpp: 391 lines (basic operations)
  - ShadersAdvancedOps.cpp: 560 lines (advanced operations)
- **Template-based architecture**: All shaders follow the `opFunc()` pattern
  - Shader templates define data flow (read → opFunc → write)
  - Operation templates define compute logic (e.g., `return a + b`)
  - Composition happens via string assembly, not runtime concatenation
- **Minimal duplication**: Helper functions eliminate 140+ lines of boilerplate
  - `generateBitwiseVecVec()` replaces 10 operations × 9 lines each
  - `generateBinaryVecVecCustom()` simplifies custom expressions
- **Easy to extend**: Adding new operations is straightforward
  - Simple ops: One line calling a generator function
  - Complex ops: Custom shader with `opFunc()` pattern
- **Better maintainability**: Clear architecture, easy to find and modify operators

## Adding New Operators

### 1. For Simple Arithmetic/Math Operations
Add to **ShadersBasicOps.cpp** using high-level generators:
```cpp
case MyNewBinaryOp:
  shaderSource = generateBinaryVecVecOpShader("+", datatype);  // For operators
  shaderName = "my_new_binary_op";
  return true;

case MyNewUnaryOp:
  shaderSource = generateUnaryShader("sqrt(a)", datatype);  // For expressions
  shaderName = "my_new_unary_op";
  return true;
```

### 2. For Bitwise Operations
Add to **ShadersAdvancedOps.cpp** using bitwise helpers:
```cpp
case MyBitwiseOp:
  shaderSource = generateBitwiseVecVec("&", datatype);  // Handles floatBitsToInt conversions
  shaderName = "my_bitwise_op";
  return true;
```

### 3. For Custom Expressions
Add to **ShadersAdvancedOps.cpp** using custom helpers:
```cpp
case MyCustomOp: {
  std::string expr = "mix(a, b, " + vecType + "(greaterThan(a, " + vecType + "(0.0))))";
  shaderSource = generateBinaryVecVecCustom(expr.c_str(), datatype);
  shaderName = "my_custom_op";
  return true;
}
```

### 4. For Complex Operations with Custom Shader
Add to **ShadersAdvancedOps.cpp** with full shader template:
```cpp
case MyComplexOp: {
  std::string shader = R"(#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint dtype_vec_size = 4;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOut {
    %VEC_DTYPE% dataOut[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) return;

    // Custom computation here
    dataOut[index] = /* your logic */;
}
)";
  shaderSource = applyDatatypeSubstitutions(shader, datatype);
  shaderName = "my_complex_op";
  return true;
}
```

### 5. For New Helper Functions
Add to **ShaderUtils.h** (declaration) and **ShaderUtils.cpp** (implementation):
```cpp
// In ShaderUtils.h
std::string generateMyHelper(const char *param, DataType datatype);

// In ShaderUtils.cpp (at end of file, before closing namespace)
std::string generateMyHelper(const char *param, DataType datatype) {
  std::string expr = /* build expression using param */;
  return generateBinaryVecVecCustom(expr.c_str(), datatype);
}
```

## Template-based Architecture: The opFunc Pattern

All element-wise shaders follow a consistent architecture that separates **data flow** from **computation logic**:

### Shader Structure
```glsl
#version 450
// ... Standard header with workgroup size, constants ...

layout(push_constant) uniform PushConstants {
    uint numElements;
};

// ┌─────────────────────────────────────────────────────────┐
// │ Operation Function (opFunc) - Customized per operation │
// └─────────────────────────────────────────────────────────┘
vec4 opFunc(vec4 a, vec4 b) {
    return a + b;  // ← This is the only part that changes!
}

// ┌─────────────────────────────────────────────────────────┐
// │ Buffer Layouts - Same for all binary vec-vec ops       │
// └─────────────────────────────────────────────────────────┘
layout(...) buffer BufferA { vec4 dataA[]; };
layout(...) buffer BufferB { vec4 dataB[]; };
layout(...) buffer BufferOut { vec4 dataOut[]; };

// ┌─────────────────────────────────────────────────────────┐
// │ Main Template - Handles data flow (same for all ops)   │
// └─────────────────────────────────────────────────────────┘
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index * dtype_vec_size >= numElements) return;

    vec4 a = dataA[index];      // Read inputs
    vec4 b = dataB[index];
    dataOut[index] = opFunc(a, b);  // Call operation, write result
}
```

### How It Works

1. **Operation Templates** define what each operation does:
   ```cpp
   // In ShaderUtils.cpp
   const char *opFuncBinaryOp = R"(
   %VEC_DTYPE% opFunc(%VEC_DTYPE% a, %VEC_DTYPE% b) {
       return a %OP% b;
   })";
   ```

2. **Shader Templates** define data flow (same for all operations):
   ```cpp
   const char *templateBinaryVecVec = R"(
   // ... buffers ...
   void main() {
       // ... read data ...
       dataOut[index] = opFunc(a, b);  // ← Calls the operation
   })";
   ```

3. **Assembly** combines them:
   ```cpp
   std::string generateBinaryVecVecOpShader(const char *op, DataType datatype) {
     std::string opFunc = getOpFuncBinaryOp(op, datatype);  // Generate opFunc
     return assembleBinaryVecVecShader(opFunc, datatype);    // Combine with template
   }
   ```

### Benefits

- **Consistency**: All shaders follow the same pattern
- **Simplicity**: Only the operation logic changes between shaders
- **Testability**: opFunc can be tested independently
- **Readability**: Clear separation of concerns
- **Performance**: No runtime overhead—everything is compile-time string assembly

## Build System Integration

The shader files are included in [CMakeLists.txt](../CMakeLists.txt):

```cmake
set(LIB_HEADERS
    ...
    shaders/Shaders.h
    shaders/ShaderUtils.h
    ...
)

set(LIB_SOURCES
    ...
    shaders/Shaders.cpp
    shaders/ShadersGenerated.cpp
    shaders/ShadersBasicOps.cpp
    shaders/ShadersAdvancedOps.cpp
    shaders/CompiledShaders.cpp
    ...
)
```

## Shader Generation Flow

```
User calls operator (e.g., add, matmul)
         ↓
getGeneratedShader(op, datatype)
         ↓
Check cache → Return if found
         ↓
Try generateBasicOpShader() → Found? → Compile to SPIR-V → Cache → Return
         ↓
Try generateAdvancedOpShader() → Found? → Compile to SPIR-V → Cache → Return
         ↓
Return std::nullopt (operator not supported)
```

## Performance Considerations

- **Shader Caching**: Generated SPIR-V is cached by (operator, datatype) key
- **Template-based Generation**: Fast string assembly from pre-defined templates
- **Vec4 Vectorization**: All element-wise operations process 4 elements per thread
- **Workgroup Size**: Optimized at 256 threads per workgroup for GPU occupancy
- **Parallel Reductions**: Shared memory tree reduction for aggregation operations

## Testing

All operators are automatically tested through the benchmark suite:
```bash
cd interface/python/benchmarks
python run_benchmarks.py
```

This verifies correctness against NumPy and measures performance vs CuPy, JAX, and PyTorch.
