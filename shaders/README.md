# CUT Shader Generation Architecture

This directory contains the modular shader generation system for CUT's Vulkan compute backend.

## File Organization

### Core Files

**[ShaderUtils.h](ShaderUtils.h)** - Reusable shader components and utilities
- GLSL shader templates (headers, push constants, buffers)
- String manipulation utilities
- Datatype conversion functions
- Shader assembly functions
- High-level shader generation helpers

**[ShadersGenerated.cpp](ShadersGenerated.cpp)** - Main entry point
- Coordinates shader generation across specialized files
- Implements SPIR-V shader caching
- Delegates to BasicOps and AdvancedOps

### Operator Implementation Files

**[ShadersBasicOps.cpp](ShadersBasicOps.cpp)** - Binary and unary operators (~400 lines)
- Binary vec-vec operations (arithmetic, comparison, min/max)
- Binary vec-scalar operations (arithmetic, comparison, min/max)
- Unary operations (math, trigonometric, rounding, activation functions)
- Ternary clamp operation

**[ShadersAdvancedOps.cpp](ShadersAdvancedOps.cpp)** - Advanced operators (~500 lines)
- Extended binary operations (bitwise, logical, special math)
- Reduction operations (sum, mean, min, max, prod, any, all)
- Matrix operations (matmul, transpose, dot)
- Conditional selection (where/select)
- Tensor creation (arange, linspace, zeros, ones, full)
- Norm operations
- Extended unary operations (isnan, isinf)

### Pre-compiled Shaders

**[CompiledShaders.cpp](CompiledShaders.cpp)** - Pre-compiled SPIR-V shaders
- Contains frequently-used shaders compiled at build time
- Reduces runtime compilation overhead

**[Shaders.cpp](Shaders.cpp)** / **[Shaders.h](Shaders.h)** - Shader API
- Public interface for shader compilation and retrieval
- SPIR-V compilation using shaderc

## Architecture Benefits

### Before Refactoring
- **Single monolithic file**: 1414 lines in ShadersGenerated.cpp
- **Hard to navigate**: All operators in one giant switch statement
- **Code duplication**: Template code repeated throughout
- **Difficult maintenance**: Finding specific operators was time-consuming

### After Refactoring
- **Modular structure**: 4 focused files
  - ShaderUtils.h: 481 lines (reusable components)
  - ShadersGenerated.cpp: 81 lines (coordination)
  - ShadersBasicOps.cpp: 441 lines (basic operations)
  - ShadersAdvancedOps.cpp: 672 lines (advanced operations)
- **Clear separation**: Basic vs advanced operations
- **Code reuse**: All templates centralized in ShaderUtils.h
- **Easy navigation**: Each file has a clear purpose
- **Better maintainability**: Easy to find and modify operators

## Adding New Operators

### 1. For Basic Element-wise Operations
Add to **ShadersBasicOps.cpp**:
```cpp
case MyNewOp:
  shaderSource = generateUnaryShader("myglslfunction(dataIn[index])", datatype);
  shaderName = "my_new_op";
  return true;
```

### 2. For Advanced Operations
Add to **ShadersAdvancedOps.cpp** with custom shader template:
```cpp
case MyComplexOp: {
  std::string shader = R"(#version 450
    // Custom GLSL shader code here
  )";
  shaderSource = applyDatatypeSubstitutions(shader, datatype);
  shaderName = "my_complex_op";
  return true;
}
```

### 3. For New Templates
Add to **ShaderUtils.h** for reuse across operators:
```cpp
static const char *myNewTemplate = R"(
  // Reusable GLSL template
)";
```

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
