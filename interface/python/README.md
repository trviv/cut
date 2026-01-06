# CUT Python Library

Python bindings for the CUT (Compute Unified Toolkit) GPU compute library.

## Installation

```bash
cd cut/python
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-build-core pybind11 numpy
pip install --no-build-isolation -e .
```

### Requirements

- Python 3.8+
- Vulkan SDK installed (with VULKAN_SDK environment variable set)
- CMake 3.15+

## Usage

### High-Level API

```python
import numpy as np
import cut

# Vector addition on GPU
a = cut.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
b = cut.Buffer(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))

result = cut.add(a, b)
print(result.numpy())  # [6.0, 8.0, 10.0, 12.0]
```

### Low-Level API

```python
import numpy as np
import cut

# Create buffers (storage buffers by default)
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
buf_a = cut.Buffer(a)
buf_b = cut.Buffer(b)
buf_out = cut.Buffer(size=a.nbytes)

# Load shader (OperatorEnum or ShaderEnum alias)
shader = cut.Shader(cut.OperatorEnum.BinaryVecVecAdd)

# Create and configure dispatch
num_elements = a.size

dispatch = cut.Dispatch(shader, thread_groups=(num_elements, 1, 1))
dispatch.bind(buf_a, 0)      # binding 0: input A
dispatch.bind(buf_b, 1)      # binding 1: input B
dispatch.bind(buf_out, 2)    # binding 2: output
dispatch.bind(num_elements, 3)  # push constant

# Execute
cut.run(dispatch)

# Read results
result = buf_out.numpy()
print(result)  # [6.0, 8.0, 10.0, 12.0]
```

### Custom Shaders

```python
import struct
import cut

# Load custom SPIR-V shader
with open("my_shader.spv", "rb") as f:
    data = f.read()
    spirv = list(struct.unpack(f"{len(data)//4}I", data))

shader = cut.Shader(spirv)
```

## API Reference

### Classes

- `cut.Buffer(data=None, size=None, is_uniform=False)` - GPU buffer
  - `data`: NumPy array to initialize buffer with
  - `size`: Buffer size in bytes (if data not provided)
  - `is_uniform`: Create uniform buffer instead of storage buffer
- `cut.Shader(spirv)` - Compute shader module
  - `spirv`: SPIR-V bytecode or `OperatorEnum` for built-in shaders
- `cut.Dispatch(shader, thread_groups)` - Dispatch configuration
  - `bind(resource, binding)`: Bind buffer or push constant data

### Functions

- `cut.run(*dispatches)` - Execute compute dispatches
- `cut.add(a, b)` - Element-wise vector addition
- `cut.subtract(a, b)` - Element-wise subtraction
- `cut.multiply(a, b)` - Element-wise multiplication
- `cut.divide(a, b)` - Element-wise division
- `cut.get_interface()` - Get the global VulkanCompute interface

### Built-in Operators

- `cut.OperatorEnum.BinaryVecVecAdd` - Vector addition
- `cut.OperatorEnum.BinaryVecVecSub` - Vector subtraction
- `cut.OperatorEnum.BinaryVecVecMul` - Vector multiplication
- `cut.OperatorEnum.BinaryVecVecDiv` - Vector division

Note: `cut.ShaderEnum` is available as a backward-compatible alias for `cut.OperatorEnum`.
