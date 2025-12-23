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
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

result = cut.vector_add(a, b)
print(result)  # [6.0, 8.0, 10.0, 12.0]
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

# Load shader
shader = cut.Shader(cut.ShaderEnum.VECTOR_ADD)

# Create and configure dispatch
num_elements = np.array([a.size], dtype=np.uint32)
workgroups = (a.size + 63) // 64

dispatch = cut.Dispatch(shader, thread_groups=(workgroups, 1, 1))
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
  - `spirv`: SPIR-V bytecode or `ShaderEnum` for built-in shaders
- `cut.Dispatch(shader, thread_groups)` - Dispatch configuration
  - `bind(resource, binding)`: Bind buffer or push constant data

### Functions

- `cut.run(*dispatches)` - Execute compute dispatches
- `cut.vector_add(a, b)` - Element-wise vector addition
- `cut.get_interface()` - Get the global VulkanCompute interface

### Built-in Shaders

- `cut.ShaderEnum.VECTOR_ADD` - Vector addition shader (64 threads/workgroup)
