"""
CUT (Compute Unified Toolkit) - GPU Compute Library

A Python library for GPU compute operations using Vulkan.
Automatically initializes a Vulkan instance on import.
"""

import numpy as np
from typing import Optional, Union, List
from . import _cut_core

__version__ = "0.1.0"

# Module-level Vulkan instance and interface (lazy initialization)
_instance: Optional[_cut_core.VulkanInstance] = None
_interface: Optional[_cut_core.VulkanCompute] = None


def _ensure_initialized():
    """Ensure Vulkan instance and interface are initialized."""
    global _instance, _interface
    if _instance is None:
        _instance = _cut_core.VulkanInstance()
        _interface = _instance.create_interface()


def get_interface() -> _cut_core.VulkanCompute:
    """Get the global Vulkan compute interface."""
    _ensure_initialized()
    return _interface


class Buffer:
    """GPU buffer wrapper with automatic memory management."""

    def __init__(self, data: Optional[np.ndarray] = None, size: Optional[int] = None,
                 is_uniform: bool = False):
        """
        Create a GPU buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            is_uniform: If True, create a uniform buffer; otherwise storage buffer
        """
        _ensure_initialized()
        if data is not None:
            data = np.ascontiguousarray(data)
            self._handle = _interface.create_buffer(data, is_uniform)
            self._size = data.nbytes
            self._dtype = data.dtype
            self._shape = data.shape
        elif size is not None:
            self._handle = _interface.create_buffer_empty(size, is_uniform)
            self._size = size
            self._dtype = None
            self._shape = None
        else:
            raise ValueError("Either data or size must be provided")

    @property
    def handle(self) -> _cut_core.ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._size

    def copy_from(self, data: np.ndarray):
        """Copy data from numpy array to GPU buffer."""
        data = np.ascontiguousarray(data)
        _interface.copy_to_buffer(self._handle, data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Copy data from GPU buffer to numpy array.

        Args:
            out: Output array (created if not provided)

        Returns:
            NumPy array with buffer contents
        """
        if out is None:
            if self._dtype is not None and self._shape is not None:
                out = np.empty(self._shape, dtype=self._dtype)
            else:
                out = np.empty(self._size, dtype=np.uint8)
        out = np.ascontiguousarray(out)
        _interface.copy_from_buffer(self._handle, out)
        return out

    def numpy(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        return self.copy_to()


class Shader:
    """GPU compute shader wrapper."""

    def __init__(self, spirv: Union[List[int], _cut_core.ShaderEnum]):
        """
        Create a shader module.

        Args:
            spirv: SPIR-V bytecode as list of uint32 or a ShaderEnum for built-in shaders
        """
        _ensure_initialized()
        if isinstance(spirv, _cut_core.ShaderEnum):
            spirv = _cut_core.get_shader(spirv)
        self._handle = _interface.create_shader_module(spirv)

    @property
    def handle(self) -> _cut_core.ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle


class Dispatch:
    """Compute dispatch configuration."""

    def __init__(self, shader: Shader, thread_groups: tuple = (1, 1, 1)):
        """
        Create a compute dispatch.

        Args:
            shader: Shader to execute
            thread_groups: Number of thread groups (x, y, z)
        """
        self._dispatch = _cut_core.ComputeDispatch(shader.handle)
        self._dispatch.set_thread_group_size(
            _cut_core.ThreadGroupSize(thread_groups[0], thread_groups[1], thread_groups[2])
        )
        self._bindings = []

    def bind(self, resource: Union[Buffer, np.ndarray], binding: int) -> "Dispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer or numpy array (for push constants)
            binding: Binding index

        Returns:
            self for chaining
        """
        if isinstance(resource, Buffer):
            self._dispatch.bind_resource(resource.handle, binding)
        else:
            data = np.ascontiguousarray(resource)
            self._dispatch.bind_data(data, binding)
            self._bindings.append(data)  # Keep reference alive
        return self

    @property
    def inner(self) -> _cut_core.ComputeDispatch:
        """Get the underlying dispatch object."""
        return self._dispatch


def run(*dispatches: Dispatch):
    """
    Execute one or more compute dispatches.

    Args:
        *dispatches: Dispatch objects to execute
    """
    _ensure_initialized()
    _interface.begin_command_buffer()
    for d in dispatches:
        _interface.encode(d.inner)
    cmd = _interface.end_command_buffer()
    _interface.submit(cmd)
    _interface.wait(cmd)


# =============================================================================
# High-level shader functions (direct API for built-in shaders)
# =============================================================================

def vector_add(a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Add two vectors element-wise on the GPU.

    Args:
        a: First input array (float32)
        b: Second input array (float32)
        out: Output array (created if not provided)

    Returns:
        Result array (a + b)
    """
    _ensure_initialized()

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if out is None:
        out = np.empty_like(a)
    else:
        out = np.ascontiguousarray(out, dtype=np.float32)

    num_elements = np.array([a.size], dtype=np.uint32)

    buf_a = Buffer(a)
    buf_b = Buffer(b)
    buf_out = Buffer(size=out.nbytes)

    shader = Shader(_cut_core.ShaderEnum.VECTOR_ADD)
    workgroups = (a.size + 63) // 64  # 64 threads per workgroup

    dispatch = Dispatch(shader, (workgroups, 1, 1))
    dispatch.bind(buf_a, 0)
    dispatch.bind(buf_b, 1)
    dispatch.bind(buf_out, 2)
    dispatch.bind(num_elements, 3)  # Push constant

    run(dispatch)

    return buf_out.copy_to(out)


# Export public API
__all__ = [
    # Classes
    "Buffer",
    "Shader",
    "Dispatch",
    # Functions
    "run",
    "get_interface",
    "vector_add",
    # Re-exports from core
    "ShaderEnum",
]

# Re-export ShaderEnum
ShaderEnum = _cut_core.ShaderEnum
