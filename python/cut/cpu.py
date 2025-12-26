"""
CUT CPU Backend - CPU Compute Library

A Python interface for CPU-based compute operations.
Uses multithreaded C++ kernels for parallel execution.
"""

import atexit
import weakref
import numpy as np
from typing import Optional, Union

from . import _cut_cpu
from ._ops import ALL_OPERATION_NAMES, register_operations

__version__ = "0.1.0"

# Re-export SIMDMode enum
SIMDMode = _cut_cpu.SIMDMode

# Module-level CPU interface (lazy initialization)
_interface: Optional[_cut_cpu.CPUCompute] = None

# Current SIMD mode for initialization
_simd_mode: _cut_cpu.SIMDMode = _cut_cpu.SIMDMode.Auto

# Kernel cache: maps OperatorEnum -> ComputeHandle
_kernel_cache: dict = {}

# Track all live buffers using weak references
_live_buffers: weakref.WeakSet = weakref.WeakSet()


def _cleanup():
    """Clean up resources in correct order."""
    global _kernel_cache, _interface
    for buf in list(_live_buffers):
        buf._handle = None
    _kernel_cache.clear()
    _interface = None


atexit.register(_cleanup)


def _ensure_initialized(num_threads: int = 0, simd_mode: Optional[_cut_cpu.SIMDMode] = None):
    """Ensure CPU interface is initialized."""
    global _interface, _simd_mode
    if simd_mode is not None:
        _simd_mode = simd_mode
    if _interface is None:
        _interface = _cut_cpu.CPUCompute(num_threads, _simd_mode)


def get_interface(num_threads: int = 0, simd_mode: Optional[_cut_cpu.SIMDMode] = None) -> _cut_cpu.CPUCompute:
    """Get the global CPU compute interface."""
    _ensure_initialized(num_threads, simd_mode)
    return _interface


def num_threads() -> int:
    """Get the number of worker threads."""
    _ensure_initialized()
    return _interface.num_threads()


def simd_mode() -> _cut_cpu.SIMDMode:
    """Get the current SIMD execution mode."""
    _ensure_initialized()
    return _interface.simd_mode()


def set_simd_mode(mode: _cut_cpu.SIMDMode):
    """
    Set the SIMD execution mode.

    Args:
        mode: SIMDMode.Scalar, SIMDMode.SSE, SIMDMode.AVX, or SIMDMode.Auto
    """
    global _simd_mode
    _simd_mode = mode
    if _interface is not None:
        _interface.set_simd_mode(mode)


class Buffer:
    """CPU buffer wrapper with automatic memory management."""

    def __init__(self, data: Optional[np.ndarray] = None, size: Optional[int] = None,
                 is_uniform: bool = False, dtype: Optional[np.dtype] = None,
                 shape: Optional[tuple] = None):
        """
        Create a CPU buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            is_uniform: Ignored for CPU backend
            dtype: Data type for the buffer (used when creating from size)
            shape: Shape for the buffer (used when creating from size)
        """
        _ensure_initialized()
        if data is not None:
            data = np.ascontiguousarray(data)
            self._handle = _interface.create_buffer(data, is_uniform)
            self._size = data.nbytes
            self._dtype = data.dtype
            self._shape = data.shape
        elif size is not None:
            # Convert size (bytes) to shape for the new API
            # Default to float32 if dtype not specified
            if dtype is None:
                dtype = np.float32
            element_size = np.dtype(dtype).itemsize
            num_elements = size // element_size
            buffer_shape = list(shape) if shape is not None else [num_elements]
            # Map numpy dtype to cut DataType
            dtype_map = {
                np.float32: _cut_cpu.DataType.Float32,
                np.float16: _cut_cpu.DataType.Float16,
                np.uint32: _cut_cpu.DataType.UInt32,
                np.int32: _cut_cpu.DataType.Int32,
            }
            cut_dtype = dtype_map.get(np.dtype(dtype).type, _cut_cpu.DataType.Float32)
            self._handle = _interface.create_buffer_empty(buffer_shape, cut_dtype, is_uniform)
            self._size = size
            self._dtype = dtype
            self._shape = tuple(buffer_shape)
        else:
            raise ValueError("Either data or size must be provided")
        _live_buffers.add(self)

    @property
    def handle(self) -> _cut_cpu.ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._size

    def copy_from(self, data: np.ndarray):
        """Copy data from numpy array to CPU buffer."""
        data = np.ascontiguousarray(data)
        _interface.copy_to_buffer(self._handle, data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy data from CPU buffer to numpy array."""
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


class Kernel:
    """CPU compute kernel wrapper."""

    def __init__(self, kernel_type: _cut_cpu.OperatorEnum, dtype=None):
        """
        Create a kernel from an OperatorEnum.

        Args:
            kernel_type: The operator enum type
            dtype: Optional dtype (ignored for CPU backend, for API compatibility)
        """
        _ensure_initialized()
        # Check cache first
        if kernel_type in _kernel_cache:
            self._handle = _kernel_cache[kernel_type]
            return
        # Create and cache
        self._handle = _interface.create_kernel(kernel_type)
        _kernel_cache[kernel_type] = self._handle

    @property
    def handle(self) -> _cut_cpu.ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle


class Dispatch:
    """Compute dispatch configuration."""

    def __init__(self, kernel: Kernel, thread_groups: tuple = (1, 1, 1)):
        """
        Create a compute dispatch.

        Args:
            kernel: Kernel to execute
            thread_groups: Number of thread groups (x, y, z)
        """
        self._dispatch = _cut_cpu.ComputeDispatch(kernel.handle)
        self._dispatch.set_workgroup_size(
            _cut_cpu.ThreadSize(thread_groups[0], thread_groups[1], thread_groups[2])
        )
        self._bindings = []

    def bind(self, resource: Union[Buffer, np.ndarray, int, float], binding: int) -> "Dispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer, numpy array, int (as uint32), or float (as float32)
            binding: Binding index

        Returns:
            self for chaining
        """
        if isinstance(resource, Buffer):
            self._dispatch.bind_resource(resource.handle, binding)
        elif isinstance(resource, np.ndarray):
            data = np.ascontiguousarray(resource)
            self._dispatch.bind_data(data, binding)
            self._bindings.append(data)  # Keep reference alive
        elif isinstance(resource, int):
            self._dispatch.bind_uint(resource, binding)
        elif isinstance(resource, float):
            self._dispatch.bind_float(resource, binding)
        return self

    @property
    def inner(self) -> _cut_cpu.ComputeDispatch:
        """Get the underlying dispatch object."""
        return self._dispatch


def run(*dispatches: Dispatch):
    """
    Execute one or more compute dispatches.

    Args:
        *dispatches: Dispatch objects to execute
    """
    _ensure_initialized()
    for d in dispatches:
        _interface.encode(d.inner)
    cmd = _interface.submit()
    _interface.wait(cmd)


# =============================================================================
# Register all operations using the shared module
# =============================================================================

register_operations(
    module_dict=globals(),
    enum_module=_cut_cpu,
    buffer_class=Buffer,
    shader_or_kernel_class=Kernel,
    dispatch_class=Dispatch,
    run_func=run,
    ensure_init=_ensure_initialized,
    backend_name="on CPU",
)


# Export public API
__all__ = [
    # Classes
    "Buffer",
    "Kernel",
    "Dispatch",
    # Core functions
    "run",
    "get_interface",
    "num_threads",
    "simd_mode",
    "set_simd_mode",
    # Re-exports from core
    "OperatorEnum",
    "SIMDMode",
] + ALL_OPERATION_NAMES

# Re-export OperatorEnum
OperatorEnum = _cut_cpu.OperatorEnum
