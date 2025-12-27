"""
CUT Unified Compute Interface

This module provides a single interface for all CUT backends (Vulkan, CPU).
Initialize with the desired backend using `init()`, then use operations directly.

Example:
    import cut.compute as cc

    # Initialize with Vulkan backend
    cc.init(cc.Backend.Vulkan)

    # Or use CPU with SIMD
    cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.Auto)

    # Use operations
    a = cc.Buffer(np.array([1, 2, 3], dtype=np.float32))
    b = cc.Buffer(np.array([4, 5, 6], dtype=np.float32))
    c = cc.add(a, b)
    result = c.numpy()
"""

from __future__ import annotations

import atexit
import gc
import weakref
import numpy as np
from typing import Optional, Union, List, Any
from enum import Enum, auto

# Import the unified C++ binding
from . import _cut_compute

from ._ops import (
    ALL_OPERATION_NAMES,
    BINARY_VEC_VEC_OPS,
    BINARY_VEC_SCALAR_OPS,
    UNARY_OPS,
    BINARY_VEC_VEC_DOCS,
    BINARY_VEC_SCALAR_DOCS,
    UNARY_DOCS,
)


# Re-export Backend and SIMDMode from C++ binding
Backend = _cut_compute.BackendType
SIMDMode = _cut_compute.SIMDMode
DataType = _cut_compute.DataType
OperatorEnum = _cut_compute.OperatorEnum
ShaderEnum = _cut_compute.OperatorEnum  # Alias for backward compatibility
ThreadSize = _cut_compute.ThreadSize
ComputeHandle = _cut_compute.ComputeHandle
ComputeDispatch = _cut_compute.ComputeDispatch


# Module state
_initialized = False
_live_buffers: weakref.WeakSet = weakref.WeakSet()

# Shader cache: maps (OperatorEnum, DataType) -> ComputeHandle
_shader_cache: dict = {}


def _cleanup():
    """Clean up resources at exit."""
    global _initialized, _shader_cache
    for buf in list(_live_buffers):
        if hasattr(buf, '_handle'):
            buf._handle = None
    _shader_cache.clear()
    _initialized = False


atexit.register(_cleanup)


def available_backends() -> List[str]:
    """
    Get list of available backends.

    Returns:
        List of backend names that can be initialized
    """
    available = []
    if _cut_compute.is_vulkan_available():
        available.append("vulkan")
    if _cut_compute.is_cpu_available():
        available.append("cpu")
    return available


def is_vulkan_available() -> bool:
    """Check if Vulkan backend is available."""
    return _cut_compute.is_vulkan_available()


def is_cpu_available() -> bool:
    """Check if CPU backend is available."""
    return _cut_compute.is_cpu_available()


def init(
    backend: Backend = Backend.CPU,
    *,
    num_threads: int = 0,
    simd_mode: SIMDMode = SIMDMode.Auto,
    force: bool = False
) -> Backend:
    """
    Initialize a compute backend.

    Args:
        backend: Backend to use (Backend.Vulkan or Backend.CPU)
        num_threads: Number of worker threads for CPU backend (0 = auto)
        simd_mode: SIMD mode for CPU backend (Scalar, SSE, AVX, Auto)
        force: If True, re-initialize even if already initialized with same backend

    Returns:
        The initialized backend type

    Raises:
        RuntimeError: If requested backend is not available

    Example:
        >>> import cut.compute as cc
        >>> cc.init(cc.Backend.Vulkan)  # Initialize Vulkan
        >>> cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.AVX)  # CPU with AVX
    """
    global _initialized, _live_buffers

    # Skip re-initialization if already initialized with same backend
    # This prevents destroying resources while buffers still exist
    if _initialized and not force:
        try:
            cur = current_backend()
            if cur == backend:
                return cur
        except Exception:
            pass

    # If force reinitializing or switching backends, clean up first
    if _initialized and force:
        # Clear all buffer references
        for buf in list(_live_buffers):
            if hasattr(buf, '_handle'):
                buf._handle = None
        _live_buffers = weakref.WeakSet()
        # Clear shader cache (shaders are tied to the backend instance)
        _shader_cache.clear()
        # Shutdown the old backend
        _cut_compute.shutdown()
        _initialized = False

    if backend == Backend.Vulkan and not is_vulkan_available():
        raise RuntimeError("Vulkan backend is not available")

    _cut_compute.init(backend, num_threads, simd_mode)
    _initialized = True

    return current_backend()


def _ensure_initialized():
    """Ensure a backend is initialized."""
    global _initialized
    if not _initialized:
        # Auto-initialize with CPU backend (safer default)
        # Users can explicitly call init(Backend.Vulkan) if they want GPU
        init(Backend.CPU, simd_mode=SIMDMode.Auto)


def current_backend() -> Backend:
    """
    Get the currently active backend.

    Returns:
        Current backend type
    """
    return _cut_compute.current_backend()


def num_threads() -> int:
    """
    Get number of worker threads (CPU backend only).

    Returns:
        Number of threads, or 0 for Vulkan backend
    """
    _ensure_initialized()
    return _cut_compute.num_threads()


def simd_mode() -> SIMDMode:
    """
    Get current SIMD mode (CPU backend only).

    Returns:
        Current SIMD mode
    """
    _ensure_initialized()
    return _cut_compute.simd_mode()


def set_simd_mode(mode: SIMDMode):
    """
    Set SIMD mode (CPU backend only).

    Args:
        mode: SIMD mode to set
    """
    _ensure_initialized()
    _cut_compute.set_simd_mode(mode)


def shutdown():
    """
    Shutdown the compute backend and release all resources.

    This function should be called before program exit when using the Vulkan
    backend to ensure proper cleanup. It:
    - Forces garbage collection to release Python buffer references
    - Clears all live buffer references
    - Clears the shader cache
    - Destroys the compute interface
    - Destroys the Vulkan instance (if using Vulkan)

    After calling shutdown(), you must call init() again before using
    any compute operations.

    Example:
        >>> import cut.compute as cc
        >>> cc.init(cc.Backend.Vulkan)
        >>> # ... use compute operations ...
        >>> cc.shutdown()  # Clean up before exit
    """
    global _initialized, _live_buffers, _shader_cache

    # Force garbage collection to release Python buffer references
    gc.collect()
    gc.collect()

    # Clear all buffer references first
    for buf in list(_live_buffers):
        if hasattr(buf, '_handle'):
            buf._handle = None
    _live_buffers = weakref.WeakSet()

    # Clear shader cache
    _shader_cache.clear()

    # Call C++ shutdown to properly destroy Vulkan resources
    _cut_compute.shutdown()

    _initialized = False


# Numpy dtype to DataType mapping
_DTYPE_MAP = {
    np.float32: DataType.Float32,
    np.float16: DataType.Float16,
    np.uint32: DataType.UInt32,
    np.int32: DataType.Int32,
}


def _numpy_dtype_to_cut(dtype) -> DataType:
    """Convert numpy dtype to CUT DataType."""
    dtype_type = np.dtype(dtype).type
    return _DTYPE_MAP.get(dtype_type, DataType.Float32)


class Buffer:
    """
    Unified buffer class for all backends.

    Provides a consistent interface for GPU/CPU buffers regardless
    of which backend is being used.
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        size: Optional[int] = None,
        is_uniform: bool = False,
        dtype: Optional[np.dtype] = None,
        shape: Optional[tuple] = None
    ):
        """
        Create a buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            is_uniform: If True, create a uniform buffer
            dtype: Data type for the buffer (used when creating from size)
            shape: Shape for the buffer (used when creating from size)

        Example:
            >>> a = Buffer(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            >>> b = Buffer(size=64, dtype=np.float32)
        """
        _ensure_initialized()

        if data is not None:
            data = np.ascontiguousarray(data)
            self._handle = _cut_compute.create_buffer(data, is_uniform)
            self._size = data.nbytes
            self._dtype = data.dtype
            self._shape = data.shape
        elif size is not None:
            if dtype is None:
                dtype = np.float32
            element_size = np.dtype(dtype).itemsize
            num_elements = size // element_size
            buffer_shape = list(shape) if shape is not None else [num_elements]

            cut_dtype = _numpy_dtype_to_cut(dtype)
            self._handle = _cut_compute.create_buffer_empty(
                buffer_shape, cut_dtype, is_uniform
            )
            self._size = size
            self._dtype = np.dtype(dtype)
            self._shape = tuple(buffer_shape)
        else:
            raise ValueError("Either data or size must be provided")

        _live_buffers.add(self)

    def _get_module(self):
        """Get the module containing operation functions."""
        import sys
        return sys.modules[__name__]

    # Operator overloading
    def __add__(self, other):
        mod = self._get_module()
        if isinstance(other, Buffer):
            return mod.add(self, other)
        return mod.add_scalar(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        mod = self._get_module()
        if isinstance(other, Buffer):
            return mod.subtract(self, other)
        return mod.subtract_scalar(self, other)

    def __rsub__(self, other):
        mod = self._get_module()
        result = mod.negative(self)
        if isinstance(other, (int, float)):
            return mod.add_scalar(result, other)
        return mod.add(result, other)

    def __mul__(self, other):
        mod = self._get_module()
        if isinstance(other, Buffer):
            return mod.multiply(self, other)
        return mod.multiply_scalar(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        mod = self._get_module()
        if isinstance(other, Buffer):
            return mod.divide(self, other)
        return mod.divide_scalar(self, other)

    def __neg__(self):
        return self._get_module().negative(self)

    @property
    def handle(self) -> ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._size

    @property
    def dtype(self) -> np.dtype:
        """Get the buffer's dtype."""
        return self._dtype

    @property
    def shape(self) -> tuple:
        """Get the buffer's shape."""
        return self._shape

    def copy_from(self, data: np.ndarray):
        """
        Copy data from numpy array to buffer.

        Args:
            data: NumPy array with data to copy
        """
        data = np.ascontiguousarray(data)
        _cut_compute.copy_to_buffer(self._handle, data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Copy data from buffer to numpy array.

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
        _cut_compute.copy_from_buffer(self._handle, out)
        return out

    def numpy(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        return self.copy_to()


class Shader:
    """
    Unified shader/kernel class.

    For Vulkan, this wraps a compiled SPIR-V shader module.
    For CPU, this wraps a kernel function reference.

    Shaders are cached by (operator, dtype) to avoid recompilation.
    """

    def __init__(
        self,
        op_or_spirv: Union[OperatorEnum, List[int]],
        dtype: Optional[np.dtype] = None
    ):
        """
        Create a shader/kernel.

        Args:
            op_or_spirv: OperatorEnum for built-in ops, or SPIR-V bytecode list
            dtype: NumPy dtype for type-specific shaders (Vulkan)

        Example:
            >>> shader = Shader(OperatorEnum.BinaryVecVecAdd)
            >>> shader = Shader(OperatorEnum.BinaryVecVecAdd, dtype=np.int32)
        """
        _ensure_initialized()

        if isinstance(op_or_spirv, (list, tuple)):
            # SPIR-V bytecode - not cached (custom shaders)
            self._handle = _cut_compute.create_shader_from_spirv(op_or_spirv)
            self._op = None
            self._cached = False
        else:
            # Operator enum - check cache first
            cut_dtype = DataType.Float32
            if dtype is not None:
                cut_dtype = _numpy_dtype_to_cut(dtype)

            cache_key = (op_or_spirv, cut_dtype)
            if cache_key in _shader_cache:
                # Use cached shader handle
                self._handle = _shader_cache[cache_key]
                self._cached = True
            else:
                # Create new shader and cache it
                self._handle = _cut_compute.create_shader(op_or_spirv, cut_dtype)
                _shader_cache[cache_key] = self._handle
                self._cached = False

            self._op = op_or_spirv

    @property
    def handle(self) -> ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle

    @property
    def cached(self) -> bool:
        """Returns True if this shader was retrieved from cache."""
        return self._cached


class Dispatch:
    """
    Unified dispatch class for executing compute operations.
    """

    def __init__(self, shader: Shader, thread_groups: tuple = (1, 1, 1)):
        """
        Create a compute dispatch.

        Args:
            shader: Shader/Kernel to execute
            thread_groups: Number of thread groups (x, y, z)
        """
        _ensure_initialized()

        self._dispatch = ComputeDispatch(shader.handle)
        self._dispatch.set_workgroup_size(
            ThreadSize(thread_groups[0], thread_groups[1], thread_groups[2])
        )
        self._bindings = []  # Keep references alive

    def bind(
        self,
        resource: Union[Buffer, np.ndarray, int, float],
        binding: int
    ) -> "Dispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer, numpy array, int, or float
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
        else:
            raise TypeError(f"Unsupported resource type: {type(resource)}")
        return self

    @property
    def inner(self) -> ComputeDispatch:
        """Get the underlying dispatch object."""
        return self._dispatch


def run(*dispatches: Dispatch):
    """
    Execute one or more compute dispatches.

    Args:
        *dispatches: Dispatch objects to execute
    """
    _ensure_initialized()

    for dispatch in dispatches:
        _cut_compute.encode(dispatch.inner)

    cmd = _cut_compute.submit()
    _cut_compute.wait(cmd)


def precompile_shaders() -> int:
    """
    Precompile all shaders and populate the shader cache (Vulkan backend only).
    For CPU backend, this is a no-op.

    Returns:
        Number of shaders compiled (0 for CPU backend)
    """
    _ensure_initialized()

    if current_backend() != Backend.Vulkan:
        return 0

    # Precompile all operator shaders for all data types
    # Using Shader class automatically populates the cache
    dtypes = [np.float32, np.int32, np.uint32]
    compiled = 0

    for op in OperatorEnum.__members__.values():
        for dtype in dtypes:
            try:
                shader = Shader(op, dtype=dtype)
                if not shader.cached:
                    compiled += 1
            except Exception:
                pass  # Some combinations may not exist

    return compiled


def get_shader_cache_stats() -> dict:
    """
    Get statistics about the shader cache.

    Returns:
        Dictionary with cache statistics:
        - size: Number of cached shaders
        - entries: List of (op_name, dtype_name) tuples for cached shaders
    """
    entries = []
    for (op, dtype) in _shader_cache.keys():
        op_name = op.name if hasattr(op, 'name') else str(op)
        dtype_name = dtype.name if hasattr(dtype, 'name') else str(dtype)
        entries.append((op_name, dtype_name))

    return {
        'size': len(_shader_cache),
        'entries': entries
    }


def clear_shader_cache():
    """
    Clear the shader cache.

    Note: This only clears the Python-side cache. The underlying shader
    modules remain valid until shutdown() is called.
    """
    global _shader_cache
    _shader_cache.clear()


# =============================================================================
# Operation Implementations
# =============================================================================

def _create_binary_op(op_enum: OperatorEnum):
    """Create a binary vec-vec operation function."""
    def binary_op(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
        _ensure_initialized()

        if a.size != b.size:
            raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

        if out is None:
            out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader = Shader(op_enum, dtype=a._dtype)
        dispatch = Dispatch(shader, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(b, 1)
        dispatch.bind(out, 2)
        dispatch.bind(num_elements, 3)

        run(dispatch)
        return out

    return binary_op


def _create_unary_op(op_enum: OperatorEnum):
    """Create a unary operation function."""
    def unary_op(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
        _ensure_initialized()

        if out is None:
            out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader = Shader(op_enum, dtype=a._dtype)
        dispatch = Dispatch(shader, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(out, 1)
        dispatch.bind(num_elements, 2)

        run(dispatch)
        return out

    return unary_op


def _create_binary_vec_scalar_op(op_enum: OperatorEnum):
    """Create a binary vec-scalar operation function."""
    def vec_scalar_op(
        a: Buffer,
        scalar: Union[int, float],
        out: Optional[Buffer] = None
    ) -> Buffer:
        _ensure_initialized()

        if out is None:
            out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader = Shader(op_enum, dtype=a._dtype)
        dispatch = Dispatch(shader, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(out, 1)

        # Pack push constants as numpy array
        dtype = a._dtype if a._dtype is not None else np.float32
        if dtype == np.int32:
            push_constants = np.array([num_elements, int(scalar)], dtype=np.int32)
        elif dtype == np.uint32:
            push_constants = np.array([num_elements, int(scalar)], dtype=np.uint32)
        else:
            push_constants = np.array([num_elements, 0], dtype=np.uint32)
            push_constants.view(np.float32)[1] = float(scalar)
        dispatch.bind(push_constants, 2)

        run(dispatch)
        return out

    return vec_scalar_op


# Register all operations into module namespace
def _register_operations():
    """Register all operation functions into the module."""
    module_globals = globals()

    # Binary vec-vec operations
    for op_name, enum_name in BINARY_VEC_VEC_OPS.items():
        enum_value = getattr(OperatorEnum, enum_name)
        func = _create_binary_op(enum_value)
        func.__name__ = op_name
        func.__doc__ = BINARY_VEC_VEC_DOCS.get(op_name, "")
        module_globals[op_name] = func

    # Binary vec-scalar operations
    for op_name, enum_name in BINARY_VEC_SCALAR_OPS.items():
        enum_value = getattr(OperatorEnum, enum_name)
        func = _create_binary_vec_scalar_op(enum_value)
        func.__name__ = op_name
        func.__doc__ = BINARY_VEC_SCALAR_DOCS.get(op_name, "")
        module_globals[op_name] = func

    # Unary operations
    for op_name, enum_name in UNARY_OPS.items():
        enum_value = getattr(OperatorEnum, enum_name)
        func = _create_unary_op(enum_value)
        func.__name__ = op_name
        func.__doc__ = UNARY_DOCS.get(op_name, "")
        module_globals[op_name] = func


# Register operations when module is imported
_register_operations()


# Helper function to get SPIR-V shaders
def get_shader(op: OperatorEnum, scalar_dtype: _cut_compute.ScalarDataType):
    """
    Get SPIR-V bytecode for a built-in shader.

    Args:
        op: Operator enum
        scalar_dtype: Scalar data type

    Returns:
        List of uint32 SPIR-V words
    """
    return _cut_compute.get_shader(op, scalar_dtype)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Backend management
    "Backend",
    "SIMDMode",
    "init",
    "shutdown",
    "available_backends",
    "is_vulkan_available",
    "is_cpu_available",
    "current_backend",
    "num_threads",
    "simd_mode",
    "set_simd_mode",
    # Data types
    "DataType",
    "OperatorEnum",
    "ShaderEnum",
    # Classes
    "Buffer",
    "Shader",
    "Dispatch",
    "ThreadSize",
    "ComputeHandle",
    "ComputeDispatch",
    # Core functions
    "run",
    "precompile_shaders",
    "get_shader",
    # Shader cache functions
    "get_shader_cache_stats",
    "clear_shader_cache",
] + ALL_OPERATION_NAMES
