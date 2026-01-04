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
from typing import Optional, Union, List

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
ComputeBinding = _cut_compute.ComputeBinding
ComputeDispatch = _cut_compute.ComputeDispatch


# Module state
_initialized = False
_live_buffers: weakref.WeakSet = weakref.WeakSet()


def _atexit_shutdown():
    """Shutdown handler called automatically at module exit."""
    global _initialized
    if _initialized:
        # Import here to avoid issues during interpreter shutdown
        import gc as gc_module
        gc_module.collect()
        gc_module.collect()

        for buf in list(_live_buffers):
            if hasattr(buf, '_handle'):
                buf._handle = None

        try:
            _cut_compute.shutdown()
        except Exception:
            pass  # Ignore errors during interpreter shutdown

        _initialized = False


atexit.register(_atexit_shutdown)


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


def is_gpu() -> bool:
    """
    Check if the current backend is a GPU backend.

    Returns:
        True if using GPU backend (Vulkan), False otherwise (CPU)
    """
    return current_backend() == Backend.Vulkan


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
    global _initialized, _live_buffers

    # Force garbage collection to release Python buffer references
    gc.collect()
    gc.collect()

    # Clear all buffer references first
    for buf in list(_live_buffers):
        if hasattr(buf, '_handle'):
            buf._handle = None
    _live_buffers = weakref.WeakSet()

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


# =============================================================================
# Operation Implementations
# =============================================================================

def _create_scalar_binding(index: int, scalar: Union[int, float], dtype):
    """
    Create a ComputeBinding for a scalar value based on the target dtype.

    Args:
        index: The binding index.
        scalar: The scalar value to bind.
        dtype: The target data type (determines how scalar is stored).

    Returns:
        A ComputeBinding with the scalar value in the appropriate format.
    """
    if dtype == np.int32:
        return ComputeBinding.from_int(index, int(scalar))
    elif dtype == np.uint32:
        return ComputeBinding.from_uint(index, int(scalar))
    else:
        return ComputeBinding.from_float(index, float(scalar))


def _create_binary_op(op_enum: OperatorEnum):
    """Create a binary vec-vec operation function."""
    def binary_op(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
        _ensure_initialized()

        if a.size != b.size:
            raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

        if out is None:
            out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

        bindings = [
            ComputeBinding(0, a._handle),
            ComputeBinding(1, b._handle),
            ComputeBinding(2, out._handle),
        ]
        _cut_compute.execute_operator(op_enum, bindings)
        return out

    return binary_op


def _create_unary_op(op_enum: OperatorEnum):
    """Create a unary operation function."""
    def unary_op(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
        _ensure_initialized()

        if out is None:
            out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

        bindings = [
            ComputeBinding(0, a._handle),
            ComputeBinding(1, out._handle),
        ]
        _cut_compute.execute_operator(op_enum, bindings)
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

        dtype = a._dtype if a._dtype is not None else np.float32
        bindings = [
            ComputeBinding(0, a._handle),
            ComputeBinding(1, out._handle),
            _create_scalar_binding(2, scalar, dtype),
        ]
        _cut_compute.execute_operator(op_enum, bindings)
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


# =============================================================================
# Special Operations (not auto-generated)
# =============================================================================

def reduce_sum(a: Buffer) -> float:
    """
    Compute the sum of all elements in the buffer.

    Args:
        a: Input buffer

    Returns:
        Sum of all elements

    Example:
        >>> a = Buffer(np.array([1, 2, 3, 4], dtype=np.float32))
        >>> result = reduce_sum(a)  # Returns 10.0
    """
    _ensure_initialized()

    # Create output buffer for single result
    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceSum, bindings)

    return float(out.numpy()[0])


def reduce_mean(a: Buffer) -> float:
    """
    Compute the mean of all elements in the buffer.

    Args:
        a: Input buffer

    Returns:
        Mean of all elements

    Example:
        >>> a = Buffer(np.array([1, 2, 3, 4], dtype=np.float32))
        >>> result = reduce_mean(a)  # Returns 2.5
    """
    _ensure_initialized()

    # For mean, we compute sum on GPU and divide by count on CPU
    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMean, bindings)

    # GPU computes sum, we divide by count
    total = float(out.numpy()[0])
    count = np.prod(a.shape)
    return total / count


def reduce_min(a: Buffer) -> float:
    """
    Find the minimum element in the buffer.

    Args:
        a: Input buffer

    Returns:
        Minimum element

    Example:
        >>> a = Buffer(np.array([3, 1, 4, 1, 5], dtype=np.float32))
        >>> result = reduce_min(a)  # Returns 1.0
    """
    _ensure_initialized()

    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMin, bindings)

    return float(out.numpy()[0])


def reduce_max(a: Buffer) -> float:
    """
    Find the maximum element in the buffer.

    Args:
        a: Input buffer

    Returns:
        Maximum element

    Example:
        >>> a = Buffer(np.array([3, 1, 4, 1, 5], dtype=np.float32))
        >>> result = reduce_max(a)  # Returns 5.0
    """
    _ensure_initialized()

    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMax, bindings)

    return float(out.numpy()[0])


def reduce_prod(a: Buffer) -> float:
    """
    Compute the product of all elements in the buffer.

    Args:
        a: Input buffer

    Returns:
        Product of all elements

    Example:
        >>> a = Buffer(np.array([1, 2, 3, 4], dtype=np.float32))
        >>> result = reduce_prod(a)  # Returns 24.0
    """
    _ensure_initialized()

    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceProd, bindings)

    return float(out.numpy()[0])


def reduce_any(a: Buffer) -> bool:
    """
    Check if any element in the buffer is non-zero (logical OR).

    Args:
        a: Input buffer

    Returns:
        True if any element is non-zero

    Example:
        >>> a = Buffer(np.array([0, 0, 1, 0], dtype=np.float32))
        >>> result = reduce_any(a)  # Returns True
    """
    _ensure_initialized()

    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceAny, bindings)

    return bool(out.numpy()[0] != 0.0)


def reduce_all(a: Buffer) -> bool:
    """
    Check if all elements in the buffer are non-zero (logical AND).

    Args:
        a: Input buffer

    Returns:
        True if all elements are non-zero

    Example:
        >>> a = Buffer(np.array([1, 2, 3, 4], dtype=np.float32))
        >>> result = reduce_all(a)  # Returns True
    """
    _ensure_initialized()

    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceAll, bindings)

    return bool(out.numpy()[0] != 0.0)


def matmul(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """
    Matrix multiplication: C = A @ B

    Args:
        a: Input matrix A with shape (M, K)
        b: Input matrix B with shape (K, N)
        out: Optional output buffer with shape (M, N)

    Returns:
        Buffer with result of matrix multiplication

    Example:
        >>> a = Buffer(np.array([[1, 2], [3, 4]], dtype=np.float32))
        >>> b = Buffer(np.array([[5, 6], [7, 8]], dtype=np.float32))
        >>> result = matmul(a, b)  # Returns [[19, 22], [43, 50]]
    """
    _ensure_initialized()

    # Get shapes
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul requires 2D matrices")

    M, K = a.shape
    K2, N = b.shape

    if K != K2:
        raise ValueError(f"Matrix dimension mismatch: A is {M}x{K}, B is {K2}x{N}")

    if out is None:
        out = Buffer(size=M * N * 4, dtype=np.float32, shape=(M, N))

    # Create shape data binding
    shape_data = np.array([M, K, N], dtype=np.uint32)

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, b._handle),
        ComputeBinding(2, out._handle),
        ComputeBinding.from_bytes(3, shape_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.MatMul, bindings)
    return out


def transpose(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """
    Matrix transpose: B = A^T

    Args:
        a: Input matrix A with shape (M, N)
        out: Optional output buffer with shape (N, M)

    Returns:
        Buffer with transposed matrix

    Example:
        >>> a = Buffer(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        >>> result = transpose(a)  # Returns [[1, 4], [2, 5], [3, 6]]
    """
    _ensure_initialized()

    if len(a.shape) != 2:
        raise ValueError("transpose requires a 2D matrix")

    M, N = a.shape

    if out is None:
        out = Buffer(size=M * N * 4, dtype=np.float32, shape=(N, M))

    # Create shape data binding
    shape_data = np.array([M, N], dtype=np.uint32)

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
        ComputeBinding.from_bytes(2, shape_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.Transpose, bindings)
    return out


def dot(a: Buffer, b: Buffer) -> float:
    """
    Dot product of two vectors: result = sum(A * B)

    Args:
        a: Input vector A
        b: Input vector B (same size as A)

    Returns:
        Scalar dot product result

    Example:
        >>> a = Buffer(np.array([1, 2, 3], dtype=np.float32))
        >>> b = Buffer(np.array([4, 5, 6], dtype=np.float32))
        >>> result = dot(a, b)  # Returns 32.0 (1*4 + 2*5 + 3*6)
    """
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")

    count = np.prod(a.shape)

    # Create output buffer for single result
    out = Buffer(size=4, dtype=np.float32, shape=(1,))

    # Create count data binding
    count_data = np.array([count], dtype=np.uint32)

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, b._handle),
        ComputeBinding(2, out._handle),
        ComputeBinding.from_bytes(3, count_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.Dot, bindings)
    return float(out.numpy()[0])


def clamp(a: Buffer, min_val: Union[int, float], max_val: Union[int, float],
          out: Optional[Buffer] = None) -> Buffer:
    """
    Clamp buffer values to a range.

    Each element is clamped to be within [min_val, max_val].

    Args:
        a: Input buffer
        min_val: Minimum value
        max_val: Maximum value
        out: Optional output buffer

    Returns:
        Buffer with clamped values

    Example:
        >>> a = Buffer(np.array([-1, 0, 5, 10], dtype=np.float32))
        >>> result = clamp(a, 0, 5)  # Returns [0, 0, 5, 5]
    """
    _ensure_initialized()

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    dtype = a._dtype if a._dtype is not None else np.float32

    # Create data binding with min and max values packed as numpy array
    if dtype == np.int32:
        clamp_data = np.array([int(min_val), int(max_val)], dtype=np.int32)
    elif dtype == np.uint32:
        clamp_data = np.array([int(min_val), int(max_val)], dtype=np.uint32)
    else:
        clamp_data = np.array([float(min_val), float(max_val)], dtype=np.float32)

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
        ComputeBinding.from_bytes(2, clamp_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.TernaryClamp, bindings)
    return out


# Helper function to get SPIR-V shaders
def get_shader(op: OperatorEnum, dtype: DataType = DataType.Float32):
    """
    Get SPIR-V bytecode for a built-in shader.

    Args:
        op: Operator enum
        dtype: Data type (default: Float32)

    Returns:
        List of uint32 SPIR-V words
    """
    return _cut_compute.get_shader(op, dtype)


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
    "is_gpu",
    "num_threads",
    "simd_mode",
    "set_simd_mode",
    # Data types
    "DataType",
    "OperatorEnum",
    "ShaderEnum",
    # Classes
    "Buffer",
    "ThreadSize",
    "ComputeHandle",
    "ComputeBinding",
    "ComputeDispatch",
    # Core functions
    "get_shader",
    # Special operations
    "clamp",
    # Reduction operations
    "reduce_sum",
    "reduce_mean",
    "reduce_min",
    "reduce_max",
    "reduce_prod",
    "reduce_any",
    "reduce_all",
    # Matrix operations
    "matmul",
    "transpose",
    "dot",
] + ALL_OPERATION_NAMES
