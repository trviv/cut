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
    a = cc.Tensor([1.0, 2.0, 3.0])  # float32 by default
    b = cc.Tensor([4.0, 5.0, 6.0])
    c = cc.add(a, b)
    result = c.tolist()
"""

from __future__ import annotations

import atexit
import gc
import weakref
import array
from enum import Enum
from typing import Optional, Union, List, Sequence

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
_live_tensors: weakref.WeakSet = weakref.WeakSet()


def _atexit_shutdown():
    """Shutdown handler called automatically at module exit."""
    global _initialized
    if _initialized:
        # Import here to avoid issues during interpreter shutdown
        import gc as gc_module
        gc_module.collect()
        gc_module.collect()

        for tensor in list(_live_tensors):
            if hasattr(tensor, '_handle'):
                tensor._handle = None

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
    global _initialized, _live_tensors

    # Skip re-initialization if already initialized with same backend
    # This prevents destroying resources while tensors still exist
    if _initialized and not force:
        try:
            cur = current_backend()
            if cur == backend:
                return cur
        except Exception:
            pass

    # If force reinitializing or switching backends, clean up first
    if _initialized and force:
        # Clear all tensor references
        for tensor in list(_live_tensors):
            if hasattr(tensor, '_handle'):
                tensor._handle = None
        _live_tensors = weakref.WeakSet()
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
    - Forces garbage collection to release Python tensor references
    - Clears all live tensor references
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
    global _initialized, _live_tensors

    # Force garbage collection to release Python tensor references
    gc.collect()
    gc.collect()

    # Clear all tensor references first
    for tensor in list(_live_tensors):
        if hasattr(tensor, '_handle'):
            tensor._handle = None
    _live_tensors = weakref.WeakSet()

    # Call C++ shutdown to properly destroy Vulkan resources
    _cut_compute.shutdown()

    _initialized = False


# =============================================================================
# DType - Native Python data type representation
# =============================================================================

class DType(Enum):
    """
    Data type enum for tensors.

    Provides dtype information without numpy dependency.
    Uses Python's array module typecodes for buffer protocol compatibility.
    """
    float32 = ('f', 4, 'float32')
    float16 = ('e', 2, 'float16')  # Note: 'e' requires Python 3.6+
    uint32 = ('I', 4, 'uint32')
    int32 = ('i', 4, 'int32')

    def __init__(self, typecode: str, itemsize: int, name: str):
        self._typecode = typecode
        self._itemsize = itemsize
        self._name = name

    @property
    def itemsize(self) -> int:
        """Size of one element in bytes."""
        return self._itemsize

    @property
    def typecode(self) -> str:
        """Array module typecode for this dtype."""
        return self._typecode

    def to_cut_dtype(self) -> DataType:
        """Convert to CUT DataType enum."""
        return _DTYPE_TO_CUT[self]

    def __str__(self):
        return self._name


# Module-level dtype aliases for convenience
float32 = DType.float32
float16 = DType.float16
uint32 = DType.uint32
int32 = DType.int32

# Mapping from DType to CUT DataType
_DTYPE_TO_CUT = {
    DType.float32: DataType.Float32,
    DType.float16: DataType.Float16,
    DType.uint32: DataType.UInt32,
    DType.int32: DataType.Int32,
}


class Tensor:
    """
    Unified tensor class for all backends.

    Provides a consistent interface for GPU/CPU tensors regardless
    of which backend is being used. Uses native Python types instead of numpy.
    """

    def __init__(
        self,
        data: Optional[Union[Sequence, array.array]] = None,
        size: Optional[int] = None,
        is_uniform: bool = False,
        dtype: Optional[DType] = None,
        shape: Optional[tuple] = None
    ):
        """
        Create a tensor.

        Args:
            data: List, tuple, or array.array to initialize tensor with (optional)
            size: Tensor size in bytes (required if data is None)
            is_uniform: If True, create a uniform buffer
            dtype: Data type for the tensor (DType.float32, DType.float16, etc.)
            shape: Shape for the tensor (used when creating from size)

        Example:
            >>> a = Tensor([1.0, 2.0, 3.0])  # float32 by default
            >>> b = Tensor([[1, 2], [3, 4]], dtype=cc.int32)
            >>> c = Tensor(size=64, dtype=cc.float32)
        """
        _ensure_initialized()

        resolved_dtype = dtype if dtype is not None else float32

        if data is not None:
            # Handle array.array directly
            if isinstance(data, array.array):
                flat_data = list(data)
                inferred_shape = (len(flat_data),) if shape is None else shape
            elif hasattr(data, 'flatten') and hasattr(data, 'shape'):
                # Handle numpy arrays - flatten and extract shape
                inferred_shape = tuple(data.shape) if shape is None else shape
                flat_data = data.flatten().tolist()
            else:
                # Flatten nested lists/tuples and infer shape (C++ implementation)
                flat_data, inferred_shape = _cut_compute.flatten_nested(data)
                if shape is not None:
                    inferred_shape = shape

            self._dtype = resolved_dtype
            self._shape = inferred_shape

            # Create array.array with contiguous memory
            arr = array.array(resolved_dtype.typecode, flat_data)
            self._handle = _cut_compute.create_buffer(arr, is_uniform)
            self._size = len(arr) * resolved_dtype.itemsize

        elif size is not None:
            num_elements = size // resolved_dtype.itemsize
            buffer_shape = tuple(shape) if shape is not None else (num_elements,)

            self._handle = _cut_compute.create_buffer_empty(
                list(buffer_shape), resolved_dtype.to_cut_dtype(), is_uniform
            )
            self._size = size
            self._dtype = resolved_dtype
            self._shape = buffer_shape
        else:
            raise ValueError("Either data or size must be provided")

        _live_tensors.add(self)

    # Operator overloading - operations are looked up at call time to avoid circular imports
    def __add__(self, other):
        if isinstance(other, Tensor):
            return globals()['add'](self, other)
        return globals()['add_scalar'](self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return globals()['subtract'](self, other)
        return globals()['subtract_scalar'](self, other)

    def __rsub__(self, other):
        result = globals()['negative'](self)
        if isinstance(other, (int, float)):
            return globals()['add_scalar'](result, other)
        return globals()['add'](result, other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return globals()['multiply'](self, other)
        return globals()['multiply_scalar'](self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return globals()['divide'](self, other)
        return globals()['divide_scalar'](self, other)

    def __neg__(self):
        return globals()['negative'](self)

    @property
    def handle(self) -> ComputeHandle:
        """Get the underlying compute handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Get tensor size in bytes."""
        return self._size

    @property
    def dtype(self) -> DType:
        """Get the tensor's dtype."""
        return self._dtype

    @property
    def shape(self) -> tuple:
        """Get the tensor's shape."""
        return self._shape

    def copy_from(self, data: Union[Sequence, array.array]):
        """
        Copy data from list/array to tensor.

        Args:
            data: List, tuple, or array.array with data to copy
        """
        if isinstance(data, array.array):
            arr = data
        else:
            flat_data, _ = _cut_compute.flatten_nested(data)
            arr = array.array(self._dtype.typecode, flat_data)
        _cut_compute.copy_to_buffer(self._handle, arr)

    def copy_to(self, out: Optional[array.array] = None) -> array.array:
        """
        Copy data from tensor to array.

        Args:
            out: Output array (created if not provided)

        Returns:
            array.array with tensor contents
        """
        if out is None:
            num_elements = _cut_compute.shape_product(list(self._shape)) if self._shape else self._size // self._dtype.itemsize
            out = array.array(self._dtype.typecode, [0] * num_elements)
        _cut_compute.copy_from_buffer(self._handle, out)
        return out

    def tolist(self) -> Union[List, float, int]:
        """
        Get tensor contents as a nested Python list matching the shape.
        Uses native C++ implementation for better performance.

        Returns:
            Nested list with tensor contents, or scalar if shape is ()
        """
        arr = self.copy_to()

        if not self._shape or self._shape == ():
            return list(arr)[0] if arr else 0

        # Use native C++ reshape for performance
        return _cut_compute.reshape_to_nested(arr, self._dtype.to_cut_dtype(), list(self._shape))


# =============================================================================
# Operation Implementations
# =============================================================================

def _create_binary_op(op_enum: OperatorEnum):
    """Create a binary vec-vec operation function."""
    def binary_op(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
        _ensure_initialized()

        if a.size != b.size:
            raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

        if out is None:
            out = Tensor(size=a.size, dtype=a._dtype, shape=a._shape)

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
    def unary_op(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
        _ensure_initialized()

        if out is None:
            out = Tensor(size=a.size, dtype=a._dtype, shape=a._shape)

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
        a: Tensor,
        scalar: Union[int, float],
        out: Optional[Tensor] = None
    ) -> Tensor:
        _ensure_initialized()

        if out is None:
            out = Tensor(size=a.size, dtype=a._dtype, shape=a._shape)

        dtype = a._dtype if a._dtype is not None else float32
        # Inline scalar binding creation
        if dtype == int32:
            scalar_binding = ComputeBinding.from_int(2, int(scalar))
        elif dtype == uint32:
            scalar_binding = ComputeBinding.from_uint(2, int(scalar))
        else:
            scalar_binding = ComputeBinding.from_float(2, float(scalar))

        bindings = [
            ComputeBinding(0, a._handle),
            ComputeBinding(1, out._handle),
            scalar_binding,
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

def reduce_sum(a: Tensor) -> float:
    """
    Compute the sum of all elements in the tensor.

    Args:
        a: Input tensor

    Returns:
        Sum of all elements

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = reduce_sum(a)  # Returns 10.0
    """
    _ensure_initialized()

    # Create output tensor for single result
    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceSum, bindings)

    return float(out.copy_to()[0])


def reduce_mean(a: Tensor) -> float:
    """
    Compute the mean of all elements in the tensor.

    Args:
        a: Input tensor

    Returns:
        Mean of all elements

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = reduce_mean(a)  # Returns 2.5
    """
    _ensure_initialized()

    # For mean, we compute sum on GPU and divide by count on CPU
    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMean, bindings)

    # GPU computes sum, we divide by count
    total = float(out.copy_to()[0])
    count = _cut_compute.shape_product(list(a.shape))
    return total / count


def reduce_min(a: Tensor) -> float:
    """
    Find the minimum element in the tensor.

    Args:
        a: Input tensor

    Returns:
        Minimum element

    Example:
        >>> a = Tensor([3, 1, 4, 1, 5])
        >>> result = reduce_min(a)  # Returns 1.0
    """
    _ensure_initialized()

    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMin, bindings)

    return float(out.copy_to()[0])


def reduce_max(a: Tensor) -> float:
    """
    Find the maximum element in the tensor.

    Args:
        a: Input tensor

    Returns:
        Maximum element

    Example:
        >>> a = Tensor([3, 1, 4, 1, 5])
        >>> result = reduce_max(a)  # Returns 5.0
    """
    _ensure_initialized()

    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceMax, bindings)

    return float(out.copy_to()[0])


def reduce_prod(a: Tensor) -> float:
    """
    Compute the product of all elements in the tensor.

    Args:
        a: Input tensor

    Returns:
        Product of all elements

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = reduce_prod(a)  # Returns 24.0
    """
    _ensure_initialized()

    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceProd, bindings)

    return float(out.copy_to()[0])


def reduce_any(a: Tensor) -> bool:
    """
    Check if any element in the tensor is non-zero (logical OR).

    Args:
        a: Input tensor

    Returns:
        True if any element is non-zero

    Example:
        >>> a = Tensor([0, 0, 1, 0])
        >>> result = reduce_any(a)  # Returns True
    """
    _ensure_initialized()

    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceAny, bindings)

    return bool(out.copy_to()[0] != 0.0)


def reduce_all(a: Tensor) -> bool:
    """
    Check if all elements in the tensor are non-zero (logical AND).

    Args:
        a: Input tensor

    Returns:
        True if all elements are non-zero

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = reduce_all(a)  # Returns True
    """
    _ensure_initialized()

    out = Tensor(size=4, dtype=float32, shape=(1,))

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.ReduceAll, bindings)

    return bool(out.copy_to()[0] != 0.0)


def matmul(a: Tensor, b: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """
    Matrix multiplication: C = A @ B

    Args:
        a: Input matrix A with shape (M, K)
        b: Input matrix B with shape (K, N)
        out: Optional output tensor with shape (M, N)

    Returns:
        Tensor with result of matrix multiplication

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
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
        out = Tensor(size=M * N * 4, dtype=float32, shape=(M, N))

    # Create shape data binding
    shape_data = array.array('I', [M, K, N])

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, b._handle),
        ComputeBinding(2, out._handle),
        ComputeBinding.from_bytes(3, shape_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.MatMul, bindings)
    return out


def transpose(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """
    Matrix transpose: B = A^T

    Args:
        a: Input matrix A with shape (M, N)
        out: Optional output tensor with shape (N, M)

    Returns:
        Tensor with transposed matrix

    Example:
        >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> result = transpose(a)  # Returns [[1, 4], [2, 5], [3, 6]]
    """
    _ensure_initialized()

    if len(a.shape) != 2:
        raise ValueError("transpose requires a 2D matrix")

    M, N = a.shape

    if out is None:
        out = Tensor(size=M * N * 4, dtype=float32, shape=(N, M))

    # Create shape data binding
    shape_data = array.array('I', [M, N])

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
        ComputeBinding.from_bytes(2, shape_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.Transpose, bindings)
    return out


def dot(a: Tensor, b: Tensor) -> float:
    """
    Dot product of two vectors: result = sum(A * B)

    Args:
        a: Input vector A
        b: Input vector B (same size as A)

    Returns:
        Scalar dot product result

    Example:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> result = dot(a, b)  # Returns 32.0 (1*4 + 2*5 + 3*6)
    """
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")

    count = _cut_compute.shape_product(list(a.shape))

    # Create output tensor for single result
    out = Tensor(size=4, dtype=float32, shape=(1,))

    # Create count data binding
    count_data = array.array('I', [count])

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, b._handle),
        ComputeBinding(2, out._handle),
        ComputeBinding.from_bytes(3, count_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.Dot, bindings)
    return float(out.copy_to()[0])


def clamp(a: Tensor, min_val: Union[int, float], max_val: Union[int, float],
          out: Optional[Tensor] = None) -> Tensor:
    """
    Clamp tensor values to a range.

    Each element is clamped to be within [min_val, max_val].

    Args:
        a: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        out: Optional output tensor

    Returns:
        Tensor with clamped values

    Example:
        >>> a = Tensor([-1, 0, 5, 10])
        >>> result = clamp(a, 0, 5)  # Returns [0, 0, 5, 5]
    """
    _ensure_initialized()

    if out is None:
        out = Tensor(size=a.size, dtype=a._dtype, shape=a._shape)

    dtype = a._dtype if a._dtype is not None else float32

    # Create data binding with min and max values packed as array
    if dtype == int32:
        clamp_data = array.array('i', [int(min_val), int(max_val)])
    elif dtype == uint32:
        clamp_data = array.array('I', [int(min_val), int(max_val)])
    else:
        clamp_data = array.array('f', [float(min_val), float(max_val)])

    bindings = [
        ComputeBinding(0, a._handle),
        ComputeBinding(1, out._handle),
        ComputeBinding.from_bytes(2, clamp_data),
    ]
    _cut_compute.execute_operator(OperatorEnum.TernaryClamp, bindings)
    return out


def where(condition: Tensor, x: Tensor, y: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """
    Select elements from x or y depending on condition.

    Returns elements from x where condition is non-zero, otherwise from y.
    This is equivalent to PyTorch's torch.where(condition, x, y).

    Args:
        condition: Condition tensor (non-zero means True)
        x: Values to select when condition is True
        y: Values to select when condition is False
        out: Optional output tensor

    Returns:
        Tensor with selected values

    Example:
        >>> cond = Tensor([1, 0, 1, 0])
        >>> x = Tensor([10, 20, 30, 40])
        >>> y = Tensor([1, 2, 3, 4])
        >>> result = where(cond, x, y)  # Returns [10, 2, 30, 4]
    """
    _ensure_initialized()

    # Check shapes match
    if condition.size != x.size or condition.size != y.size:
        raise ValueError("condition, x, and y must have the same size")

    if out is None:
        out = Tensor(size=x.size, dtype=x._dtype, shape=x._shape)

    bindings = [
        ComputeBinding(0, condition._handle),
        ComputeBinding(1, x._handle),
        ComputeBinding(2, y._handle),
        ComputeBinding(3, out._handle),
    ]
    _cut_compute.execute_operator(OperatorEnum.TernarySelect, bindings)
    return out


def concat(tensors: List[Tensor], axis: int = 0, out: Optional[Tensor] = None) -> Tensor:
    """
    Concatenate tensors along an axis.

    Similar to PyTorch's torch.cat() and NumPy's concatenate().

    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate (default: 0)
        out: Optional output tensor

    Returns:
        Concatenated tensor

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6]])
        >>> result = concat([a, b], axis=0)  # Shape: (3, 2)
    """
    _ensure_initialized()

    if not tensors:
        raise ValueError("Need at least one tensor to concatenate")

    if len(tensors) == 1:
        return tensors[0]

    # For now, implement a simple version that works with flattened tensors
    # Full implementation would need backend support for arbitrary axis
    if axis != 0:
        raise NotImplementedError("Only axis=0 concatenation is currently supported")

    # Calculate total size
    total_size = sum(t.size for t in tensors)
    dtype = tensors[0]._dtype if tensors[0]._dtype is not None else float32

    if out is None:
        out = Tensor(size=total_size, dtype=dtype)

    # Copy each tensor into output
    offset = 0
    for t in tensors:
        data = t.tolist()
        out_data = out.tolist()
        out_data[offset:offset+t.size] = data
        out.from_list(out_data)
        offset += t.size

    return out


def stack(tensors: List[Tensor], axis: int = 0, out: Optional[Tensor] = None) -> Tensor:
    """
    Stack tensors along a new axis.

    Similar to PyTorch's torch.stack() and NumPy's stack().
    Unlike concat, this creates a new dimension.

    Args:
        tensors: List of tensors to stack (must have same shape)
        axis: Axis along which to stack (default: 0)
        out: Optional output tensor

    Returns:
        Stacked tensor

    Example:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> result = stack([a, b], axis=0)  # Shape: (2, 3)
    """
    _ensure_initialized()

    if not tensors:
        raise ValueError("Need at least one tensor to stack")

    if len(tensors) == 1:
        # Add new dimension
        t = tensors[0]
        return Tensor(data=t.tolist(), shape=[1] + list(t._shape))

    # Check all tensors have the same shape
    first_shape = tensors[0]._shape
    for t in tensors[1:]:
        if t._shape != first_shape:
            raise ValueError("All tensors must have the same shape for stacking")

    # For now, support axis=0 only
    if axis != 0:
        raise NotImplementedError("Only axis=0 stacking is currently supported")

    # Create output shape
    new_shape = [len(tensors)] + list(first_shape)
    total_size = len(tensors) * tensors[0].size
    dtype = tensors[0]._dtype if tensors[0]._dtype is not None else float32

    if out is None:
        out = Tensor(size=total_size, dtype=dtype, shape=new_shape)

    # Copy each tensor into output
    all_data = []
    for t in tensors:
        all_data.extend(t.tolist())

    out.from_list(all_data)
    return out


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1, out: Optional[Tensor] = None) -> Tensor:
    """
    Flatten tensor dimensions.

    Similar to PyTorch's torch.flatten().

    Args:
        input: Input tensor
        start_dim: First dimension to flatten (default: 0)
        end_dim: Last dimension to flatten (default: -1, meaning last dim)
        out: Optional output tensor

    Returns:
        Flattened tensor

    Example:
        >>> a = Tensor(shape=(2, 3, 4), data=list(range(24)))
        >>> result = flatten(a)  # Shape: (24,)
        >>> result = flatten(a, start_dim=1)  # Shape: (2, 12)
    """
    _ensure_initialized()

    if input._shape is None or len(input._shape) == 0:
        # Already flat
        return input

    shape = list(input._shape)
    ndim = len(shape)

    # Handle negative indices
    if end_dim < 0:
        end_dim = ndim + end_dim

    if start_dim < 0:
        start_dim = ndim + start_dim

    # Validate
    if start_dim < 0 or start_dim >= ndim:
        raise ValueError(f"start_dim {start_dim} out of range for {ndim}D tensor")
    if end_dim < 0 or end_dim >= ndim:
        raise ValueError(f"end_dim {end_dim} out of range for {ndim}D tensor")
    if start_dim > end_dim:
        raise ValueError(f"start_dim {start_dim} must be <= end_dim {end_dim}")

    # Calculate new shape
    if start_dim == 0 and end_dim == ndim - 1:
        # Flatten everything
        new_shape = [input.size]
    else:
        # Flatten only the specified dimensions
        new_shape = shape[:start_dim]
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]
        new_shape.append(flattened_size)
        new_shape.extend(shape[end_dim + 1:])

    if out is None:
        out = Tensor(size=input.size, dtype=input._dtype, shape=new_shape)

    # Copy data
    out.from_list(input.tolist())
    return out


def norm(input: Tensor, p: Union[float, str] = 2, dim: Optional[int] = None,
         keepdim: bool = False, out: Optional[Tensor] = None) -> Tensor:
    """
    Calculate vector or matrix norm.

    Similar to PyTorch's torch.norm().

    Args:
        input: Input tensor
        p: Norm order. Can be:
           - float: p-norm (1, 2, inf, -inf, etc.)
           - 'fro': Frobenius norm (for matrices)
        dim: Dimension along which to compute norm (None = entire tensor)
        keepdim: Whether to keep reduced dimensions
        out: Optional output tensor

    Returns:
        Norm value(s)

    Example:
        >>> a = Tensor([3, 4])
        >>> result = norm(a)  # L2 norm: 5.0
        >>> result = norm(a, p=1)  # L1 norm: 7.0
    """
    _ensure_initialized()

    # For now, implement basic functionality
    if dim is not None:
        raise NotImplementedError("Per-dimension norm not yet implemented")

    data = input.tolist()

    if p == 'fro' or p == 2:
        # Frobenius / L2 norm
        sum_sq = sum(x * x for x in data)
        result = sum_sq ** 0.5
    elif p == 1:
        # L1 norm
        result = sum(abs(x) for x in data)
    elif p == float('inf'):
        # Infinity norm (max absolute value)
        result = max(abs(x) for x in data)
    elif p == float('-inf'):
        # Negative infinity norm (min absolute value)
        result = min(abs(x) for x in data)
    else:
        # General p-norm
        result = sum(abs(x) ** p for x in data) ** (1.0 / p)

    # Return scalar tensor
    if out is None:
        out = Tensor([result], dtype=input._dtype)
    else:
        out.from_list([result])

    return out


def arange(start: Union[int, float], end: Optional[Union[int, float]] = None,
           step: Union[int, float] = 1, dtype: Optional[DataType] = None) -> Tensor:
    """
    Create a tensor with evenly spaced values in a range.

    Similar to PyTorch's torch.arange() and NumPy's arange().

    Args:
        start: Starting value (or end if 'end' is None)
        end: End value (exclusive)
        step: Step between values
        dtype: Data type (default: float32)

    Returns:
        Tensor with range values

    Example:
        >>> result = arange(5)  # [0, 1, 2, 3, 4]
        >>> result = arange(2, 10, 2)  # [2, 4, 6, 8]
        >>> result = arange(0, 1, 0.1)  # [0.0, 0.1, 0.2, ..., 0.9]
    """
    _ensure_initialized()

    # Handle single argument case
    if end is None:
        end = start
        start = 0

    # Calculate number of elements
    if step == 0:
        raise ValueError("step cannot be zero")

    n = int((end - start) / step)
    if n < 0:
        n = 0

    # Generate values
    values = [start + i * step for i in range(n)]

    if dtype is None:
        dtype = float32

    return Tensor(values, dtype=dtype)


def linspace(start: Union[int, float], end: Union[int, float], steps: int = 100,
             dtype: Optional[DataType] = None) -> Tensor:
    """
    Create a tensor with evenly spaced values between start and end.

    Similar to PyTorch's torch.linspace() and NumPy's linspace().

    Args:
        start: Starting value
        end: End value (inclusive)
        steps: Number of values to generate
        dtype: Data type (default: float32)

    Returns:
        Tensor with linearly spaced values

    Example:
        >>> result = linspace(0, 10, 5)  # [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> result = linspace(1, 2, 3)  # [1.0, 1.5, 2.0]
    """
    _ensure_initialized()

    if steps < 1:
        raise ValueError("steps must be at least 1")

    if steps == 1:
        values = [start]
    else:
        step_size = (end - start) / (steps - 1)
        values = [start + i * step_size for i in range(steps)]

    if dtype is None:
        dtype = float32

    return Tensor(values, dtype=dtype)


def zeros(*size, dtype: Optional[DataType] = None, shape: Optional[List[int]] = None) -> Tensor:
    """
    Create a tensor filled with zeros.

    Similar to PyTorch's torch.zeros() and NumPy's zeros().

    Args:
        *size: Size specification (can be single int or multiple ints for shape)
        dtype: Data type (default: float32)
        shape: Alternative way to specify shape

    Returns:
        Tensor filled with zeros

    Example:
        >>> result = zeros(5)  # [0, 0, 0, 0, 0]
        >>> result = zeros(2, 3)  # [[0, 0, 0], [0, 0, 0]]
        >>> result = zeros(shape=(2, 3))
    """
    _ensure_initialized()

    if shape is None:
        if len(size) == 0:
            raise ValueError("Must specify size or shape")
        elif len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = list(size[0])
        else:
            shape = list(size)

    # Calculate total size
    total_size = 1
    for dim in shape:
        total_size *= dim

    if dtype is None:
        dtype = float32

    return Tensor([0] * total_size, dtype=dtype, shape=shape)


def ones(*size, dtype: Optional[DataType] = None, shape: Optional[List[int]] = None) -> Tensor:
    """
    Create a tensor filled with ones.

    Similar to PyTorch's torch.ones() and NumPy's ones().

    Args:
        *size: Size specification (can be single int or multiple ints for shape)
        dtype: Data type (default: float32)
        shape: Alternative way to specify shape

    Returns:
        Tensor filled with ones

    Example:
        >>> result = ones(5)  # [1, 1, 1, 1, 1]
        >>> result = ones(2, 3)  # [[1, 1, 1], [1, 1, 1]]
        >>> result = ones(shape=(2, 3))
    """
    _ensure_initialized()

    if shape is None:
        if len(size) == 0:
            raise ValueError("Must specify size or shape")
        elif len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = list(size[0])
        else:
            shape = list(size)

    # Calculate total size
    total_size = 1
    for dim in shape:
        total_size *= dim

    if dtype is None:
        dtype = float32

    return Tensor([1] * total_size, dtype=dtype, shape=shape)


def full(*size, fill_value: Union[int, float] = 0, dtype: Optional[DataType] = None,
         shape: Optional[List[int]] = None) -> Tensor:
    """
    Create a tensor filled with a specific value.

    Similar to PyTorch's torch.full() and NumPy's full().

    Args:
        *size: Size specification (can be single int or multiple ints for shape)
        fill_value: Value to fill the tensor with
        dtype: Data type (default: float32)
        shape: Alternative way to specify shape

    Returns:
        Tensor filled with fill_value

    Example:
        >>> result = full(5, fill_value=3.14)  # [3.14, 3.14, 3.14, 3.14, 3.14]
        >>> result = full(2, 3, fill_value=7)  # [[7, 7, 7], [7, 7, 7]]
    """
    _ensure_initialized()

    if shape is None:
        if len(size) == 0:
            raise ValueError("Must specify size or shape")
        elif len(size) == 1 and isinstance(size[0], (list, tuple)):
            shape = list(size[0])
        else:
            shape = list(size)

    # Calculate total size
    total_size = 1
    for dim in shape:
        total_size *= dim

    if dtype is None:
        dtype = float32

    return Tensor([fill_value] * total_size, dtype=dtype, shape=shape)


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
    "DType",
    "float32",
    "float16",
    "uint32",
    "int32",
    "OperatorEnum",
    "ShaderEnum",
    # Classes
    "Tensor",
    "ThreadSize",
    "ComputeHandle",
    "ComputeBinding",
    "ComputeDispatch",
    # Core functions
    "get_shader",
    # Special operations
    "clamp",
    "where",
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
    # Tensor manipulation
    "concat",
    "stack",
    "flatten",
    # Norms
    "norm",
    # Tensor creation
    "arange",
    "linspace",
    "zeros",
    "ones",
    "full",
] + ALL_OPERATION_NAMES
