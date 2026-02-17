"""
CUT Unified Compute Interface

This module provides a unified interface for CUT Vulkan backend.
Initialize with the desired backend using `init()`, then use operations directly.

Example:
    import cut.compute as cc

    # Initialize with Vulkan backend
    cc.init(cc.Backend.Vulkan)

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


# Re-export Backend from C++ binding
Backend = _cut_compute.BackendType
# Add CPU for backward compatibility (CPU backend was removed)
Backend.CPU = -1  # Dummy value for compatibility

# SIMDMode is no longer supported (CPU backend removed)
# Create a stub enum for backward compatibility
class SIMDMode:
    """Deprecated: SIMD mode selection (CPU backend removed)."""
    Auto = 0
    Scalar = 1
    SSE = 2
    AVX = 3
    AVX2 = 4

DataType = _cut_compute.DataType
OperatorEnum = _cut_compute.OperatorEnum
ShaderEnum = _cut_compute.OperatorEnum  # Alias for backward compatibility
ThreadSize = _cut_compute.ThreadSize
ComputeHandle = _cut_compute.ComputeHandle
ComputeBinding = _cut_compute.ComputeBinding


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
    if is_vulkan_available():
        available.append("vulkan")
    if is_cpu_available():
        available.append("cpu")
    return available


def is_vulkan_available() -> bool:
    """Check if Vulkan backend is available."""
    return _cut_compute.is_vulkan_available()


def is_cpu_available() -> bool:
    """Check if CPU backend is available (always False - CPU backend removed)."""
    return False


def init(
    backend: Backend = Backend.Vulkan,
    *,
    num_threads: int = 0,
    simd_mode: SIMDMode = SIMDMode.Auto,
    force: bool = False
) -> Backend:
    """
    Initialize a compute backend.

    Args:
        backend: Backend to use (Backend.Vulkan)
        num_threads: Deprecated (CPU backend removed)
        simd_mode: Deprecated (CPU backend removed)
        force: If True, re-initialize even if already initialized with same backend

    Returns:
        The initialized backend type

    Raises:
        RuntimeError: If requested backend is not available

    Example:
        >>> import cut.compute as cc
        >>> cc.init(cc.Backend.Vulkan)  # Initialize Vulkan
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

    if backend == Backend.CPU:
        raise RuntimeError("CPU backend is no longer supported (removed)")

    if backend == Backend.Vulkan and not is_vulkan_available():
        raise RuntimeError("Vulkan backend is not available")

    # Note: num_threads and simd_mode parameters are deprecated (CPU backend removed)
    # They are kept in the function signature for backward compatibility
    _cut_compute.init(backend)
    _initialized = True

    return current_backend()


def _ensure_initialized():
    """Ensure a backend is initialized."""
    global _initialized
    if not _initialized:
        # Auto-initialize with Vulkan backend (CPU backend removed)
        init(Backend.Vulkan)


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
        True if using GPU backend (Vulkan), False otherwise
    """
    return current_backend() == Backend.Vulkan


def num_threads() -> int:
    """
    Get number of worker threads (deprecated - CPU backend removed).

    Returns:
        Always returns 0
    """
    _ensure_initialized()
    return _cut_compute.num_threads()


def simd_mode() -> SIMDMode:
    """
    Get current SIMD mode (deprecated - CPU backend removed).

    Returns:
        Current SIMD mode
    """
    _ensure_initialized()
    return _cut_compute.simd_mode()


def set_simd_mode(mode: SIMDMode):
    """
    Set SIMD mode (deprecated - CPU backend removed).

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

# Reverse mapping from CUT DataType to DType
_CUT_TO_DTYPE = {
    DataType.Float32: DType.float32,
    DataType.Float16: DType.float16,
    DataType.UInt32: DType.uint32,
    DataType.Int32: DType.int32,
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
            # Create buffer with correct multidimensional shape so the GPU
            # buffer knows the innermost dimension for alignment/stride
            # calculations.  Using create_buffer_empty + copy_to_buffer
            # instead of create_buffer (which infers a flat 1D shape from
            # the array.array) ensures dim-reduction strides are correct.
            self._handle = _cut_compute.create_buffer_empty(
                list(self._shape), resolved_dtype.to_cut_dtype(), is_uniform
            )
            _cut_compute.copy_to_buffer(self._handle, arr)
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

    def _to_view(self):
        """Return the Tensor handle for use with C++ operations."""
        return self._handle

    @classmethod
    def _from_view(cls, handle, py_dtype=None, shape=None):
        """Create a Tensor from a handle returned by an operation."""
        t = object.__new__(cls)
        t._handle = handle
        if shape is not None:
            buf_shape = list(shape)
        else:
            buf_shape = _cut_compute.get_buffer_shape(handle)
            # Strip trailing 1s from the 4-dim padded shape
            while len(buf_shape) > 1 and buf_shape[-1] == 1:
                buf_shape = buf_shape[:-1]
        t._shape = tuple(buf_shape)
        if py_dtype is not None:
            t._dtype = py_dtype
        else:
            t._dtype = _CUT_TO_DTYPE[_cut_compute.get_buffer_dtype(handle)]
        t._size = _cut_compute.get_buffer_size_bytes(handle)
        _live_tensors.add(t)
        return t

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

    def item(self) -> Union[float, int]:
        """
        Extract the scalar value from a single-element tensor.

        Returns:
            Python scalar (float or int depending on dtype)
        """
        arr = self.copy_to()
        return arr[0]

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

    def __repr__(self) -> str:
        data = self.tolist()
        dtype_str = str(self._dtype)
        return f"Tensor({_format_tensor(data)}, dtype={dtype_str})"

    def __str__(self) -> str:
        return self.__repr__()


def _format_tensor(data, indent=0) -> str:
    """Format nested list data like numpy array printing."""
    if not isinstance(data, list):
        # Scalar: format floats cleanly, rounding to ~8 significant digits
        # like numpy's default float32 display
        if isinstance(data, float):
            return f"{data:.8g}"
        return str(data)

    if not data:
        return "[]"

    # 1D list
    if not isinstance(data[0], list):
        items = ", ".join(_format_tensor(x) for x in data)
        return f"[{items}]"

    # Multi-dimensional: each sub-array on its own line
    sep = ",\n" + " " * (indent + 1)
    items = sep.join(_format_tensor(row, indent + 1) for row in data)
    return f"[{items}]"


# Save reference to builtin sum before we shadow it
import builtins as _builtins
builtins_sum = _builtins.sum

# =============================================================================
# Operation Implementations - using C++ Operations layer
# =============================================================================

def _create_binary_op(op_enum: OperatorEnum):
    """Create a binary vec-vec operation function using C++ backend."""
    def binary_op(a: Tensor, b: Tensor, out: 'Tensor' = None) -> Tensor:
        _ensure_initialized()
        result = _cut_compute.ops_binary(op_enum, a._to_view(), b._to_view())
        if out is not None:
            out._handle = result
            buf_shape = _cut_compute.get_buffer_shape(result)
            while len(buf_shape) > 1 and buf_shape[-1] == 1:
                buf_shape = buf_shape[:-1]
            out._shape = tuple(buf_shape)
            return out
        return Tensor._from_view(result, a._dtype)
    return binary_op


def _create_unary_op(op_enum: OperatorEnum):
    """Create a unary operation function using C++ backend."""
    def unary_op(a: Tensor) -> Tensor:
        _ensure_initialized()
        result = _cut_compute.ops_unary(op_enum, a._to_view())
        return Tensor._from_view(result, a._dtype)
    return unary_op


def _create_binary_vec_scalar_op(op_enum: OperatorEnum):
    """Create a binary vec-scalar operation function using C++ backend."""
    def vec_scalar_op(a: Tensor, scalar: Union[int, float]) -> Tensor:
        _ensure_initialized()
        result = _cut_compute.ops_vec_scalar(op_enum, a._to_view(), scalar)
        return Tensor._from_view(result, a._dtype)
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
# Special Operations - using C++ Operations layer
# =============================================================================

def sum(a: Tensor, dim: Optional[int] = None) -> Union[float, 'Tensor']:
    """
    Compute the sum of elements in the tensor, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: scalar sum of all elements.
        If dim is specified: Tensor with the given dimension reduced.

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = sum(a)  # Returns 10.0
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> sum(x, dim=0)  # Returns Tensor([5, 7, 9])
        >>> sum(x, dim=1)  # Returns Tensor([6, 15])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceSum, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def mean(a: Tensor, dim: Optional[int] = None) -> Union[float, 'Tensor']:
    """
    Compute the mean of elements in the tensor, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: scalar mean of all elements.
        If dim is specified: Tensor with the given dimension reduced.

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = mean(a)  # Returns 2.5
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> mean(x, dim=0)  # Returns Tensor([2.5, 3.5, 4.5])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceMean, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def min(a: Tensor, dim: Optional[int] = None) -> Union[float, 'Tensor']:
    """
    Find the minimum element in the tensor, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: scalar minimum.
        If dim is specified: Tensor with the given dimension reduced.

    Example:
        >>> a = Tensor([3, 1, 4, 1, 5])
        >>> result = min(a)  # Returns 1.0
        >>> x = Tensor([[3, 1], [4, 2]])
        >>> min(x, dim=0)  # Returns Tensor([3, 1])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceMin, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def max(a: Tensor, dim: Optional[int] = None) -> Union[float, 'Tensor']:
    """
    Find the maximum element in the tensor, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: scalar maximum.
        If dim is specified: Tensor with the given dimension reduced.

    Example:
        >>> a = Tensor([3, 1, 4, 1, 5])
        >>> result = max(a)  # Returns 5.0
        >>> x = Tensor([[3, 1], [4, 2]])
        >>> max(x, dim=0)  # Returns Tensor([4, 2])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceMax, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def prod(a: Tensor, dim: Optional[int] = None) -> Union[float, 'Tensor']:
    """
    Compute the product of elements in the tensor, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: scalar product of all elements.
        If dim is specified: Tensor with the given dimension reduced.

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = prod(a)  # Returns 24.0
        >>> x = Tensor([[1, 2], [3, 4]])
        >>> prod(x, dim=0)  # Returns Tensor([3, 8])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceProd, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def any(a: Tensor, dim: Optional[int] = None) -> Union[bool, 'Tensor']:
    """
    Check if any element in the tensor is non-zero, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: True if any element is non-zero.
        If dim is specified: Tensor with 1.0 where any element along dim is non-zero.

    Example:
        >>> a = Tensor([0, 0, 1, 0])
        >>> result = any(a)  # Returns True
        >>> x = Tensor([[0, 1], [0, 0]])
        >>> any(x, dim=0)  # Returns Tensor([0, 1])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceAny, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item() != 0


def all(a: Tensor, dim: Optional[int] = None) -> Union[bool, 'Tensor']:
    """
    Check if all elements in the tensor are non-zero, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to reduce. If None, reduces all elements.

    Returns:
        If dim is None: True if all elements are non-zero.
        If dim is specified: Tensor with 1.0 where all elements along dim are non-zero.

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> result = all(a)  # Returns True
        >>> x = Tensor([[1, 0], [1, 1]])
        >>> all(x, dim=0)  # Returns Tensor([1, 0])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceAll, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item() != 0


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
    result = _cut_compute.ops_matmul(a._to_view(), b._to_view())
    return Tensor._from_view(result, float32)


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
    result = _cut_compute.ops_transpose(a._to_view())
    return Tensor._from_view(result, float32)


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
    result = _cut_compute.ops_dot(a._to_view(), b._to_view())
    t = Tensor._from_view(result, a._dtype)
    return t.item()


def conv1d(input: Tensor, weight: Tensor,
           stride: int = 1, padding: int = 0) -> Tensor:
    """
    1D convolution over an input signal.

    Args:
        input: Input tensor of shape (N, C_in, L_in)
        weight: Filters of shape (C_out, C_in, kL)
        stride: Stride of the convolution
        padding: Zero-padding added to both sides

    Returns:
        Output tensor of shape (N, C_out, L_out)
    """
    _ensure_initialized()
    result = _cut_compute.ops_conv1d(
        input._to_view(), weight._to_view(),
        int(stride), int(padding))
    return Tensor._from_view(result, input._dtype)


def conv2d(input: Tensor, weight: Tensor,
           stride: Union[int, tuple] = 1,
           padding: Union[int, tuple] = 0) -> Tensor:
    """
    2D convolution over an input image.

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Filters of shape (C_out, C_in, kH, kW)
        stride: Stride of the convolution (int or (sH, sW))
        padding: Zero-padding (int or (pH, pW))

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    _ensure_initialized()
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    result = _cut_compute.ops_conv2d(
        input._to_view(), weight._to_view(),
        int(sh), int(sw), int(ph), int(pw))
    return Tensor._from_view(result, input._dtype)


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
    result = _cut_compute.ops_clamp(a._to_view(), float(min_val), float(max_val))
    return Tensor._from_view(result, a._dtype)


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
    result = _cut_compute.ops_where(condition._to_view(), x._to_view(), y._to_view())
    return Tensor._from_view(result, x._dtype)


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
        return input
    result = _cut_compute.ops_flatten(input._to_view(), start_dim, end_dim)
    exact_shape = _cut_compute.get_buffer_shape(result)
    return Tensor._from_view(result, input._dtype, shape=exact_shape)


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

    if p == 2 or p == 'fro':
        result = _cut_compute.ops_norm(input._to_view(), dim)
        t = Tensor._from_view(result, input._dtype)
        return t if dim is not None else t
    elif dim is not None:
        raise NotImplementedError("Per-dimension norm only supports p=2 (L2 norm)")

    data = input.tolist()

    if p == 1:
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

    if dtype is None:
        dtype = float32

    result = _cut_compute.ops_arange(float(start), float(end), float(step), dtype.to_cut_dtype())
    return Tensor._from_view(result, dtype)


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

    if dtype is None:
        dtype = float32

    result = _cut_compute.ops_linspace(float(start), float(end), steps, dtype.to_cut_dtype())
    return Tensor._from_view(result, dtype)


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

    if dtype is None:
        dtype = float32

    result = _cut_compute.ops_full([int(s) for s in shape], 0.0, dtype.to_cut_dtype())
    return Tensor._from_view(result, dtype)


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

    if dtype is None:
        dtype = float32

    result = _cut_compute.ops_full([int(s) for s in shape], 1.0, dtype.to_cut_dtype())
    return Tensor._from_view(result, dtype)


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

    if dtype is None:
        dtype = float32

    result = _cut_compute.ops_full([int(s) for s in shape], float(fill_value), dtype.to_cut_dtype())
    return Tensor._from_view(result, dtype)


def zeros_like(input: Tensor, dtype: Optional[DType] = None) -> Tensor:
    """
    Create a tensor of zeros with the same shape and dtype as input.

    Args:
        input: Reference tensor
        dtype: Override data type (default: same as input)

    Returns:
        Tensor filled with zeros

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> result = zeros_like(a)  # [[0, 0], [0, 0]]
    """
    d = dtype if dtype is not None else input._dtype
    result = _cut_compute.ops_full(list(input._shape), 0.0, d.to_cut_dtype())
    return Tensor._from_view(result, d)


def ones_like(input: Tensor, dtype: Optional[DType] = None) -> Tensor:
    """
    Create a tensor of ones with the same shape and dtype as input.

    Args:
        input: Reference tensor
        dtype: Override data type (default: same as input)

    Returns:
        Tensor filled with ones

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> result = ones_like(a)  # [[1, 1], [1, 1]]
    """
    d = dtype if dtype is not None else input._dtype
    result = _cut_compute.ops_full(list(input._shape), 1.0, d.to_cut_dtype())
    return Tensor._from_view(result, d)


def full_like(input: Tensor, fill_value: Union[int, float], dtype: Optional[DType] = None) -> Tensor:
    """
    Create a tensor filled with fill_value, same shape and dtype as input.

    Args:
        input: Reference tensor
        fill_value: Value to fill with
        dtype: Override data type (default: same as input)

    Returns:
        Tensor filled with fill_value

    Example:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> result = full_like(a, 7.0)  # [[7, 7], [7, 7]]
    """
    d = dtype if dtype is not None else input._dtype
    result = _cut_compute.ops_full(list(input._shape), float(fill_value), d.to_cut_dtype())
    return Tensor._from_view(result, d)


def var(a: Tensor, dim: Optional[int] = None, correction: int = 1) -> Union[float, 'Tensor']:
    """
    Compute the variance of elements, optionally along a dimension.

    Uses Bessel's correction (N-1) by default, matching PyTorch behavior.

    Args:
        a: Input tensor
        dim: Dimension along which to compute. If None, computes over all elements.
        correction: Degrees of freedom correction (default: 1 for unbiased)

    Returns:
        Variance value or tensor

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> var(a)  # 1.6667 (unbiased)
    """
    _ensure_initialized()
    result = _cut_compute.ops_variance(a._to_view(), correction, dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else t.item()


def std(a: Tensor, dim: Optional[int] = None, correction: int = 1) -> Union[float, 'Tensor']:
    """
    Compute the standard deviation of elements, optionally along a dimension.

    Uses Bessel's correction (N-1) by default, matching PyTorch behavior.

    Args:
        a: Input tensor
        dim: Dimension along which to compute. If None, computes over all elements.
        correction: Degrees of freedom correction (default: 1 for unbiased)

    Returns:
        Standard deviation value or tensor

    Example:
        >>> a = Tensor([1, 2, 3, 4])
        >>> std(a)  # 1.2910
    """
    _ensure_initialized()
    v = var(a, dim=dim, correction=correction)
    if dim is not None:
        return globals()['sqrt'](v)
    import math
    return math.sqrt(v)


def softmax(a: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along a dimension.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        a: Input tensor
        dim: Dimension along which to compute softmax (default: -1, last dim)

    Returns:
        Tensor with softmax probabilities

    Example:
        >>> a = Tensor([[1, 2, 3], [1, 2, 3]])
        >>> result = softmax(a, dim=1)
    """
    _ensure_initialized()
    result = _cut_compute.ops_softmax(a._to_view(), dim)
    return Tensor._from_view(result, a._dtype)


def log_softmax(a: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log softmax along a dimension.

    log_softmax(x_i) = x_i - log(sum(exp(x_j)))

    Args:
        a: Input tensor
        dim: Dimension along which to compute (default: -1, last dim)

    Returns:
        Tensor with log softmax values

    Example:
        >>> a = Tensor([[1, 2, 3], [1, 2, 3]])
        >>> result = log_softmax(a, dim=1)
    """
    _ensure_initialized()
    result = _cut_compute.ops_log_softmax(a._to_view(), dim)
    return Tensor._from_view(result, a._dtype)


def cumsum(a: Tensor, dim: int = 0) -> Tensor:
    """
    Compute the cumulative sum along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to compute cumulative sum (default: 0)

    Returns:
        Tensor with cumulative sums

    Example:
        >>> a = Tensor([1.0, 2.0, 3.0, 4.0])
        >>> cumsum(a)  # Returns [1, 3, 6, 10]
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> cumsum(x, dim=0)  # Returns [[1, 2, 3], [5, 7, 9]]
    """
    _ensure_initialized()
    result = _cut_compute.ops_cumulative(a._to_view(), OperatorEnum.CumSum, dim)
    return Tensor._from_view(result, a._dtype)


def cumprod(a: Tensor, dim: int = 0) -> Tensor:
    """
    Compute the cumulative product along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to compute cumulative product (default: 0)

    Returns:
        Tensor with cumulative products

    Example:
        >>> a = Tensor([1.0, 2.0, 3.0, 4.0])
        >>> cumprod(a)  # Returns [1, 2, 6, 24]
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> cumprod(x, dim=0)  # Returns [[1, 2, 3], [4, 10, 18]]
    """
    _ensure_initialized()
    result = _cut_compute.ops_cumulative(a._to_view(), OperatorEnum.CumProd, dim)
    return Tensor._from_view(result, a._dtype)


def argmax(a: Tensor, dim: Optional[int] = None) -> Union[int, 'Tensor']:
    """
    Return the index of the maximum value, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to find argmax. If None, returns
             the index into the flattened tensor.

    Returns:
        If dim is None: integer index of the maximum element.
        If dim is specified: Tensor of indices along that dimension.

    Example:
        >>> a = Tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        >>> argmax(a)  # Returns 4
        >>> x = Tensor([[1.0, 3.0], [4.0, 2.0]])
        >>> argmax(x, dim=0)  # Returns Tensor([1, 0])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceArgmax, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else int(t.item())


def argmin(a: Tensor, dim: Optional[int] = None) -> Union[int, 'Tensor']:
    """
    Return the index of the minimum value, optionally along a dimension.

    Args:
        a: Input tensor
        dim: Dimension along which to find argmin. If None, returns
             the index into the flattened tensor.

    Returns:
        If dim is None: integer index of the minimum element.
        If dim is specified: Tensor of indices along that dimension.

    Example:
        >>> a = Tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        >>> argmin(a)  # Returns 1
        >>> x = Tensor([[1.0, 3.0], [4.0, 2.0]])
        >>> argmin(x, dim=0)  # Returns Tensor([0, 1])
    """
    _ensure_initialized()
    result = _cut_compute.ops_reduce(OperatorEnum.ReduceArgmin, a._to_view(), dim)
    t = Tensor._from_view(result, a._dtype)
    return t if dim is not None else int(t.item())


def reshape(a: Tensor, *shape) -> Tensor:
    """
    Reshape a tensor to a new shape without changing its data.

    Returns a view sharing the same GPU buffer. Supports -1 for one
    inferred dimension.

    Args:
        a: Input tensor
        *shape: New shape (can be individual ints or a single tuple/list)

    Returns:
        Reshaped tensor (view)

    Example:
        >>> a = Tensor([1, 2, 3, 4, 5, 6])
        >>> reshape(a, 2, 3)  # Shape: (2, 3)
        >>> reshape(a, (3, -1))  # Shape: (3, 2)
    """
    _ensure_initialized()

    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        new_shape = list(shape[0])
    else:
        new_shape = list(shape)

    result = _cut_compute.ops_reshape(a._to_view(), [int(s) for s in new_shape])
    exact_shape = _cut_compute.get_buffer_shape(result)
    return Tensor._from_view(result, a._dtype, shape=exact_shape)


def view(a: Tensor, *shape) -> Tensor:
    """
    Return a new tensor with the same data but different shape.

    Alias for reshape(). Matches PyTorch's Tensor.view() semantics.

    Args:
        a: Input tensor
        *shape: New shape

    Returns:
        Reshaped tensor (view)
    """
    return reshape(a, *shape)


def squeeze(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Remove size-1 dimensions from a tensor's shape.

    Args:
        a: Input tensor
        dim: If given, only squeeze that specific dimension
             (no-op if that dim is not size 1).
             If None, squeeze all size-1 dimensions.

    Returns:
        Squeezed tensor (view)

    Example:
        >>> a = Tensor([1, 2, 3], shape=(1, 3, 1))
        >>> squeeze(a)  # Shape: (3,)
        >>> squeeze(a, dim=0)  # Shape: (3, 1)
    """
    _ensure_initialized()
    result = _cut_compute.ops_squeeze(a._to_view(), dim)
    exact_shape = _cut_compute.get_buffer_shape(result)
    return Tensor._from_view(result, a._dtype, shape=exact_shape)


def unsqueeze(a: Tensor, dim: int) -> Tensor:
    """
    Insert a dimension of size 1 at the specified position.

    Args:
        a: Input tensor
        dim: Position at which to insert the new dimension.
             Supports negative indexing.

    Returns:
        Tensor with added dimension (view)

    Example:
        >>> a = Tensor([1, 2, 3])  # Shape: (3,)
        >>> unsqueeze(a, 0)  # Shape: (1, 3)
        >>> unsqueeze(a, 1)  # Shape: (3, 1)
        >>> unsqueeze(a, -1)  # Shape: (3, 1)
    """
    _ensure_initialized()
    result = _cut_compute.ops_unsqueeze(a._to_view(), dim)
    exact_shape = _cut_compute.get_buffer_shape(result)
    return Tensor._from_view(result, a._dtype, shape=exact_shape)


def unflatten(a: Tensor, dim: int, sizes: Sequence[int]) -> Tensor:
    """
    Expand a single dimension into multiple dimensions.

    Args:
        a: Input tensor
        dim: Dimension to unflatten
        sizes: New shape for the unflattened dimension

    Returns:
        Tensor with unflattened dimension (view)

    Example:
        >>> a = Tensor(list(range(12)), shape=(12,))
        >>> unflatten(a, 0, (3, 4))  # Shape: (3, 4)
        >>> b = Tensor(list(range(12)), shape=(2, 6))
        >>> unflatten(b, 1, (2, 3))  # Shape: (2, 2, 3)
    """
    _ensure_initialized()
    result = _cut_compute.ops_unflatten(a._to_view(), dim, [int(s) for s in sizes])
    exact_shape = _cut_compute.get_buffer_shape(result)
    return Tensor._from_view(result, a._dtype, shape=exact_shape)


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Union[float, 'Tensor']:
    """
    Mean Squared Error loss.

    Args:
        input: Predicted values
        target: Target values
        reduction: 'mean' (default), 'sum', or 'none'

    Returns:
        Loss value

    Example:
        >>> pred = Tensor([1.0, 2.0, 3.0])
        >>> target = Tensor([1.5, 2.5, 3.5])
        >>> loss = mse_loss(pred, target)  # 0.25
    """
    _ensure_initialized()
    diff = globals()['subtract'](input, target)
    sq = globals()['square'](diff)
    if reduction == 'none':
        return sq
    elif reduction == 'sum':
        return sum(sq)
    else:
        return mean(sq)


def l1_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Union[float, 'Tensor']:
    """
    Mean Absolute Error (L1) loss.

    Args:
        input: Predicted values
        target: Target values
        reduction: 'mean' (default), 'sum', or 'none'

    Returns:
        Loss value

    Example:
        >>> pred = Tensor([1.0, 2.0, 3.0])
        >>> target = Tensor([1.5, 2.5, 3.5])
        >>> loss = l1_loss(pred, target)  # 0.5
    """
    _ensure_initialized()
    diff = globals()['subtract'](input, target)
    abs_diff = globals()['abs'](diff)
    if reduction == 'none':
        return abs_diff
    elif reduction == 'sum':
        return sum(abs_diff)
    else:
        return mean(abs_diff)


def cross_entropy_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Union[float, 'Tensor']:
    """
    Cross entropy loss (expects raw logits, not probabilities).

    Computes -sum(target * log_softmax(input)) per sample.
    For classification, target should be one-hot encoded.

    Args:
        input: Logits tensor (N, C) or (C,)
        target: Target tensor, same shape as input (one-hot or soft labels)
        reduction: 'mean' (default), 'sum', or 'none'

    Returns:
        Loss value

    Example:
        >>> logits = Tensor([[1.0, 2.0, 3.0]])
        >>> target = Tensor([[0.0, 0.0, 1.0]])
        >>> loss = cross_entropy_loss(logits, target)
    """
    _ensure_initialized()
    log_probs = log_softmax(input, dim=-1)
    neg_log_probs = globals()['negative'](log_probs)
    loss_per_element = globals()['multiply'](target, neg_log_probs)

    if reduction == 'none':
        return loss_per_element
    elif reduction == 'sum':
        return sum(loss_per_element)
    else:
        return mean(loss_per_element)


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
    # Core functions
    "get_shader",
    # Special operations
    "clamp",
    "where",
    # Reduction operations
    "sum",
    "mean",
    "min",
    "max",
    "prod",
    "any",
    "all",
    # Argmax/Argmin
    "argmax",
    "argmin",
    # Cumulative operations
    "cumsum",
    "cumprod",
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
    "zeros_like",
    "ones_like",
    "full_like",
    # Statistical operations
    "var",
    "std",
    # Shape operations
    "reshape",
    "view",
    "squeeze",
    "unsqueeze",
    "unflatten",
    # Softmax
    "softmax",
    "log_softmax",
    # Loss functions
    "mse_loss",
    "l1_loss",
    "cross_entropy_loss",
] + ALL_OPERATION_NAMES
