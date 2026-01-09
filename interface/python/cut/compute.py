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
