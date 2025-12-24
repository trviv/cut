"""
CUT CPU Backend - CPU Compute Library

A Python interface for CPU-based compute operations.
Uses multithreaded C++ kernels for parallel execution.
"""

import atexit
import weakref
import numpy as np
from typing import Optional, Union, List
from . import _cut_cpu

__version__ = "0.1.0"

# Module-level CPU interface (lazy initialization)
_interface: Optional[_cut_cpu.CPUCompute] = None

# Kernel cache: maps CPUKernelType -> ComputeHandle
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


def _ensure_initialized(num_threads: int = 0):
    """Ensure CPU interface is initialized."""
    global _interface
    if _interface is None:
        _interface = _cut_cpu.CPUCompute(num_threads)


def get_interface(num_threads: int = 0) -> _cut_cpu.CPUCompute:
    """Get the global CPU compute interface."""
    _ensure_initialized(num_threads)
    return _interface


def num_threads() -> int:
    """Get the number of worker threads."""
    _ensure_initialized()
    return _interface.num_threads()


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
            self._handle = _interface.create_buffer_empty(size, is_uniform)
            self._size = size
            self._dtype = dtype
            self._shape = shape
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

    def __init__(self, kernel_type: _cut_cpu.CPUKernelType):
        """
        Create a kernel from a CPUKernelType.

        Args:
            kernel_type: The CPU kernel type
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
# High-level kernel functions (direct API for built-in kernels)
# =============================================================================

def _binary_op(a: Buffer, b: Buffer, kernel_type, out: Optional[Buffer] = None) -> Buffer:
    """Generic binary operation on CPU."""
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    num_elements = a.size // 4  # float32 is 4 bytes

    kernel = Kernel(kernel_type)
    dispatch = Dispatch(kernel, (num_elements, 1, 1))
    dispatch.bind(a, 0)
    dispatch.bind(b, 1)
    dispatch.bind(out, 2)
    dispatch.bind(num_elements, 3)

    run(dispatch)

    return out


def _unary_op(a: Buffer, kernel_type, out: Optional[Buffer] = None) -> Buffer:
    """Generic unary operation on CPU."""
    _ensure_initialized()

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    num_elements = a.size // 4  # float32 is 4 bytes

    kernel = Kernel(kernel_type)
    dispatch = Dispatch(kernel, (num_elements, 1, 1))
    dispatch.bind(a, 0)
    dispatch.bind(out, 1)
    dispatch.bind(num_elements, 2)

    run(dispatch)

    return out


# =============================================================================
# Binary arithmetic operations
# =============================================================================

def add(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Add two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecAdd, out)


def subtract(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Subtract two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecSub, out)


def multiply(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Multiply two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecMul, out)


def divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Divide two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecDiv, out)


def mod(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Modulo of two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecMod, out)


def power(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Power of two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecPow, out)


def floor_divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor division of two buffers element-wise on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecFloorDiv, out)


# =============================================================================
# Binary comparison operations
# =============================================================================

def equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise equality comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecEqual, out)


def not_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise inequality comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecNotEqual, out)


def less(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecLess, out)


def less_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than-or-equal comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecLessEqual, out)


def greater(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecGreater, out)


def greater_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than-or-equal comparison on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecGreaterEqual, out)


# =============================================================================
# Binary min/max operations
# =============================================================================

def minimum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise minimum of two buffers on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecMin, out)


def maximum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise maximum of two buffers on CPU."""
    return _binary_op(a, b, _cut_cpu.CPUKernelType.BinaryVecVecMax, out)


# =============================================================================
# Unary operations
# =============================================================================

def negative(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Negate buffer element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryNeg, out)


def abs(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Absolute value element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryAbs, out)


def sqrt(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square root element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnarySqrt, out)


def exp(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Exponential element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryExp, out)


def log(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Natural logarithm element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryLog, out)


def log2(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-2 logarithm element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryLog2, out)


def log10(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-10 logarithm element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryLog10, out)


def sin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnarySin, out)


def cos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Cosine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryCos, out)


def tan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Tangent element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryTan, out)


def arcsin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse sine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryAsin, out)


def arccos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse cosine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryAcos, out)


def arctan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse tangent element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryAtan, out)


def sinh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic sine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnarySinh, out)


def cosh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic cosine element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryCosh, out)


def tanh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic tangent element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryTanh, out)


def floor(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryFloor, out)


def ceil(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Ceiling element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryCeil, out)


def round(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Round element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryRound, out)


def sign(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sign element-wise on CPU (-1, 0, or 1)."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnarySign, out)


def reciprocal(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Reciprocal (1/x) element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnaryReciprocal, out)


def square(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square (x*x) element-wise on CPU."""
    return _unary_op(a, _cut_cpu.CPUKernelType.UnarySquare, out)


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
    # Binary arithmetic operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "mod",
    "power",
    "floor_divide",
    # Binary comparison operations
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    # Binary min/max operations
    "minimum",
    "maximum",
    # Unary operations
    "negative",
    "abs",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "sign",
    "reciprocal",
    "square",
    # Re-exports from core
    "CPUKernelType",
]

# Re-export CPUKernelType
CPUKernelType = _cut_cpu.CPUKernelType
