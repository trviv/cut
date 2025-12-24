"""
CUT CPU Backend - CPU Compute Library

A Python interface for CPU-based compute operations.
Uses multi-threading for parallel execution.
"""

import atexit
import weakref
import numpy as np
from typing import Optional, Callable
from . import _cut_cpu

__version__ = "0.1.0"

# Module-level CPU interface (lazy initialization)
_interface: Optional[_cut_cpu.CPUCompute] = None

# Shader cache: maps ShaderEnum -> ComputeHandle
_shader_cache: dict = {}

# Kernel cache: maps ShaderEnum -> registered kernel
_kernel_cache: dict = {}

# Track all live buffers using weak references
_live_buffers: weakref.WeakSet = weakref.WeakSet()


def _cleanup():
    """Clean up resources."""
    global _shader_cache, _kernel_cache, _interface
    for buf in list(_live_buffers):
        buf._handle = None
    _shader_cache.clear()
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
                 dtype: Optional[np.dtype] = None, shape: Optional[tuple] = None):
        """
        Create a CPU buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            dtype: Data type for the buffer (used when creating from size)
            shape: Shape for the buffer (used when creating from size)
        """
        _ensure_initialized()
        if data is not None:
            data = np.ascontiguousarray(data)
            self._handle = _interface.create_buffer(data)
            self._size = data.nbytes
            self._dtype = data.dtype
            self._shape = data.shape
        elif size is not None:
            self._handle = _interface.create_buffer_empty(size)
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
        """Copy data from numpy array to buffer."""
        data = np.ascontiguousarray(data)
        _interface.copy_to_buffer(self._handle, data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy data from buffer to numpy array."""
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


class Dispatch:
    """Compute dispatch configuration for CPU."""

    def __init__(self, shader_handle, iterations: int = 1):
        """
        Create a compute dispatch.

        Args:
            shader_handle: Shader handle
            iterations: Number of iterations (workgroup size x)
        """
        self._dispatch = _cut_cpu.ComputeDispatch(shader_handle)
        self._dispatch.set_workgroup_size(_cut_cpu.ThreadSize(iterations, 1, 1))
        self._bindings = []

    def bind(self, resource, binding: int) -> "Dispatch":
        """Bind a resource to a binding point."""
        if isinstance(resource, Buffer):
            self._dispatch.bind_resource(resource.handle, binding)
        elif isinstance(resource, np.ndarray):
            data = np.ascontiguousarray(resource)
            self._dispatch.bind_data(data, binding)
            self._bindings.append(data)
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
    """Execute one or more compute dispatches."""
    _ensure_initialized()
    _interface.begin_command_buffer()
    for d in dispatches:
        _interface.encode(d.inner)
    cmd = _interface.end_command_buffer()
    _interface.submit(cmd)
    _interface.wait(cmd)


# =============================================================================
# Built-in CPU kernels matching the GPU shaders
# =============================================================================

def _make_binary_kernel(op_func):
    """Create a binary operation kernel."""
    def kernel(idx: int, bindings: list, pc_ptr: int):
        import ctypes
        a_ptr = bindings[0]
        b_ptr = bindings[1]
        out_ptr = bindings[2]
        n = ctypes.cast(pc_ptr, ctypes.POINTER(ctypes.c_uint32)).contents.value

        base = idx * 4
        for i in range(4):
            if base + i < n:
                a_val = ctypes.cast(a_ptr + (base + i) * 4, ctypes.POINTER(ctypes.c_float)).contents.value
                b_val = ctypes.cast(b_ptr + (base + i) * 4, ctypes.POINTER(ctypes.c_float)).contents.value
                result = op_func(a_val, b_val)
                ctypes.cast(out_ptr + (base + i) * 4, ctypes.POINTER(ctypes.c_float)).contents.value = result
    return kernel


def _make_unary_kernel(op_func):
    """Create a unary operation kernel."""
    def kernel(idx: int, bindings: list, pc_ptr: int):
        import ctypes
        in_ptr = bindings[0]
        out_ptr = bindings[1]
        n = ctypes.cast(pc_ptr, ctypes.POINTER(ctypes.c_uint32)).contents.value

        base = idx * 4
        for i in range(4):
            if base + i < n:
                in_val = ctypes.cast(in_ptr + (base + i) * 4, ctypes.POINTER(ctypes.c_float)).contents.value
                result = op_func(in_val)
                ctypes.cast(out_ptr + (base + i) * 4, ctypes.POINTER(ctypes.c_float)).contents.value = result
    return kernel


def _register_builtin_kernels():
    """Register built-in kernels for all shader operations."""
    import math

    # Binary arithmetic operations
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecAdd] = _make_binary_kernel(lambda a, b: a + b)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecSub] = _make_binary_kernel(lambda a, b: a - b)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecMul] = _make_binary_kernel(lambda a, b: a * b)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecDiv] = _make_binary_kernel(lambda a, b: a / b)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecMod] = _make_binary_kernel(lambda a, b: a - b * math.floor(a / b))
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecPow] = _make_binary_kernel(lambda a, b: math.pow(a, b))
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecFloorDiv] = _make_binary_kernel(lambda a, b: math.floor(a / b))

    # Binary comparison operations
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecEqual] = _make_binary_kernel(lambda a, b: 1.0 if a == b else 0.0)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecNotEqual] = _make_binary_kernel(lambda a, b: 1.0 if a != b else 0.0)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecLess] = _make_binary_kernel(lambda a, b: 1.0 if a < b else 0.0)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecLessEqual] = _make_binary_kernel(lambda a, b: 1.0 if a <= b else 0.0)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecGreater] = _make_binary_kernel(lambda a, b: 1.0 if a > b else 0.0)
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecGreaterEqual] = _make_binary_kernel(lambda a, b: 1.0 if a >= b else 0.0)

    # Binary min/max operations
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecMin] = _make_binary_kernel(lambda a, b: min(a, b))
    _kernel_cache[_cut_cpu.ShaderEnum.BinaryVecVecMax] = _make_binary_kernel(lambda a, b: max(a, b))

    # Unary operations
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryNeg] = _make_unary_kernel(lambda x: -x)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryAbs] = _make_unary_kernel(abs)
    _kernel_cache[_cut_cpu.ShaderEnum.UnarySqrt] = _make_unary_kernel(math.sqrt)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryExp] = _make_unary_kernel(math.exp)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryLog] = _make_unary_kernel(math.log)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryLog2] = _make_unary_kernel(math.log2)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryLog10] = _make_unary_kernel(math.log10)
    _kernel_cache[_cut_cpu.ShaderEnum.UnarySin] = _make_unary_kernel(math.sin)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryCos] = _make_unary_kernel(math.cos)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryTan] = _make_unary_kernel(math.tan)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryAsin] = _make_unary_kernel(math.asin)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryAcos] = _make_unary_kernel(math.acos)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryAtan] = _make_unary_kernel(math.atan)
    _kernel_cache[_cut_cpu.ShaderEnum.UnarySinh] = _make_unary_kernel(math.sinh)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryCosh] = _make_unary_kernel(math.cosh)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryTanh] = _make_unary_kernel(math.tanh)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryFloor] = _make_unary_kernel(math.floor)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryCeil] = _make_unary_kernel(math.ceil)
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryRound] = _make_unary_kernel(round)
    _kernel_cache[_cut_cpu.ShaderEnum.UnarySign] = _make_unary_kernel(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    _kernel_cache[_cut_cpu.ShaderEnum.UnaryReciprocal] = _make_unary_kernel(lambda x: 1.0 / x)
    _kernel_cache[_cut_cpu.ShaderEnum.UnarySquare] = _make_unary_kernel(lambda x: x * x)


# =============================================================================
# High-level operations using direct NumPy (much faster than Python kernels)
# =============================================================================

def _binary_op_numpy(a: Buffer, b: Buffer, op_func, out: Optional[Buffer] = None) -> Buffer:
    """Generic binary operation using NumPy directly (fast path)."""
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

    # Read data from buffers
    a_data = a.numpy()
    b_data = b.numpy()

    # Perform operation
    result = op_func(a_data, b_data)

    # Write to output buffer
    if out is None:
        out = Buffer(data=result)
    else:
        out.copy_from(result)

    return out


def _unary_op_numpy(a: Buffer, op_func, out: Optional[Buffer] = None) -> Buffer:
    """Generic unary operation using NumPy directly (fast path)."""
    _ensure_initialized()

    # Read data from buffer
    a_data = a.numpy()

    # Perform operation
    result = op_func(a_data)

    # Write to output buffer
    if out is None:
        out = Buffer(data=result)
    else:
        out.copy_from(result)

    return out


# =============================================================================
# Binary arithmetic operations
# =============================================================================

def add(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Add two buffers element-wise."""
    return _binary_op_numpy(a, b, np.add, out)


def subtract(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Subtract two buffers element-wise."""
    return _binary_op_numpy(a, b, np.subtract, out)


def multiply(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Multiply two buffers element-wise."""
    return _binary_op_numpy(a, b, np.multiply, out)


def divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Divide two buffers element-wise."""
    return _binary_op_numpy(a, b, np.divide, out)


def mod(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Modulo of two buffers element-wise."""
    return _binary_op_numpy(a, b, np.mod, out)


def power(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Power of two buffers element-wise."""
    return _binary_op_numpy(a, b, np.power, out)


def floor_divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor division of two buffers element-wise."""
    return _binary_op_numpy(a, b, np.floor_divide, out)


# =============================================================================
# Binary comparison operations
# =============================================================================

def equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise equality comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x == y).astype(np.float32), out)


def not_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise inequality comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x != y).astype(np.float32), out)


def less(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x < y).astype(np.float32), out)


def less_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than-or-equal comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x <= y).astype(np.float32), out)


def greater(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x > y).astype(np.float32), out)


def greater_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than-or-equal comparison."""
    return _binary_op_numpy(a, b, lambda x, y: (x >= y).astype(np.float32), out)


# =============================================================================
# Binary min/max operations
# =============================================================================

def minimum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise minimum."""
    return _binary_op_numpy(a, b, np.minimum, out)


def maximum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise maximum."""
    return _binary_op_numpy(a, b, np.maximum, out)


# =============================================================================
# Unary operations
# =============================================================================

def negative(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Negate element-wise."""
    return _unary_op_numpy(a, np.negative, out)


def abs(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Absolute value element-wise."""
    return _unary_op_numpy(a, np.abs, out)


def sqrt(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square root element-wise."""
    return _unary_op_numpy(a, np.sqrt, out)


def exp(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Exponential element-wise."""
    return _unary_op_numpy(a, np.exp, out)


def log(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Natural logarithm element-wise."""
    return _unary_op_numpy(a, np.log, out)


def log2(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-2 logarithm element-wise."""
    return _unary_op_numpy(a, np.log2, out)


def log10(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-10 logarithm element-wise."""
    return _unary_op_numpy(a, np.log10, out)


def sin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sine element-wise."""
    return _unary_op_numpy(a, np.sin, out)


def cos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Cosine element-wise."""
    return _unary_op_numpy(a, np.cos, out)


def tan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Tangent element-wise."""
    return _unary_op_numpy(a, np.tan, out)


def arcsin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse sine element-wise."""
    return _unary_op_numpy(a, np.arcsin, out)


def arccos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse cosine element-wise."""
    return _unary_op_numpy(a, np.arccos, out)


def arctan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse tangent element-wise."""
    return _unary_op_numpy(a, np.arctan, out)


def sinh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic sine element-wise."""
    return _unary_op_numpy(a, np.sinh, out)


def cosh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic cosine element-wise."""
    return _unary_op_numpy(a, np.cosh, out)


def tanh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic tangent element-wise."""
    return _unary_op_numpy(a, np.tanh, out)


def floor(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor element-wise."""
    return _unary_op_numpy(a, np.floor, out)


def ceil(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Ceiling element-wise."""
    return _unary_op_numpy(a, np.ceil, out)


def round(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Round element-wise."""
    return _unary_op_numpy(a, np.round, out)


def sign(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sign element-wise (-1, 0, or 1)."""
    return _unary_op_numpy(a, np.sign, out)


def reciprocal(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Reciprocal (1/x) element-wise."""
    return _unary_op_numpy(a, np.reciprocal, out)


def square(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square (x*x) element-wise."""
    return _unary_op_numpy(a, np.square, out)


# Export public API
__all__ = [
    # Classes
    "Buffer",
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
]

# Re-export ShaderEnum
ShaderEnum = _cut_cpu.ShaderEnum
