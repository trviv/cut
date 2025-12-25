"""
CUT (Compute Unified Toolkit) - GPU Compute Library

A Python library for GPU compute operations using Vulkan.
Automatically initializes a Vulkan instance on import.
"""

from __future__ import annotations

import atexit
import weakref
import numpy as np
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import _cut_core as _cut_core_type

__version__ = "0.1.0"

# Try to import Vulkan backend - may fail if not built or on unsupported platform
try:
    from . import _cut_core
    _VULKAN_AVAILABLE = True
except ImportError:
    _cut_core = None
    _VULKAN_AVAILABLE = False

# Module-level Vulkan instance and interface (lazy initialization)
_instance = None
_interface = None

# Shader cache: maps ShaderEnum -> ComputeHandle (VkShaderModule)
_shader_cache: dict = {}

# Track all live buffers using weak references
_live_buffers: weakref.WeakSet = weakref.WeakSet()


def _cleanup():
    """Clean up resources in correct order: buffers -> shaders -> interface -> instance."""
    global _shader_cache, _interface, _instance
    # Invalidate all live buffers first
    for buf in list(_live_buffers):
        buf._handle = None
    _shader_cache.clear()
    _interface = None
    _instance = None


atexit.register(_cleanup)


def _ensure_initialized():
    """Ensure Vulkan instance and interface are initialized."""
    global _instance, _interface
    if not _VULKAN_AVAILABLE:
        raise RuntimeError("Vulkan backend not available. Build with Vulkan support or use the CPU backend.")
    if _instance is None:
        _instance = _cut_core.VulkanInstance()
        _interface = _instance.create_interface()


def precompile_shaders():
    """
    Precompile and cache all built-in shaders.
    Call this once at startup to avoid compilation overhead during operations.
    """
    _ensure_initialized()

    # All shader enums used in benchmarks
    shader_enums = [
        # Binary arithmetic operations
        _cut_core.ShaderEnum.BinaryVecVecAdd,
        _cut_core.ShaderEnum.BinaryVecVecSub,
        _cut_core.ShaderEnum.BinaryVecVecMul,
        _cut_core.ShaderEnum.BinaryVecVecDiv,
        _cut_core.ShaderEnum.BinaryVecVecMod,
        _cut_core.ShaderEnum.BinaryVecVecPow,
        _cut_core.ShaderEnum.BinaryVecVecFloorDiv,
        # Binary comparison operations
        _cut_core.ShaderEnum.BinaryVecVecEqual,
        _cut_core.ShaderEnum.BinaryVecVecNotEqual,
        _cut_core.ShaderEnum.BinaryVecVecLess,
        _cut_core.ShaderEnum.BinaryVecVecLessEqual,
        _cut_core.ShaderEnum.BinaryVecVecGreater,
        _cut_core.ShaderEnum.BinaryVecVecGreaterEqual,
        # Binary min/max operations
        _cut_core.ShaderEnum.BinaryVecVecMin,
        _cut_core.ShaderEnum.BinaryVecVecMax,
        # Unary operations
        _cut_core.ShaderEnum.UnaryNeg,
        _cut_core.ShaderEnum.UnaryAbs,
        _cut_core.ShaderEnum.UnarySqrt,
        _cut_core.ShaderEnum.UnaryExp,
        _cut_core.ShaderEnum.UnaryLog,
        _cut_core.ShaderEnum.UnaryLog2,
        _cut_core.ShaderEnum.UnaryLog10,
        _cut_core.ShaderEnum.UnarySin,
        _cut_core.ShaderEnum.UnaryCos,
        _cut_core.ShaderEnum.UnaryTan,
        _cut_core.ShaderEnum.UnaryAsin,
        _cut_core.ShaderEnum.UnaryAcos,
        _cut_core.ShaderEnum.UnaryAtan,
        _cut_core.ShaderEnum.UnarySinh,
        _cut_core.ShaderEnum.UnaryCosh,
        _cut_core.ShaderEnum.UnaryTanh,
        _cut_core.ShaderEnum.UnaryFloor,
        _cut_core.ShaderEnum.UnaryCeil,
        _cut_core.ShaderEnum.UnaryRound,
        _cut_core.ShaderEnum.UnarySign,
        _cut_core.ShaderEnum.UnaryReciprocal,
        _cut_core.ShaderEnum.UnarySquare,
    ]

    for shader_enum in shader_enums:
        if shader_enum not in _shader_cache:
            spirv_code = _cut_core.get_shader(shader_enum, _cut_core.ScalarDataType.Float)
            handle = _interface.create_shader_module(spirv_code)
            _shader_cache[shader_enum] = handle


def get_interface() -> "_cut_core_type.VulkanCompute":
    """Get the global Vulkan compute interface."""
    _ensure_initialized()
    return _interface


class Buffer:
    """GPU buffer wrapper with automatic memory management."""

    def __init__(self, data: Optional[np.ndarray] = None, size: Optional[int] = None,
                 is_uniform: bool = False, dtype: Optional[np.dtype] = None,
                 shape: Optional[tuple] = None):
        """
        Create a GPU buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            is_uniform: If True, create a uniform buffer; otherwise storage buffer
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
    def handle(self) -> "_cut_core_type.ComputeHandle":
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

    def __init__(self, spirv: Union[List[int], "_cut_core_type.ShaderEnum"]):
        """
        Create a shader module.

        Args:
            spirv: SPIR-V bytecode as list of uint32 or a ShaderEnum for built-in shaders
        """
        _ensure_initialized()
        if isinstance(spirv, _cut_core.ShaderEnum):
            # Check cache first
            if spirv in _shader_cache:
                self._handle = _shader_cache[spirv]
                return
            # Compile and cache
            spirv_code = _cut_core.get_shader(spirv, _cut_core.ScalarDataType.Float)
            self._handle = _interface.create_shader_module(spirv_code)
            _shader_cache[spirv] = self._handle
        else:
            self._handle = _interface.create_shader_module(spirv)

    @property
    def handle(self) -> "_cut_core_type.ComputeHandle":
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
        self._dispatch.set_workgroup_size(
            _cut_core.ThreadSize(thread_groups[0], thread_groups[1], thread_groups[2])
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
    def inner(self) -> "_cut_core_type.ComputeDispatch":
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
# High-level shader functions (direct API for built-in shaders)
# =============================================================================

def _binary_op(a: Buffer, b: Buffer, shader_enum, out: Optional[Buffer] = None) -> Buffer:
    """Generic binary operation on GPU."""
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    num_elements = a.size // 4  # float32 is 4 bytes

    shader = Shader(shader_enum)
    # Pass number of elements; runtime divides by tgSize and dtypeVecSize
    dispatch = Dispatch(shader, (num_elements, 1, 1))
    dispatch.bind(a, 0)
    dispatch.bind(b, 1)
    dispatch.bind(out, 2)
    dispatch.bind(num_elements, 3)

    run(dispatch)

    return out


def _unary_op(a: Buffer, shader_enum, out: Optional[Buffer] = None) -> Buffer:
    """Generic unary operation on GPU."""
    _ensure_initialized()

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    num_elements = a.size // 4  # float32 is 4 bytes

    shader = Shader(shader_enum)
    # Pass number of elements; runtime divides by tgSize and dtypeVecSize
    dispatch = Dispatch(shader, (num_elements, 1, 1))
    dispatch.bind(a, 0)
    dispatch.bind(out, 1)
    dispatch.bind(num_elements, 2)

    run(dispatch)

    return out


# =============================================================================
# Binary arithmetic operations
# =============================================================================

def add(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Add two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecAdd, out)


def subtract(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Subtract two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecSub, out)


def multiply(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Multiply two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecMul, out)


def divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Divide two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecDiv, out)


def mod(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Modulo of two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecMod, out)


def power(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Power of two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecPow, out)


def floor_divide(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor division of two buffers element-wise on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecFloorDiv, out)


# =============================================================================
# Binary comparison operations
# =============================================================================

def equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise equality comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecEqual, out)


def not_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise inequality comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecNotEqual, out)


def less(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecLess, out)


def less_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise less-than-or-equal comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecLessEqual, out)


def greater(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecGreater, out)


def greater_equal(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise greater-than-or-equal comparison on GPU. Returns 1.0 for True, 0.0 for False."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecGreaterEqual, out)


# =============================================================================
# Binary min/max operations
# =============================================================================

def minimum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise minimum of two buffers on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecMin, out)


def maximum(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Element-wise maximum of two buffers on GPU."""
    return _binary_op(a, b, _cut_core.ShaderEnum.BinaryVecVecMax, out)


# =============================================================================
# Unary operations
# =============================================================================

def negative(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Negate buffer element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryNeg, out)


def abs(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Absolute value element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryAbs, out)


def sqrt(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square root element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnarySqrt, out)


def exp(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Exponential element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryExp, out)


def log(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Natural logarithm element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryLog, out)


def log2(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-2 logarithm element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryLog2, out)


def log10(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Base-10 logarithm element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryLog10, out)


def sin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnarySin, out)


def cos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Cosine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryCos, out)


def tan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Tangent element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryTan, out)


def arcsin(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse sine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryAsin, out)


def arccos(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse cosine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryAcos, out)


def arctan(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Inverse tangent element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryAtan, out)


def sinh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic sine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnarySinh, out)


def cosh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic cosine element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryCosh, out)


def tanh(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Hyperbolic tangent element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryTanh, out)


def floor(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Floor element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryFloor, out)


def ceil(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Ceiling element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryCeil, out)


def round(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Round element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryRound, out)


def sign(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Sign element-wise on GPU (-1, 0, or 1)."""
    return _unary_op(a, _cut_core.ShaderEnum.UnarySign, out)


def reciprocal(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Reciprocal (1/x) element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnaryReciprocal, out)


def square(a: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """Square (x*x) element-wise on GPU."""
    return _unary_op(a, _cut_core.ShaderEnum.UnarySquare, out)


def vector_add(a: Buffer, b: Buffer, out: Optional[Buffer] = None) -> Buffer:
    """
    Add two buffers element-wise on the GPU.

    Args:
        a: First input buffer (float32)
        b: Second input buffer (float32)
        out: Output buffer (created if not provided)

    Returns:
        Result buffer (a + b)
    """
    _ensure_initialized()

    if a.size != b.size:
        raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

    if out is None:
        out = Buffer(size=a.size, dtype=a._dtype, shape=a._shape)

    num_elements = a.size // 4  # float32 is 4 bytes

    shader = Shader(_cut_core.ShaderEnum.VECTOR_ADD)
    workgroups = (num_elements + 63) // 64  # 64 threads per workgroup

    dispatch = Dispatch(shader, (workgroups, 1, 1))
    dispatch.bind(a, 0)
    dispatch.bind(b, 1)
    dispatch.bind(out, 2)
    dispatch.bind(num_elements, 3)  # Push constant

    run(dispatch)

    return out


# Export public API
__all__ = [
    # Classes
    "Buffer",
    "Shader",
    "Dispatch",
    # Core functions
    "run",
    "get_interface",
    "precompile_shaders",
    "vector_add",
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
    "ShaderEnum",
]

# Re-export ShaderEnum if available
if _VULKAN_AVAILABLE:
    ShaderEnum = _cut_core.ShaderEnum
else:
    ShaderEnum = None

# Expose availability flag for benchmarks
def is_vulkan_available() -> bool:
    """Check if Vulkan backend is available."""
    return _VULKAN_AVAILABLE
