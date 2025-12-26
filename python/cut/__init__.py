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
import sys

from ._ops import ALL_OPERATION_NAMES, register_operations
from ._base import BaseBuffer, BaseDispatch

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

# Shader cache: maps (OperatorEnum, ScalarDataType) -> ComputeHandle (VkShaderModule)
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
        # Binary arithmetic operations (vec-scalar)
        _cut_core.ShaderEnum.BinaryVecScalarAdd,
        _cut_core.ShaderEnum.BinaryVecScalarSub,
        _cut_core.ShaderEnum.BinaryVecScalarMul,
        _cut_core.ShaderEnum.BinaryVecScalarDiv,
        _cut_core.ShaderEnum.BinaryVecScalarMod,
        _cut_core.ShaderEnum.BinaryVecScalarPow,
        _cut_core.ShaderEnum.BinaryVecScalarFloorDiv,
        # Binary comparison operations (vec-scalar)
        _cut_core.ShaderEnum.BinaryVecScalarEqual,
        _cut_core.ShaderEnum.BinaryVecScalarNotEqual,
        _cut_core.ShaderEnum.BinaryVecScalarLess,
        _cut_core.ShaderEnum.BinaryVecScalarLessEqual,
        _cut_core.ShaderEnum.BinaryVecScalarGreater,
        _cut_core.ShaderEnum.BinaryVecScalarGreaterEqual,
        # Binary min/max operations (vec-scalar)
        _cut_core.ShaderEnum.BinaryVecScalarMin,
        _cut_core.ShaderEnum.BinaryVecScalarMax,
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
        cache_key = (shader_enum, _cut_core.ScalarDataType.Float)
        if cache_key not in _shader_cache:
            spirv_code = _cut_core.get_shader(shader_enum, _cut_core.ScalarDataType.Float)
            handle = _interface.create_shader_module(spirv_code)
            _shader_cache[cache_key] = handle


def get_interface() -> "_cut_core_type.VulkanCompute":
    """Get the global Vulkan compute interface."""
    _ensure_initialized()
    return _interface


class Buffer(BaseBuffer):
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
            self._init_from_data(_interface, data, is_uniform)
        elif size is not None:
            self._init_empty(_interface, _cut_core, size, dtype, shape, is_uniform)
        else:
            raise ValueError("Either data or size must be provided")
        _live_buffers.add(self)

    def _get_module(self):
        """Get the module containing operation functions."""
        return sys.modules[__name__]

    def _get_interface(self):
        """Get the compute interface."""
        return _interface


class Shader:
    """GPU compute shader wrapper."""

    def __init__(self, spirv: Union[List[int], "_cut_core_type.ShaderEnum"],
                 dtype: Optional[np.dtype] = None):
        """
        Create a shader module.

        Args:
            spirv: SPIR-V bytecode as list of uint32 or a ShaderEnum for built-in shaders
            dtype: NumPy dtype to select the appropriate shader variant (default: float32)
        """
        _ensure_initialized()
        if isinstance(spirv, _cut_core.ShaderEnum):
            # Map numpy dtype to ScalarDataType
            scalar_dtype = _cut_core.ScalarDataType.Float
            if dtype is not None:
                dtype = np.dtype(dtype)
                if dtype == np.int32:
                    scalar_dtype = _cut_core.ScalarDataType.Int
                elif dtype == np.uint32:
                    scalar_dtype = _cut_core.ScalarDataType.UInt
                elif dtype == np.float16:
                    scalar_dtype = _cut_core.ScalarDataType.Half

            # Check cache first with (op_enum, scalar_dtype) key
            cache_key = (spirv, scalar_dtype)
            if cache_key in _shader_cache:
                self._handle = _shader_cache[cache_key]
                return
            # Compile and cache
            spirv_code = _cut_core.get_shader(spirv, scalar_dtype)
            self._handle = _interface.create_shader_module(spirv_code)
            _shader_cache[cache_key] = self._handle
        else:
            self._handle = _interface.create_shader_module(spirv)

    @property
    def handle(self) -> "_cut_core_type.ComputeHandle":
        """Get the underlying compute handle."""
        return self._handle


class Dispatch(BaseDispatch):
    """Compute dispatch configuration."""

    def __init__(self, shader: Shader, thread_groups: tuple = (1, 1, 1)):
        """
        Create a compute dispatch.

        Args:
            shader: Shader to execute
            thread_groups: Number of thread groups (x, y, z)
        """
        dispatch_obj = _cut_core.ComputeDispatch(shader.handle)
        super().__init__(dispatch_obj, _cut_core.ThreadSize)
        self.set_workgroup_size(thread_groups)

    def bind(self, resource: Union[Buffer, np.ndarray, int, float], binding: int) -> "Dispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer, numpy array, int (as uint32), or float (as float32)
            binding: Binding index

        Returns:
            self for chaining
        """
        return super().bind(resource, binding, Buffer)


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

if _VULKAN_AVAILABLE:
    register_operations(
        module_dict=globals(),
        enum_module=_cut_core,
        buffer_class=Buffer,
        shader_or_kernel_class=Shader,
        dispatch_class=Dispatch,
        run_func=run,
        ensure_init=_ensure_initialized,
        backend_name="on GPU",
    )


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
    # Re-exports from core
    "OperatorEnum",
    "ShaderEnum",  # Backward compatibility alias
] + ALL_OPERATION_NAMES

# Re-export OperatorEnum and ShaderEnum (alias) if available
if _VULKAN_AVAILABLE:
    OperatorEnum = _cut_core.OperatorEnum
    ShaderEnum = _cut_core.ShaderEnum  # Backward compatibility alias
else:
    OperatorEnum = None
    ShaderEnum = None

# Expose availability flag for benchmarks
def is_vulkan_available() -> bool:
    """Check if Vulkan backend is available."""
    return _VULKAN_AVAILABLE
