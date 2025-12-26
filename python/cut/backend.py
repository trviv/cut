"""
CUT Unified Backend Interface

Provides a single interface for all CUT backends (Vulkan, CPU, CPU+SIMD).
Use `init()` to select and initialize a backend, then use the module-level
functions for operations.

Example:
    import cut.backend as cut

    # Initialize with Vulkan backend
    cut.init("vulkan")

    # Or use CPU with SIMD
    cut.init("cpu", simd=True)

    # Use operations
    a = cut.Buffer(np.array([1, 2, 3], dtype=np.float32))
    b = cut.Buffer(np.array([4, 5, 6], dtype=np.float32))
    c = cut.add(a, b)
    result = c.numpy()
"""

from __future__ import annotations

import atexit
import weakref
import numpy as np
from typing import Optional, Union, List, Callable, Any
from enum import Enum, auto

from ._ops import ALL_OPERATION_NAMES, register_operations


class Backend(Enum):
    """Available compute backends."""
    VULKAN = auto()
    CPU = auto()
    CPU_SIMD = auto()


# Current backend state
_current_backend: Optional[Backend] = None
_backend_module = None
_interface = None

# Track all live buffers using weak references
_live_buffers: weakref.WeakSet = weakref.WeakSet()


def _cleanup():
    """Clean up resources."""
    global _interface, _backend_module, _current_backend
    for buf in list(_live_buffers):
        if hasattr(buf, '_handle'):
            buf._handle = None
    _interface = None
    _backend_module = None
    _current_backend = None


atexit.register(_cleanup)


def available_backends() -> List[str]:
    """
    Get list of available backends.

    Returns:
        List of backend names that can be used with init()
    """
    available = []

    # Check Vulkan
    try:
        from . import _cut_core
        # Import the main module to check vulkan availability
        from . import is_vulkan_available
        if is_vulkan_available():
            available.append("vulkan")
    except ImportError:
        pass

    # Check CPU
    try:
        from . import _cut_cpu
        available.append("cpu")
        available.append("cpu_simd")
    except ImportError:
        pass

    return available


def current_backend() -> Optional[str]:
    """
    Get the currently active backend.

    Returns:
        Name of current backend or None if not initialized
    """
    if _current_backend is None:
        return None
    return {
        Backend.VULKAN: "vulkan",
        Backend.CPU: "cpu",
        Backend.CPU_SIMD: "cpu_simd",
    }.get(_current_backend)


def init(backend: str = "auto", *, num_threads: int = 0, simd: bool = False) -> str:
    """
    Initialize a compute backend.

    Args:
        backend: Backend to use. One of:
            - "auto": Automatically select best available (vulkan > cpu_simd > cpu)
            - "vulkan": Use Vulkan GPU backend
            - "cpu": Use CPU backend (scalar mode)
            - "cpu_simd": Use CPU backend with SIMD (SSE/AVX)
        num_threads: Number of worker threads for CPU backend (0 = auto)
        simd: Deprecated, use backend="cpu_simd" instead

    Returns:
        Name of the initialized backend

    Raises:
        RuntimeError: If requested backend is not available
    """
    global _current_backend, _backend_module, _interface

    # Handle simd parameter for backward compatibility
    if simd and backend == "cpu":
        backend = "cpu_simd"

    available = available_backends()

    if not available:
        raise RuntimeError("No CUT backends available. Please build and install the library.")

    # Auto-select best backend
    if backend == "auto":
        if "vulkan" in available:
            backend = "vulkan"
        elif "cpu_simd" in available:
            backend = "cpu_simd"
        elif "cpu" in available:
            backend = "cpu"
        else:
            raise RuntimeError("No backends available")

    # Validate backend choice
    if backend not in available and backend != "cpu_simd":
        raise RuntimeError(f"Backend '{backend}' is not available. Available: {available}")

    # cpu_simd is just cpu with SIMD mode enabled
    if backend == "cpu_simd" and "cpu" not in available:
        raise RuntimeError(f"CPU backend not available for cpu_simd mode")

    # Clean up previous backend
    if _current_backend is not None:
        _cleanup()

    # Initialize the selected backend
    if backend == "vulkan":
        _init_vulkan()
        _current_backend = Backend.VULKAN
    elif backend == "cpu":
        _init_cpu(num_threads=num_threads, simd=False)
        _current_backend = Backend.CPU
    elif backend == "cpu_simd":
        _init_cpu(num_threads=num_threads, simd=True)
        _current_backend = Backend.CPU_SIMD
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Register operations into this module's namespace
    _register_current_backend_ops()

    return current_backend()


def _init_vulkan():
    """Initialize Vulkan backend."""
    global _backend_module, _interface
    from . import _cut_core
    # Import the cut module (parent package)
    import cut as vulkan_module

    if not vulkan_module.is_vulkan_available():
        raise RuntimeError("Vulkan backend not available")

    _backend_module = vulkan_module
    vulkan_module._ensure_initialized()
    _interface = vulkan_module.get_interface()


def _init_cpu(num_threads: int = 0, simd: bool = False):
    """Initialize CPU backend."""
    global _backend_module, _interface
    from . import cpu as cpu_module
    from . import _cut_cpu

    _backend_module = cpu_module

    # Set SIMD mode
    if simd:
        cpu_module.set_simd_mode(_cut_cpu.SIMDMode.Auto)
    else:
        cpu_module.set_simd_mode(_cut_cpu.SIMDMode.Scalar)

    cpu_module._ensure_initialized(num_threads=num_threads)
    _interface = cpu_module.get_interface()


def _ensure_initialized():
    """Ensure a backend is initialized."""
    if _current_backend is None:
        # Auto-initialize with best available backend
        init("auto")


def _register_current_backend_ops():
    """Register operations from current backend into this module."""
    global _backend_module

    if _backend_module is None:
        return

    # Get the module's globals
    module_globals = globals()

    # For each operation, create a wrapper that unwraps our Buffer to backend Buffer
    for op_name in ALL_OPERATION_NAMES:
        if hasattr(_backend_module, op_name):
            backend_func = getattr(_backend_module, op_name)

            def make_wrapper(func):
                def wrapper(*args, **kwargs):
                    # Unwrap any Buffer arguments to their backend buffers
                    # Also convert numpy arrays to backend buffers
                    unwrapped_args = []
                    for arg in args:
                        if isinstance(arg, Buffer):
                            unwrapped_args.append(arg._backend_buffer)
                        elif isinstance(arg, np.ndarray):
                            # Convert numpy array to backend buffer
                            unwrapped_args.append(_backend_module.Buffer(arg))
                        else:
                            unwrapped_args.append(arg)

                    # Call the backend function
                    result = func(*unwrapped_args, **kwargs)

                    # If the result is a numpy array, return it directly
                    if isinstance(result, np.ndarray):
                        return result

                    # Wrap the result if it's a backend buffer
                    if hasattr(result, 'numpy') and hasattr(result, '_handle'):
                        # Create a wrapper that points to the result
                        wrapped = object.__new__(Buffer)
                        wrapped._backend_buffer = result
                        _live_buffers.add(wrapped)
                        return wrapped
                    return result
                return wrapper

            module_globals[op_name] = make_wrapper(backend_func)


def get_interface():
    """Get the underlying compute interface."""
    _ensure_initialized()
    return _interface


def precompile_shaders():
    """
    Precompile all shaders (Vulkan backend only).
    For CPU backend, this is a no-op.
    """
    _ensure_initialized()
    if _current_backend == Backend.VULKAN and hasattr(_backend_module, 'precompile_shaders'):
        _backend_module.precompile_shaders()


def num_threads() -> int:
    """Get number of worker threads (CPU backend only)."""
    _ensure_initialized()
    if _current_backend in (Backend.CPU, Backend.CPU_SIMD):
        return _backend_module.num_threads()
    return 0


def simd_mode() -> Optional[str]:
    """Get current SIMD mode (CPU backend only)."""
    _ensure_initialized()
    if _current_backend in (Backend.CPU, Backend.CPU_SIMD):
        mode = _backend_module.simd_mode()
        return str(mode).split('.')[-1]  # Convert enum to string
    return None


class Buffer:
    """
    Unified buffer class that wraps the backend-specific buffer.

    This class provides a consistent interface regardless of which
    backend is being used.
    """

    def __init__(self, data: Optional[np.ndarray] = None, size: Optional[int] = None,
                 is_uniform: bool = False, dtype: Optional[np.dtype] = None,
                 shape: Optional[tuple] = None):
        """
        Create a buffer.

        Args:
            data: NumPy array to initialize buffer with (optional)
            size: Buffer size in bytes (required if data is None)
            is_uniform: If True, create a uniform buffer
            dtype: Data type for the buffer (used when creating from size)
            shape: Shape for the buffer (used when creating from size)
        """
        _ensure_initialized()

        # Create the underlying backend buffer
        self._backend_buffer = _backend_module.Buffer(
            data=data, size=size, is_uniform=is_uniform,
            dtype=dtype, shape=shape
        )
        _live_buffers.add(self)

    @property
    def handle(self):
        """Get the underlying compute handle."""
        return self._backend_buffer.handle

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._backend_buffer.size

    @property
    def _dtype(self):
        """Get the buffer's dtype."""
        return self._backend_buffer._dtype

    @property
    def _shape(self):
        """Get the buffer's shape."""
        return self._backend_buffer._shape

    @property
    def _handle(self):
        """Get the underlying handle (for compatibility)."""
        return self._backend_buffer._handle

    @_handle.setter
    def _handle(self, value):
        """Set the underlying handle (for cleanup)."""
        self._backend_buffer._handle = value

    def copy_from(self, data: np.ndarray):
        """Copy data from numpy array to buffer."""
        self._backend_buffer.copy_from(data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy data from buffer to numpy array."""
        return self._backend_buffer.copy_to(out)

    def numpy(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        return self._backend_buffer.numpy()


class Shader:
    """
    Unified shader/kernel class.

    For Vulkan, this wraps a compiled shader module.
    For CPU, this wraps a kernel function.
    """

    def __init__(self, op_enum, dtype: Optional[np.dtype] = None):
        """
        Create a shader/kernel.

        Args:
            op_enum: The operator enum (ShaderEnum for Vulkan, OperatorEnum for CPU)
            dtype: NumPy dtype to select the appropriate variant
        """
        _ensure_initialized()

        if _current_backend == Backend.VULKAN:
            self._backend_shader = _backend_module.Shader(op_enum, dtype=dtype)
        else:
            self._backend_shader = _backend_module.Kernel(op_enum)

    @property
    def handle(self):
        """Get the underlying compute handle."""
        return self._backend_shader.handle


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
        self._backend_dispatch = _backend_module.Dispatch(shader._backend_shader, thread_groups)

    def bind(self, resource: Union[Buffer, np.ndarray, int, float], binding: int) -> "Dispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer, numpy array, int, or float
            binding: Binding index

        Returns:
            self for chaining
        """
        if isinstance(resource, Buffer):
            self._backend_dispatch.bind(resource._backend_buffer, binding)
        else:
            self._backend_dispatch.bind(resource, binding)
        return self

    @property
    def inner(self):
        """Get the underlying dispatch object."""
        return self._backend_dispatch.inner


def run(*dispatches: Dispatch):
    """
    Execute one or more compute dispatches.

    Args:
        *dispatches: Dispatch objects to execute
    """
    _ensure_initialized()
    backend_dispatches = [d._backend_dispatch for d in dispatches]
    _backend_module.run(*backend_dispatches)


# Export public API
__all__ = [
    # Backend management
    "Backend",
    "init",
    "available_backends",
    "current_backend",
    # Classes
    "Buffer",
    "Shader",
    "Dispatch",
    # Core functions
    "run",
    "get_interface",
    "precompile_shaders",
    "num_threads",
    "simd_mode",
] + ALL_OPERATION_NAMES
