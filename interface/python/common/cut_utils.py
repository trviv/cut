"""
CUT backend initialization and management utilities.

This module provides utilities for initializing and managing CUT backends.
Used by both tests and benchmarks.
"""

import gc
from typing import Tuple, Any
from contextlib import contextmanager


def cleanup():
    """Force cleanup of all buffers."""
    gc.collect()
    gc.collect()


def init_backend(cc, backend_name: str, force: bool = False) -> Tuple[str, Any]:
    """
    Initialize a CUT backend by name.

    Args:
        cc: The cut.compute module
        backend_name: One of 'auto', 'vulkan', 'cpu', 'cpu_simd'
        force: Whether to force re-initialization

    Returns:
        Tuple of (initialized_backend_name, backend_enum)
    """
    backend_map = {
        'auto': cc.Backend.CPU,
        'vulkan': cc.Backend.Vulkan,
        'cpu': cc.Backend.CPU,
        'cpu_simd': cc.Backend.CPU,
    }

    simd_map = {
        'cpu': cc.SIMDMode.Scalar,
        'cpu_simd': cc.SIMDMode.Auto,
    }

    backend_enum = backend_map.get(backend_name, cc.Backend.CPU)
    simd_mode = simd_map.get(backend_name, cc.SIMDMode.Auto)

    if backend_enum == cc.Backend.Vulkan:
        cc.init(cc.Backend.Vulkan, force=force)
        return "vulkan", backend_enum
    else:
        cc.init(cc.Backend.CPU, simd_mode=simd_mode, force=force)
        return backend_name if backend_name in ('cpu', 'cpu_simd') else 'cpu', backend_enum


@contextmanager
def backend_context(cc, backend_name: str = 'auto'):
    """
    Context manager for backend initialization with automatic cleanup.

    Usage:
        with backend_context(cc, 'vulkan') as backend_info:
            name, backend_enum = backend_info
            # ... use backend ...
    """
    name, backend_enum = init_backend(cc, backend_name, force=True)
    try:
        yield name, backend_enum
    finally:
        cc.shutdown()


def get_backend_info(cc) -> dict:
    """
    Get information about the current backend.

    Args:
        cc: The cut.compute module

    Returns:
        Dictionary with backend information
    """
    info = {
        'current_backend': cc.current_backend(),
        'vulkan_available': cc.is_vulkan_available(),
        'cpu_available': cc.is_cpu_available(),
    }

    if cc.current_backend() == cc.Backend.CPU:
        info['num_threads'] = cc.num_threads()
        info['simd_mode'] = cc.simd_mode()
    else:
        info['num_threads'] = 0
        info['simd_mode'] = None

    return info
