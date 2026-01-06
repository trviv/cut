"""
CUT (Compute Unified Toolkit) - GPU/CPU Compute Library

A Python library for compute operations using Vulkan or CPU backends.

Example:
    import cut

    # Initialize with backend
    cut.init(cut.Backend.Vulkan)  # GPU
    cut.init(cut.Backend.CPU, simd_mode=cut.SIMDMode.Auto)  # CPU with SIMD

    # Use operations
    a = cut.Tensor(np.array([1, 2, 3], dtype=np.float32))
    b = cut.Tensor(np.array([4, 5, 6], dtype=np.float32))
    c = cut.add(a, b)
    result = c.numpy()
"""

# Re-export everything from compute module
from .compute import *
from .compute import __all__

__version__ = "0.1.0"
