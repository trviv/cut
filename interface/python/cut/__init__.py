"""
CUT (Compute Unified Toolkit) - Vulkan GPU compute toolkit

A Python library for GPU compute operations using a Vulkan backend.
Uses native Python types (no numpy dependency).

Example:
    import cut

    # Initialize with backend
    cut.init(cut.Backend.Vulkan)  # GPU

    # Use operations
    a = cut.Tensor([1.0, 2.0, 3.0])  # float32 by default
    b = cut.Tensor([4.0, 5.0, 6.0])
    c = cut.add(a, b)
    result = c.tolist()
"""

# Re-export everything from compute module
from .compute import *
from .compute import __all__

__version__ = "0.1.0"
