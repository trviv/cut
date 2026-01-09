"""
Benchmark result data structures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BackendResult:
    """Result from a single backend benchmark."""
    time_ms: float = float('nan')
    std_ms: float = 0.0
    valid: bool = False
    speedup: float = float('nan')

    def is_available(self) -> bool:
        """Check if this result has valid timing data."""
        return not np.isnan(self.time_ms)


@dataclass
class BenchmarkResult:
    """Complete result for one operation across all backends."""
    name: str
    category: str
    numpy: BackendResult
    vulkan: BackendResult = field(default_factory=BackendResult)
    cpu: BackendResult = field(default_factory=BackendResult)
    cpu_simd: BackendResult = field(default_factory=BackendResult)
    cupy: BackendResult = field(default_factory=BackendResult)
    jax: BackendResult = field(default_factory=BackendResult)
    pytorch: BackendResult = field(default_factory=BackendResult)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON export."""
        def backend_to_dict(br: BackendResult) -> dict:
            return {
                'time_ms': br.time_ms if not np.isnan(br.time_ms) else None,
                'std_ms': br.std_ms,
                'valid': br.valid,
                'speedup': br.speedup if not np.isnan(br.speedup) else None,
            }

        return {
            'name': self.name,
            'category': self.category,
            'numpy': backend_to_dict(self.numpy),
            'vulkan': backend_to_dict(self.vulkan),
            'cpu': backend_to_dict(self.cpu),
            'cpu_simd': backend_to_dict(self.cpu_simd),
            'cupy': backend_to_dict(self.cupy),
            'jax': backend_to_dict(self.jax),
            'pytorch': backend_to_dict(self.pytorch),
        }
