"""
Benchmark result data structures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import OrderedDict


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
    """Complete result for one operation across all backends.

    Uses a dynamic dict so any number of backends can be added.
    The 'numpy' key is always the reference baseline.
    """
    name: str
    category: str
    backends: Dict[str, BackendResult] = field(default_factory=OrderedDict)

    # Keep legacy accessors for backward compatibility
    @property
    def numpy(self) -> BackendResult:
        return self.backends.get('numpy', BackendResult())

    @property
    def vulkan(self) -> BackendResult:
        return self.backends.get('vulkan', BackendResult())

    @property
    def cupy(self) -> BackendResult:
        return self.backends.get('cupy', BackendResult())

    @property
    def jax(self) -> BackendResult:
        return self.backends.get('jax', BackendResult())

    @property
    def pytorch(self) -> BackendResult:
        return self.backends.get('pytorch', BackendResult())

    def get(self, backend_name: str) -> BackendResult:
        return self.backends.get(backend_name, BackendResult())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON export."""
        def backend_to_dict(br: BackendResult) -> dict:
            return {
                'time_ms': br.time_ms if not np.isnan(br.time_ms) else None,
                'std_ms': br.std_ms,
                'valid': br.valid,
                'speedup': br.speedup if not np.isnan(br.speedup) else None,
            }

        d = {
            'name': self.name,
            'category': self.category,
        }
        for name, br in self.backends.items():
            d[name] = backend_to_dict(br)
        return d
