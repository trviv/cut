"""
Benchmark configuration settings.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_elements: int = 1_000_000
    num_iterations: int = 10
    warmup_iterations: int = 3
    seed: int = 42

    def __post_init__(self):
        if self.num_elements <= 0:
            raise ValueError("num_elements must be positive")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
