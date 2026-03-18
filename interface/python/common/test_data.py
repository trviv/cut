"""
Test data generation for CUT benchmarks and tests.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestData:
    """Container for test arrays used in benchmarks and tests."""
    a: np.ndarray
    b: np.ndarray
    a_pos: np.ndarray
    b_pos: np.ndarray
    a_unit: np.ndarray
    b_small: np.ndarray
    a_div10: np.ndarray
    a_tan_safe: np.ndarray
    # 2D matrix data for matmul/softmax/layernorm benchmarks
    mat_a: Optional[np.ndarray] = None
    mat_b: Optional[np.ndarray] = None
    mat_2d: Optional[np.ndarray] = None

    @classmethod
    def generate(cls, num_elements: int, seed: int = 42) -> 'TestData':
        """
        Generate random test data with various array types.

        Args:
            num_elements: Number of elements in each array
            seed: Random seed for reproducibility

        Returns:
            TestData instance with generated arrays
        """
        np.random.seed(seed)
        N = num_elements
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        # Matrix dimensions for matmul benchmarks
        # Use sqrt-ish dimensions to keep total elements manageable
        mat_dim = int(np.sqrt(N))
        mat_dim = max(mat_dim, 64)  # At least 64x64

        return cls(
            a=a,
            b=b,
            a_pos=np.abs(a) + 0.1,
            b_pos=np.abs(b) + 0.1,
            a_unit=np.clip(a, -0.99, 0.99).astype(np.float32),
            b_small=(np.random.randn(N) * 2).astype(np.float32),
            a_div10=(a / 10).astype(np.float32),
            a_tan_safe=np.clip(a, -1.0, 1.0).astype(np.float32),
            mat_a=np.random.randn(mat_dim, mat_dim).astype(np.float32),
            mat_b=np.random.randn(mat_dim, mat_dim).astype(np.float32),
            mat_2d=np.random.randn(mat_dim, mat_dim).astype(np.float32),
        )

    def get_array_mapping(self) -> dict:
        """
        Get a mapping of array ids to named arrays for buffer conversion.

        Returns:
            Dictionary mapping id(array) to array name
        """
        mapping = {
            id(self.a): 'a',
            id(self.b): 'b',
            id(self.a_pos): 'a_pos',
            id(self.b_pos): 'b_pos',
            id(self.a_unit): 'a_unit',
            id(self.b_small): 'b_small',
            id(self.a_div10): 'a_div10',
            id(self.a_tan_safe): 'a_tan_safe',
        }
        if self.mat_a is not None:
            mapping[id(self.mat_a)] = 'mat_a'
        if self.mat_b is not None:
            mapping[id(self.mat_b)] = 'mat_b'
        if self.mat_2d is not None:
            mapping[id(self.mat_2d)] = 'mat_2d'
        return mapping
