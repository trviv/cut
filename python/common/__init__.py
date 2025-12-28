"""
Common utilities for CUT benchmarks and tests.

This module provides shared functionality used across benchmarks and tests.
"""

from .test_data import TestData
from .config import BenchmarkConfig
from .results import BackendResult, BenchmarkResult
from .timing import TimingContext, benchmark_function
from .verification import verify_results
from .formatting import Colors, OutputFormatter
from .operations import get_operations, BINARY_ARITHMETIC_OPS, COMPARISON_OPS, UNARY_OPS

# Re-export from cut_utils for convenience
from .cut_utils import init_backend, backend_context, get_backend_info

__all__ = [
    # Test data
    'TestData',
    # Configuration
    'BenchmarkConfig',
    # Results
    'BackendResult',
    'BenchmarkResult',
    # Timing
    'TimingContext',
    'benchmark_function',
    # Verification
    'verify_results',
    # Formatting
    'Colors',
    'OutputFormatter',
    # Operations
    'get_operations',
    'BINARY_ARITHMETIC_OPS',
    'COMPARISON_OPS',
    'UNARY_OPS',
    # Backend management (from cut_utils)
    'init_backend',
    'backend_context',
    'get_backend_info',
]
