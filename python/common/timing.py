"""
Timing utilities for benchmarks.
"""

import time
import numpy as np
from typing import Callable, Tuple, List, Any
from contextlib import contextmanager


@contextmanager
def TimingContext():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000  # Return ms


def benchmark_function(
    func: Callable,
    args: tuple,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    sync_func: Callable = None,
) -> Tuple[float, float, Any]:
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: The function to benchmark
        args: Arguments to pass to the function
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations
        sync_func: Optional synchronization function (for GPU backends)

    Returns:
        Tuple of (mean_time_ms, std_time_ms, last_result)
    """
    # Warmup
    for _ in range(warmup_iterations):
        result = func(*args)
        if sync_func:
            sync_func()

    # Timed runs
    times: List[float] = []
    for _ in range(num_iterations):
        if sync_func:
            sync_func()
        start = time.perf_counter()
        result = func(*args)
        if sync_func:
            sync_func()
        end = time.perf_counter()
        times.append(end - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000

    return mean_ms, std_ms, result


def run_numpy_benchmark(
    np_func: Callable,
    np_args: tuple,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> Tuple[float, float, np.ndarray]:
    """
    Run a NumPy benchmark.

    Args:
        np_func: NumPy function to benchmark
        np_args: Arguments to pass to the function
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Tuple of (mean_time_ms, std_time_ms, result_array)
    """
    mean_ms, std_ms, result = benchmark_function(
        np_func, np_args, num_iterations, warmup_iterations
    )
    return mean_ms, std_ms, result.astype(np.float32)
