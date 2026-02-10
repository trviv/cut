#!/usr/bin/env python3
"""
Benchmark comparing CUT operations against NumPy CPU operations.
Tests all operations with 100K elements using the unified backend interface.

Usage:
    python benchmark.py                    # Auto-select best backend
    python benchmark.py --backend vulkan   # Use Vulkan GPU
    python benchmark.py --backend cpu      # Use CPU (scalar)
    python benchmark.py --backend cpu_simd # Use CPU with SIMD
"""

import argparse
import time
import numpy as np
from typing import List, Dict

import cut.compute as cut
from common import (
    init_backend,
    TestData,
    BenchmarkConfig,
    verify_results,
    OutputFormatter,
)
from common.operations import (
    BINARY_ARITHMETIC_OPS,
    COMPARISON_OPS,
    MINMAX_OPS,
    UNARY_OPS,
)

# Default configuration
DEFAULT_CONFIG = BenchmarkConfig(
    num_elements=100_000,
    num_iterations=10,
    warmup_iterations=3,
    seed=42,
)


def benchmark_numpy(func, *args, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Run a NumPy function multiple times and return average time and result."""
    # Warmup
    for _ in range(config.warmup_iterations):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    return avg_time, result


def benchmark_cut(func, *args, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Run a CUT function multiple times and return average time and result."""
    # Convert numpy arrays to Tensors
    tensor_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            tensor_args.append(cut.Tensor(arg))
        else:
            tensor_args.append(arg)

    # Warmup
    for _ in range(config.warmup_iterations):
        result = func(*tensor_args)

    # Timed runs
    times = []
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        result = func(*tensor_args)
        result_list = result.tolist() # do this so commands are flushed
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms

    # Extract numpy array from result
    flat = result.copy_to()
    result = np.array(list(flat), dtype=np.float32)

    return avg_time, result


def run_benchmarks(backend_name: str, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Run all benchmarks and display results."""
    # Initialize backend using shared helper
    initialized_backend, _ = init_backend(cut, backend_name, force=True)

    OutputFormatter.header(
        f"CUT vs NumPy Benchmark - {config.num_elements:,} elements ({config.num_iterations} iterations)"
    )
    print(f"Backend: {initialized_backend.upper()}")

    if cut.current_backend() == cut.Backend.Vulkan:
        print()  # blank line
    else:
        print(f"\nUsing {cut.num_threads()} threads, SIMD mode: {cut.simd_mode()}\n")

    # Generate test data
    data = TestData.generate(config.num_elements, config.seed)

    results: List[Dict] = []

    def run_op_category(category_name: str, ops: list, is_unary: bool = False):
        """Run benchmarks for a category of operations."""
        print(f"\n--- {category_name} ---")

        for op_info in ops:
            if is_unary:
                name, np_func, arr_name = op_info
                op_a = getattr(data, arr_name)
                op_b = None
            else:
                name, np_func, arr1, arr2 = op_info
                op_a = getattr(data, arr1)
                op_b = getattr(data, arr2)

            cut_func = getattr(cut, name, None)
            if cut_func is None:
                continue

            if is_unary:
                cut_time, cut_result = benchmark_cut(cut_func, op_a, config=config)
                np_time, np_result = benchmark_numpy(np_func, op_a, config=config)
            else:
                cut_time, cut_result = benchmark_cut(cut_func, op_a, op_b, config=config)
                np_time, np_result = benchmark_numpy(np_func, op_a, op_b, config=config)

            np_result = np_result.astype(np.float32)
            valid = verify_results(cut_result, np_result, name)
            speedup = np_time / cut_time if cut_time > 0 else float('inf')

            results.append({
                "name": name,
                "cut_ms": cut_time,
                "np_ms": np_time,
                "speedup": speedup,
                "valid": valid
            })

            OutputFormatter.print_result_line(name, cut_time, np_time, speedup, valid)

    # Run Binary Arithmetic Operations
    run_op_category("Binary Arithmetic Operations", BINARY_ARITHMETIC_OPS)

    # Run Binary Comparison Operations
    comparison_ops = [
        (name, lambda x, y, n=name: getattr(np, n)(x, y).astype(np.float32), arr1, arr2)
        for name, arr1, arr2 in COMPARISON_OPS
    ]
    run_op_category("Binary Comparison Operations", comparison_ops)

    # Run Binary Min/Max Operations
    run_op_category("Binary Min/Max Operations", MINMAX_OPS)

    # Run Unary Operations
    run_op_category("Unary Operations", UNARY_OPS, is_unary=True)

    # Summary
    OutputFormatter.header("SUMMARY")
    print(f"Backend: {initialized_backend}")
    OutputFormatter.print_summary_stats(results)

    # Cleanup all buffers before returning
    cut.shutdown()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CUT Benchmark - Compare CUT operations against NumPy"
    )
    parser.add_argument(
        '--backend', '-b',
        choices=['auto', 'vulkan', 'cpu_simd'],
        default='auto',
        help='Backend to use (default: auto)'
    )
    parser.add_argument(
        '--list-backends',
        action='store_true',
        help='List available backends and exit'
    )

    args = parser.parse_args()

    if args.list_backends:
        print("Available backends:", cut.available_backends())
        return

    run_benchmarks(args.backend)


if __name__ == "__main__":
    main()
