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

import gc
import numpy as np
import time
import argparse
from typing import Callable, Tuple, Dict, List

import cut.compute as cut

# Number of elements to test
N = 100_000

# Number of iterations for timing
NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 3


def benchmark_numpy(func: Callable, *args, name: str = "") -> Tuple[float, np.ndarray]:
    """Run a NumPy function multiple times and return average time and result."""
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    return avg_time, result


def benchmark_cut(func: Callable, *args, name: str = "") -> Tuple[float, np.ndarray]:
    """Run a CUT function multiple times and return average time and result."""
    # Convert numpy arrays to Buffers
    buffer_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            buffer_args.append(cut.Buffer(arg))
        else:
            buffer_args.append(arg)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = func(*buffer_args)

    # Timed runs
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result = func(*buffer_args)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times) * 1000  # Convert to ms

    # Extract numpy array from result
    if hasattr(result, 'numpy'):
        result = result.numpy()

    return avg_time, result


def verify_results(cut_result: np.ndarray, np_result: np.ndarray, op_name: str, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify that CUT and NumPy results match within tolerance."""
    try:
        np.testing.assert_allclose(cut_result, np_result, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"  WARNING: {op_name} results differ: {e}")
        return False


def _cleanup_buffers():
    """Force cleanup of all buffers before shutdown."""
    gc.collect()
    gc.collect()


def run_benchmarks(backend_name: str):
    """Run all benchmarks and display results."""
    # Map string backend names to Backend enum
    backend_map = {
        'auto': cut.Backend.CPU,  # Default to CPU for auto
        'vulkan': cut.Backend.Vulkan,
        'cpu': cut.Backend.CPU,
        'cpu_simd': cut.Backend.CPU,
    }

    simd_map = {
        'cpu': cut.SIMDMode.Scalar,
        'cpu_simd': cut.SIMDMode.Auto,
    }

    backend_enum = backend_map.get(backend_name, cut.Backend.CPU)
    simd_mode = simd_map.get(backend_name, cut.SIMDMode.Auto)

    # Initialize the backend
    if backend_enum == cut.Backend.Vulkan:
        cut.init(cut.Backend.Vulkan, force=True)
        initialized_backend = "vulkan"
    else:
        cut.init(cut.Backend.CPU, simd_mode=simd_mode, force=True)
        initialized_backend = backend_name if backend_name in ('cpu', 'cpu_simd') else 'cpu'

    print("=" * 80)
    print(f"CUT vs NumPy Benchmark - {N:,} elements ({NUM_ITERATIONS} iterations)")
    print(f"Backend: {initialized_backend.upper()}")
    print("=" * 80)

    # Precompile shaders if using Vulkan
    if cut.current_backend() == cut.Backend.Vulkan:
        print("\nPrecompiling shaders...")
        cut.precompile_shaders()
        print("Shader precompilation complete.\n")
    else:
        print(f"\nUsing {cut.num_threads()} threads, SIMD mode: {cut.simd_mode()}\n")

    # Generate test data
    np.random.seed(42)
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    # For operations that need positive inputs
    a_pos = np.abs(a) + 0.1
    b_pos = np.abs(b) + 0.1

    # For operations that need values in [-1, 1]
    a_unit = np.clip(a, -0.99, 0.99).astype(np.float32)

    # For small exponents (to avoid overflow in pow)
    b_small = (np.random.randn(N) * 2).astype(np.float32)

    results: List[Dict] = []

    # ==========================================================================
    # Binary Arithmetic Operations
    # ==========================================================================
    print("\n--- Binary Arithmetic Operations ---")

    binary_ops = [
        ("add", cut.add, np.add, a, b),
        ("subtract", cut.subtract, np.subtract, a, b),
        ("multiply", cut.multiply, np.multiply, a, b),
        ("divide", cut.divide, np.divide, a, b_pos),  # Avoid div by zero
        ("mod", cut.mod, np.mod, a_pos, b_pos),  # Positive values for mod
        ("power", cut.power, np.power, a_pos, b_small),  # Positive base, small exp
        ("floor_divide", cut.floor_divide, np.floor_divide, a, b_pos),
    ]

    for name, cut_func, np_func, op_a, op_b in binary_ops:
        cut_time, cut_result = benchmark_cut(cut_func, op_a, op_b)
        np_time, np_result = benchmark_numpy(np_func, op_a, op_b)

        valid = verify_results(cut_result, np_result, name)
        speedup = np_time / cut_time if cut_time > 0 else float('inf')

        results.append({
            "name": name,
            "cut_ms": cut_time,
            "np_ms": np_time,
            "speedup": speedup,
            "valid": valid
        })

        status = "OK" if valid else "MISMATCH"
        faster = "CUT" if speedup > 1 else "NumPy"
        print(f"  {name:15s}: CUT={cut_time:8.3f}ms, NumPy={np_time:8.3f}ms, {speedup:6.2f}x ({faster} faster) [{status}]")

    # ==========================================================================
    # Binary Comparison Operations
    # ==========================================================================
    print("\n--- Binary Comparison Operations ---")

    comparison_ops = [
        ("equal", cut.equal, lambda x, y: np.equal(x, y).astype(np.float32), a, b),
        ("not_equal", cut.not_equal, lambda x, y: np.not_equal(x, y).astype(np.float32), a, b),
        ("less", cut.less, lambda x, y: np.less(x, y).astype(np.float32), a, b),
        ("less_equal", cut.less_equal, lambda x, y: np.less_equal(x, y).astype(np.float32), a, b),
        ("greater", cut.greater, lambda x, y: np.greater(x, y).astype(np.float32), a, b),
        ("greater_equal", cut.greater_equal, lambda x, y: np.greater_equal(x, y).astype(np.float32), a, b),
    ]

    for name, cut_func, np_func, op_a, op_b in comparison_ops:
        cut_time, cut_result = benchmark_cut(cut_func, op_a, op_b)
        np_time, np_result = benchmark_numpy(np_func, op_a, op_b)

        valid = verify_results(cut_result, np_result, name)
        speedup = np_time / cut_time if cut_time > 0 else float('inf')

        results.append({
            "name": name,
            "cut_ms": cut_time,
            "np_ms": np_time,
            "speedup": speedup,
            "valid": valid
        })

        status = "OK" if valid else "MISMATCH"
        faster = "CUT" if speedup > 1 else "NumPy"
        print(f"  {name:15s}: CUT={cut_time:8.3f}ms, NumPy={np_time:8.3f}ms, {speedup:6.2f}x ({faster} faster) [{status}]")

    # ==========================================================================
    # Binary Min/Max Operations
    # ==========================================================================
    print("\n--- Binary Min/Max Operations ---")

    minmax_ops = [
        ("minimum", cut.minimum, np.minimum, a, b),
        ("maximum", cut.maximum, np.maximum, a, b),
    ]

    for name, cut_func, np_func, op_a, op_b in minmax_ops:
        cut_time, cut_result = benchmark_cut(cut_func, op_a, op_b)
        np_time, np_result = benchmark_numpy(np_func, op_a, op_b)

        valid = verify_results(cut_result, np_result, name)
        speedup = np_time / cut_time if cut_time > 0 else float('inf')

        results.append({
            "name": name,
            "cut_ms": cut_time,
            "np_ms": np_time,
            "speedup": speedup,
            "valid": valid
        })

        status = "OK" if valid else "MISMATCH"
        faster = "CUT" if speedup > 1 else "NumPy"
        print(f"  {name:15s}: CUT={cut_time:8.3f}ms, NumPy={np_time:8.3f}ms, {speedup:6.2f}x ({faster} faster) [{status}]")

    # ==========================================================================
    # Unary Operations
    # ==========================================================================
    print("\n--- Unary Operations ---")

    unary_ops = [
        ("negative", cut.negative, np.negative, a),
        ("abs", cut.abs, np.abs, a),
        ("sqrt", cut.sqrt, np.sqrt, a_pos),
        ("exp", cut.exp, np.exp, a / 10),  # Scale down to avoid overflow
        ("log", cut.log, np.log, a_pos),
        ("log2", cut.log2, np.log2, a_pos),
        ("log10", cut.log10, np.log10, a_pos),
        ("sin", cut.sin, np.sin, a),
        ("cos", cut.cos, np.cos, a),
        ("tan", cut.tan, np.tan, a),
        ("arcsin", cut.arcsin, np.arcsin, a_unit),
        ("arccos", cut.arccos, np.arccos, a_unit),
        ("arctan", cut.arctan, np.arctan, a),
        ("sinh", cut.sinh, np.sinh, a / 10),  # Scale down to avoid overflow
        ("cosh", cut.cosh, np.cosh, a / 10),
        ("tanh", cut.tanh, np.tanh, a),
        ("floor", cut.floor, np.floor, a),
        ("ceil", cut.ceil, np.ceil, a),
        ("round", cut.round, np.round, a),
        ("sign", cut.sign, np.sign, a),
        ("reciprocal", cut.reciprocal, np.reciprocal, a_pos),
        ("square", cut.square, np.square, a),
    ]

    for name, cut_func, np_func, op_a in unary_ops:
        cut_time, cut_result = benchmark_cut(cut_func, op_a)
        np_time, np_result = benchmark_numpy(np_func, op_a)

        valid = verify_results(cut_result, np_result.astype(np.float32), name)
        speedup = np_time / cut_time if cut_time > 0 else float('inf')

        results.append({
            "name": name,
            "cut_ms": cut_time,
            "np_ms": np_time,
            "speedup": speedup,
            "valid": valid
        })

        status = "OK" if valid else "MISMATCH"
        faster = "CUT" if speedup > 1 else "NumPy"
        print(f"  {name:15s}: CUT={cut_time:8.3f}ms, NumPy={np_time:8.3f}ms, {speedup:6.2f}x ({faster} faster) [{status}]")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_ops = len(results)
    valid_ops = sum(1 for r in results if r["valid"])
    cut_faster = sum(1 for r in results if r["speedup"] > 1)
    np_faster = total_ops - cut_faster

    avg_speedup = np.mean([r["speedup"] for r in results])
    median_speedup = np.median([r["speedup"] for r in results])

    print(f"Backend: {initialized_backend}")
    print(f"Total operations tested: {total_ops}")
    print(f"Results matching: {valid_ops}/{total_ops}")
    print(f"CUT faster: {cut_faster}, NumPy faster: {np_faster}")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Median speedup: {median_speedup:.2f}x")

    # Find best and worst performers
    best = max(results, key=lambda r: r["speedup"])
    worst = min(results, key=lambda r: r["speedup"])

    print(f"\nBest CUT performance: {best['name']} ({best['speedup']:.2f}x faster)")
    print(f"Worst CUT performance: {worst['name']} ({worst['speedup']:.2f}x)")

    # Cleanup all buffers before returning
    _cleanup_buffers()
    cut.shutdown()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CUT Benchmark - Compare CUT operations against NumPy"
    )
    parser.add_argument(
        '--backend', '-b',
        choices=['auto', 'vulkan', 'cpu', 'cpu_simd'],
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
