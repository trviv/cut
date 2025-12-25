#!/usr/bin/env python3
"""
Benchmark comparing CUT GPU operations against NumPy CPU operations.
Tests all operations in ShaderEnum with 100K elements.
"""

import numpy as np
import time
from typing import Callable, Tuple, Dict, List
import cut

# Number of elements to test
N = 100_000

# Number of iterations for timing
NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 3


def benchmark(func: Callable, *args, name: str = "") -> Tuple[float, np.ndarray]:
    """Run a function multiple times and return average time and result."""
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


def verify_results(cut_result: np.ndarray, np_result: np.ndarray, op_name: str, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify that CUT and NumPy results match within tolerance."""
    try:
        np.testing.assert_allclose(cut_result, np_result, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"  WARNING: {op_name} results differ: {e}")
        return False


def run_benchmarks():
    """Run all benchmarks and display results."""
    print("=" * 80)
    print(f"CUT vs NumPy Benchmark - {N:,} elements ({NUM_ITERATIONS} iterations)")
    print("=" * 80)

    # Precompile and cache all shaders before benchmarking
    print("\nPrecompiling shaders...")
    cut.precompile_shaders()
    print("Shader precompilation complete.\n")

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
        cut_time, cut_result = benchmark(cut_func, op_a, op_b, name=name)
        np_time, np_result = benchmark(np_func, op_a, op_b, name=name)

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
        cut_time, cut_result = benchmark(cut_func, op_a, op_b, name=name)
        np_time, np_result = benchmark(np_func, op_a, op_b, name=name)

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
        cut_time, cut_result = benchmark(cut_func, op_a, op_b, name=name)
        np_time, np_result = benchmark(np_func, op_a, op_b, name=name)

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
        cut_time, cut_result = benchmark(cut_func, op_a, name=name)
        np_time, np_result = benchmark(np_func, op_a, name=name)

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

    return results


if __name__ == "__main__":
    results = run_benchmarks()
