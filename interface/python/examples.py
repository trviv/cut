#!/usr/bin/env python3
"""
CUT Library Examples

This script demonstrates various use cases of the CUT (Compute Unified Toolkit) library
for GPU and CPU accelerated element-wise operations.

Run with: python examples.py
"""

import numpy as np
import cut.compute as cc


def example_basic_operations():
    """Example 1: Basic element-wise operations."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Element-wise Operations")
    print("=" * 60)

    # Create input arrays
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

    # Create CUT tensors
    buf_a = cc.Tensor(a)
    buf_b = cc.Tensor(b)

    # Perform operations
    result_add = cc.add(buf_a, buf_b)
    result_mul = cc.multiply(buf_a, buf_b)
    result_div = cc.divide(buf_b, buf_a)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {result_add.numpy()}")
    print(f"a * b = {result_mul.numpy()}")
    print(f"b / a = {result_div.numpy()}")


def example_operator_overloading():
    """Example 2: Using Python operators with CUT tensors."""
    print("\n" + "=" * 60)
    print("Example 2: Operator Overloading")
    print("=" * 60)

    a = cc.Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = cc.Tensor(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))

    # Use Python operators directly
    c = a + b
    d = a * b
    e = b - a
    f = -a  # Unary negation

    print(f"a = {a.numpy()}")
    print(f"b = {b.numpy()}")
    print(f"a + b = {c.numpy()}")
    print(f"a * b = {d.numpy()}")
    print(f"b - a = {e.numpy()}")
    print(f"-a = {f.numpy()}")

    # Scalar operations
    g = a + 10.0
    h = a * 2.0
    print(f"a + 10 = {g.numpy()}")
    print(f"a * 2 = {h.numpy()}")


def example_math_functions():
    """Example 3: Mathematical functions."""
    print("\n" + "=" * 60)
    print("Example 3: Mathematical Functions")
    print("=" * 60)

    # Create test data
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    buf_x = cc.Tensor(x)

    # Trigonometric functions
    sin_x = cc.sin(buf_x)
    cos_x = cc.cos(buf_x)

    print(f"x = {x}")
    print(f"sin(x) = {sin_x.numpy()}")
    print(f"cos(x) = {cos_x.numpy()}")

    # Exponential and logarithmic
    positive = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    buf_pos = cc.Tensor(positive)

    exp_x = cc.exp(cc.Tensor(np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)))
    log_x = cc.log(buf_pos)
    sqrt_x = cc.sqrt(buf_pos)

    print(f"\npositive = {positive}")
    print(f"exp([0, 0.5, 1, 1.5, 2]) = {exp_x.numpy()}")
    print(f"log(positive) = {log_x.numpy()}")
    print(f"sqrt(positive) = {sqrt_x.numpy()}")


def example_comparison_operations():
    """Example 4: Comparison operations."""
    print("\n" + "=" * 60)
    print("Example 4: Comparison Operations")
    print("=" * 60)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 3.0, 2.0, 7.0], dtype=np.float32)

    buf_a = cc.Tensor(a)
    buf_b = cc.Tensor(b)

    # Comparisons return 1.0 for true, 0.0 for false
    eq = cc.equal(buf_a, buf_b)
    lt = cc.less(buf_a, buf_b)
    gt = cc.greater(buf_a, buf_b)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a == b: {eq.numpy()}")
    print(f"a < b:  {lt.numpy()}")
    print(f"a > b:  {gt.numpy()}")

    # Scalar comparison
    threshold = 3.0
    above = cc.greater_scalar(buf_a, threshold)
    print(f"\na > {threshold}: {above.numpy()}")


def example_large_array_processing():
    """Example 5: Processing large arrays."""
    print("\n" + "=" * 60)
    print("Example 5: Large Array Processing")
    print("=" * 60)

    import time

    # Create large arrays
    n = 1_000_000
    np.random.seed(42)
    data = np.random.randn(n).astype(np.float32)

    # Time NumPy
    start = time.perf_counter()
    np_result = np.tanh(data)
    np_time = (time.perf_counter() - start) * 1000

    # Time CUT
    buf = cc.Tensor(data)
    start = time.perf_counter()
    cut_result = cc.tanh(buf)
    cut_time = (time.perf_counter() - start) * 1000

    # Verify results match
    np.testing.assert_allclose(cut_result.numpy(), np_result, rtol=1e-5)

    print(f"Array size: {n:,} elements")
    print(f"NumPy tanh: {np_time:.3f} ms")
    print(f"CUT tanh:   {cut_time:.3f} ms")
    print(f"Speedup:    {np_time / cut_time:.2f}x")
    print("Results verified to match!")


def example_chained_operations():
    """Example 6: Chaining multiple operations."""
    print("\n" + "=" * 60)
    print("Example 6: Chained Operations")
    print("=" * 60)

    # Compute: result = sqrt(a^2 + b^2)  (Euclidean norm components)
    a = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    b = np.array([4.0, 3.0, 12.0, 8.0], dtype=np.float32)

    buf_a = cc.Tensor(a)
    buf_b = cc.Tensor(b)

    # Chain operations
    a_squared = cc.square(buf_a)
    b_squared = cc.square(buf_b)
    sum_squares = cc.add(a_squared, b_squared)
    result = cc.sqrt(sum_squares)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"sqrt(a^2 + b^2) = {result.numpy()}")
    print(f"Expected: {np.sqrt(a**2 + b**2)}")


def example_min_max_clamp():
    """Example 7: Min, max, and clamping operations."""
    print("\n" + "=" * 60)
    print("Example 7: Min/Max and Clamping")
    print("=" * 60)

    data = np.array([-2.0, -0.5, 0.3, 0.8, 1.5, 3.0], dtype=np.float32)
    buf = cc.Tensor(data)

    # Clamp to [0, 1] range using min and max with scalars
    clamped_low = cc.maximum_scalar(buf, 0.0)  # max(data, 0) - clamp lower
    buf_clamped_low = cc.Tensor(clamped_low.numpy())
    clamped = cc.minimum_scalar(buf_clamped_low, 1.0)  # min(result, 1) - clamp upper

    print(f"Original: {data}")
    print(f"Clamped to [0, 1]: {clamped.numpy()}")

    # Element-wise min/max between two arrays
    a = np.array([1.0, 5.0, 3.0, 7.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    buf_a = cc.Tensor(a)
    buf_b = cc.Tensor(b)

    min_ab = cc.minimum(buf_a, buf_b)
    max_ab = cc.maximum(buf_a, buf_b)

    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"min(a, b) = {min_ab.numpy()}")
    print(f"max(a, b) = {max_ab.numpy()}")


def example_integer_operations():
    """Example 8: Integer operations."""
    print("\n" + "=" * 60)
    print("Example 8: Integer Operations")
    print("=" * 60)

    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    b = np.array([3, 7, 4, 8, 9], dtype=np.int32)

    buf_a = cc.Tensor(a)
    buf_b = cc.Tensor(b)

    result_add = cc.add(buf_a, buf_b)
    result_mul = cc.multiply(buf_a, buf_b)
    result_min = cc.minimum(buf_a, buf_b)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {result_add.numpy()}")
    print(f"a * b = {result_mul.numpy()}")
    print(f"min(a, b) = {result_min.numpy()}")


def example_backend_switching():
    """Example 9: Switching between backends."""
    print("\n" + "=" * 60)
    print("Example 9: Backend Information")
    print("=" * 60)

    print(f"Available backends: {cc.available_backends()}")
    print(f"Current backend: {cc.current_backend()}")

    if cc.current_backend() == cc.Backend.Vulkan:
        print("Running on Vulkan GPU")
    else:
        print("Running on unknown backend")


def main():
    """Run all examples."""
    print("=" * 60)
    print("CUT Library Examples")
    print("=" * 60)

    # Initialize backend
    print(f"\nInitializing CUT with Vulkan backend...")
    cc.init(cc.Backend.Vulkan)
    print(f"Backend: {cc.current_backend()}")

    # Run examples
    example_basic_operations()
    example_operator_overloading()
    example_math_functions()
    example_comparison_operations()
    example_large_array_processing()
    example_chained_operations()
    example_min_max_clamp()
    example_integer_operations()
    example_backend_switching()

    # Cleanup
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
