"""
Tests for chained operations in CUT compute interface.

These tests verify that operations can be correctly chained together,
where outputs of one or more operations feed into subsequent operations.
Tests cover various dependency patterns including:
- Linear chains (A -> B -> C)
- Diamond patterns (A -> B, A -> C, B + C -> D)
- Fan-out patterns (A -> B, A -> C, A -> D)
- Fan-in patterns (A, B, C -> D)
- Complex DAGs with multiple dependencies
"""

import numpy as np
import pytest
import cut.compute as cc

import gc


def _cleanup():
    """Force cleanup of all buffers."""
    gc.collect()
    gc.collect()


def get_available_backends():
    """Get list of available backends as pytest parameters."""
    backends = []
    if cc.is_cpu_available():
        backends.append(pytest.param(cc.Backend.CPU, id="cpu"))
    if cc.is_vulkan_available():
        backends.append(pytest.param(cc.Backend.Vulkan, id="vulkan"))
    return backends


@pytest.fixture(params=get_available_backends())
def backend(request):
    """Fixture that yields each available backend."""
    backend_type = request.param
    if backend_type == cc.Backend.Vulkan:
        cc.init(backend_type)
    else:
        cc.init(backend_type, simd_mode=cc.SIMDMode.Auto)
    yield backend_type
    _cleanup()
    cc.shutdown()


# =============================================================================
# Linear Chain Tests (A -> B -> C -> ...)
# =============================================================================

class TestLinearChains:
    """Tests for linear operation chains where each op depends on the previous."""

    def test_two_op_chain_add_multiply(self, backend):
        """Test: result = (a + b) * c"""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        c = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))

        # Chain: add -> multiply
        result = (a + b) * c

        expected = (np.array([1.0, 2.0, 3.0, 4.0]) + 0.5) * 2.0
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_three_op_chain_arithmetic(self, backend):
        """Test: result = ((a + b) * c) - d"""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        c = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
        d = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))

        result = ((a + b) * c) - d

        expected = ((np.array([1.0, 2.0, 3.0, 4.0]) + 1.0) * 2.0) - 0.5
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_four_op_chain_with_unary(self, backend):
        """Test: result = sqrt(abs(a - b) + c)"""
        a = cc.Buffer(np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32))
        b = cc.Buffer(np.array([5.0, 2.0, 10.0, 12.0], dtype=np.float32))
        c = cc.Buffer(np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32))

        result = cc.sqrt(cc.abs(a - b) + c)

        expected = np.sqrt(np.abs(np.array([1.0, 4.0, 9.0, 16.0]) -
                                   np.array([5.0, 2.0, 10.0, 12.0])) +
                          np.array([3.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_long_chain_five_ops(self, backend):
        """Test: result = floor((((a + b) * c) / d) - e)"""
        a = cc.Buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
        b = cc.Buffer(np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32))
        c = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
        d = cc.Buffer(np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))
        e = cc.Buffer(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

        result = cc.floor((((a + b) * c) / d) - e)

        a_np = np.array([10.0, 20.0, 30.0, 40.0])
        b_np = np.array([5.0, 5.0, 5.0, 5.0])
        c_np = np.array([2.0, 2.0, 2.0, 2.0])
        d_np = np.array([3.0, 3.0, 3.0, 3.0])
        e_np = np.array([0.1, 0.2, 0.3, 0.4])
        expected = np.floor((((a_np + b_np) * c_np) / d_np) - e_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_operator_overload_chain(self, backend):
        """Test chaining with Python operator overloading."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        c = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))

        # Using operator overloading
        result = (a + b) * c - a

        a_np = np.array([1.0, 2.0, 3.0, 4.0])
        expected = (a_np + 0.5) * 2.0 - a_np
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


# =============================================================================
# Diamond Pattern Tests (A -> B, A -> C, B + C -> D)
# =============================================================================

class TestDiamondPatterns:
    """Tests for diamond dependency patterns where one input feeds multiple paths."""

    def test_simple_diamond(self, backend):
        """
        Test diamond pattern:
            a
           / \
          b   c   (b = a+1, c = a*2)
           \ /
            d     (d = b + c)
        """
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        b = a + 1.0       # a + 1
        c = a * 2.0       # a * 2
        d = b + c         # b + c

        a_np = np.array([1.0, 2.0, 3.0, 4.0])
        expected = (a_np + 1) + (a_np * 2)  # 3a + 1
        np.testing.assert_allclose(d.numpy(), expected, rtol=1e-5)

    def test_diamond_with_unary_branches(self, backend):
        """
        Diamond with different unary ops on each branch:
            a
           / \
        sqrt  square
           \ /
            *
        """
        a = cc.Buffer(np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float32))

        result = cc.sqrt(a) * cc.square(a)

        a_np = np.array([4.0, 9.0, 16.0, 25.0])
        expected = np.sqrt(a_np) * (a_np ** 2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_nested_diamond(self, backend):
        """
        Nested diamond pattern:
              a
             / \
            b   c
           /|   |\
          d e   f g
           \|   |/
            h   i
             \ /
              j
        Where operations chain through multiple levels.
        """
        a = cc.Buffer(np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))

        # First split
        b = a + 1.0      # a + 1
        c = a * 2.0      # a * 2

        # Second split from b
        d = cc.square(b)
        e = cc.sqrt(b)

        # Second split from c
        f = -c
        g = cc.abs(c)

        # Merge branches
        h = d + e
        i = f + g

        # Final merge
        result = h * i

        a_np = np.array([2.0, 3.0, 4.0, 5.0])
        b_np = a_np + 1
        c_np = a_np * 2
        d_np = b_np ** 2
        e_np = np.sqrt(b_np)
        f_np = -c_np
        g_np = np.abs(c_np)
        h_np = d_np + e_np
        i_np = f_np + g_np
        expected = h_np * i_np
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)


# =============================================================================
# Fan-Out Tests (A -> B, A -> C, A -> D)
# =============================================================================

class TestFanOutPatterns:
    """Tests for fan-out patterns where one input feeds many outputs."""

    def test_triple_fan_out(self, backend):
        """Test one input feeding three independent operations."""
        a = cc.Buffer(np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float32))

        b = cc.sqrt(a)
        c = cc.square(a)
        d = cc.negative(a)

        a_np = np.array([4.0, 9.0, 16.0, 25.0])
        np.testing.assert_allclose(b.numpy(), np.sqrt(a_np), rtol=1e-5)
        np.testing.assert_allclose(c.numpy(), a_np ** 2, rtol=1e-5)
        np.testing.assert_allclose(d.numpy(), -a_np, rtol=1e-5)

    def test_fan_out_with_merge(self, backend):
        """Fan-out that eventually merges: result = sqrt(a) + square(a) + abs(a)"""
        a = cc.Buffer(np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float32))

        result = cc.sqrt(a) + cc.square(a) + cc.abs(a)

        a_np = np.array([4.0, 9.0, 16.0, 25.0])
        expected = np.sqrt(a_np) + (a_np ** 2) + np.abs(a_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_fan_out_reuse_intermediate(self, backend):
        """Test reusing an intermediate result in multiple places."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))

        # intermediate is used twice
        intermediate = a + b
        result = (intermediate * a) + (intermediate * b)

        a_np = np.array([1.0, 2.0, 3.0, 4.0])
        b_np = np.array([2.0, 2.0, 2.0, 2.0])
        inter_np = a_np + b_np
        expected = (inter_np * a_np) + (inter_np * b_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


# =============================================================================
# Fan-In Tests (A, B, C -> D)
# =============================================================================

class TestFanInPatterns:
    """Tests for fan-in patterns where multiple inputs combine."""

    def test_three_input_fan_in(self, backend):
        """Test combining three independent inputs."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        c = cc.Buffer(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))

        # Combine a, b, c into one result
        result = (a + b) * c

        expected = (np.array([1.0, 2.0, 3.0, 4.0]) + 0.5) * 2.0
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_weighted_sum_pattern(self, backend):
        """Test weighted sum: w1*a + w2*b + w3*c"""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32))
        c = cc.Buffer(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

        result = a * 0.5 + b * 0.3 + c * 0.2

        a_np = np.array([1.0, 2.0, 3.0, 4.0])
        b_np = np.array([4.0, 3.0, 2.0, 1.0])
        c_np = np.array([1.0, 1.0, 1.0, 1.0])
        expected = 0.5 * a_np + 0.3 * b_np + 0.2 * c_np
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


# =============================================================================
# Mixed Scalar and Vector Operations in Chains
# =============================================================================

class TestMixedScalarVectorChains:
    """Tests for chains mixing scalar and vector operations."""

    def test_chain_with_scalar_ops(self, backend):
        """Test chain mixing vector-vector and vector-scalar ops."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))

        # (a + b) * 2 + 1
        result = (a + b) * 2.0 + 1.0

        expected = (np.array([1.0, 2.0, 3.0, 4.0]) + 0.5) * 2.0 + 1.0
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_normalization_chain(self, backend):
        """Test normalize-like chain: (x - min) / (max - min)"""
        x = cc.Buffer(np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32))

        result = (x - 2.0) / (8.0 - 2.0)

        x_np = np.array([2.0, 4.0, 6.0, 8.0])
        expected = (x_np - 2.0) / (8.0 - 2.0)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_power_chain_with_scalars(self, backend):
        """Test chain: (a^2 + b^2)^0.5 using power ops"""
        a = cc.Buffer(np.array([3.0, 4.0, 5.0, 12.0], dtype=np.float32))
        b = cc.Buffer(np.array([4.0, 3.0, 12.0, 5.0], dtype=np.float32))

        result = cc.sqrt(cc.square(a) + cc.square(b))

        a_np = np.array([3.0, 4.0, 5.0, 12.0])
        b_np = np.array([4.0, 3.0, 12.0, 5.0])
        expected = np.sqrt(a_np**2 + b_np**2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)


# =============================================================================
# Transcendental Function Chains
# =============================================================================

class TestTranscendentalChains:
    """Tests for chains involving transcendental functions (sin, cos, exp, log)."""

    def test_trig_identity_chain(self, backend):
        """Test sin^2 + cos^2 = 1"""
        x = cc.Buffer(np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32))

        result = cc.square(cc.sin(x)) + cc.square(cc.cos(x))

        expected = np.ones(4, dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_exp_log_inverse(self, backend):
        """Test exp(log(x)) = x"""
        x = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        result = cc.exp(cc.log(x))

        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_softmax_like_chain(self, backend):
        """Test softmax-like chain: exp(x) / sum(exp(x)) approximation."""
        x = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        result = cc.exp(x) / 84.79

        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.exp(x_np) / 84.79
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)


# =============================================================================
# Complex DAG Patterns
# =============================================================================

class TestComplexDAGs:
    """Tests for complex DAG structures with multiple dependencies."""

    def test_quadratic_formula_like(self, backend):
        """
        Compute quadratic-like formula: (-b + sqrt(b^2 - 4ac)) / 2a
        Using a=1, b=-5, c=6 -> roots at 2 and 3
        """
        # Broadcast to vectors for testing
        a = cc.Buffer(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        b = cc.Buffer(np.array([-5.0, -6.0, -7.0, -8.0], dtype=np.float32))
        c = cc.Buffer(np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32))

        # (-b + sqrt(b^2 - 4ac)) / (2a)
        discriminant = cc.square(b) - a * c * 4.0
        result = (-b + cc.sqrt(discriminant)) / (a * 2.0)

        a_np = np.array([1.0, 1.0, 1.0, 1.0])
        b_np = np.array([-5.0, -6.0, -7.0, -8.0])
        c_np = np.array([6.0, 8.0, 10.0, 12.0])
        expected = (-b_np + np.sqrt(b_np**2 - 4*a_np*c_np)) / (2*a_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_distance_formula(self, backend):
        """Test Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)"""
        x1 = cc.Buffer(np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))
        y1 = cc.Buffer(np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))
        x2 = cc.Buffer(np.array([3.0, 4.0, 4.0, 6.0], dtype=np.float32))
        y2 = cc.Buffer(np.array([4.0, 3.0, 5.0, 6.0], dtype=np.float32))

        dx = x2 - x1
        dy = y2 - y1
        result = cc.sqrt(cc.square(dx) + cc.square(dy))

        x1_np = np.array([0.0, 0.0, 1.0, 2.0])
        y1_np = np.array([0.0, 0.0, 1.0, 2.0])
        x2_np = np.array([3.0, 4.0, 4.0, 6.0])
        y2_np = np.array([4.0, 3.0, 5.0, 6.0])
        expected = np.sqrt((x2_np - x1_np)**2 + (y2_np - y1_np)**2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_sigmoid_approximation(self, backend):
        """Test sigmoid-like computation: 1 / (1 + exp(-x))"""
        x = cc.Buffer(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        ones = cc.Buffer(np.ones(8, dtype=np.float32))

        result = ones / (ones + cc.exp(-x))

        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 1.0 / (1.0 + np.exp(-x_np))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_relu_like_chain(self, backend):
        """Test ReLU-like: max(0, x)"""
        x = cc.Buffer(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5], dtype=np.float32))
        zeros = cc.Buffer(np.zeros(8, dtype=np.float32))

        result = cc.maximum(x, zeros)

        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5])
        expected = np.maximum(0, x_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


# =============================================================================
# Large Chain Tests
# =============================================================================

class TestLargeChains:
    """Tests with larger data sizes and longer chains."""

    def test_large_linear_chain(self, backend):
        """Test linear chain with 100K elements."""
        n = 100_000
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)
        c_np = np.random.randn(n).astype(np.float32)

        a = cc.Buffer(a_np)
        b = cc.Buffer(b_np)
        c = cc.Buffer(c_np)

        result = cc.sqrt(cc.abs((a + b) * c))

        expected = np.sqrt(np.abs((a_np + b_np) * c_np))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-6)

    def test_large_diamond_chain(self, backend):
        """Test diamond pattern with 100K elements."""
        n = 100_000
        a_np = np.abs(np.random.randn(n).astype(np.float32)) + 0.1  # Positive for sqrt

        a = cc.Buffer(a_np)

        result = cc.sqrt(a) + cc.log(a)

        expected = np.sqrt(a_np) + np.log(a_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3, atol=1e-5)

    def test_ten_op_chain(self, backend):
        """Test a 10-operation chain."""
        n = 10_000
        a_np = np.abs(np.random.randn(n).astype(np.float32)) + 1.0

        a = cc.Buffer(a_np)

        t1 = a + 1.0
        t2 = t1 * 2.0
        t3 = cc.sqrt(t2)
        t4 = cc.square(t3)
        t5 = t4 - 1.0
        t6 = cc.abs(t5)
        t7 = cc.log(t6)
        t8 = cc.exp(t7)
        t9 = t8 + 1.0
        result = cc.floor(t9)

        t1_np = a_np + 1
        t2_np = t1_np * 2
        t3_np = np.sqrt(t2_np)
        t4_np = t3_np ** 2
        t5_np = t4_np - 1
        t6_np = np.abs(t5_np)
        t7_np = np.log(t6_np)
        t8_np = np.exp(t7_np)
        t9_np = t8_np + 1
        expected = np.floor(t9_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-3, atol=1e-4)


# =============================================================================
# Integer Operation Chains
# =============================================================================

class TestIntegerChains:
    """Tests for chains with integer data types."""

    def test_int32_linear_chain(self, backend):
        """Test linear chain with int32 data."""
        # Note: int32 operations may have limited support on some backends
        if backend == cc.Backend.CPU:
            pytest.skip("int32 multiply not fully supported on CPU backend")

        a = cc.Buffer(np.array([1, 2, 3, 4], dtype=np.int32))
        b = cc.Buffer(np.array([5, 6, 7, 8], dtype=np.int32))
        c = cc.Buffer(np.array([2, 2, 2, 2], dtype=np.int32))

        result = ((a + b) * c) - a

        a_np = np.array([1, 2, 3, 4], dtype=np.int32)
        b_np = np.array([5, 6, 7, 8], dtype=np.int32)
        c_np = np.array([2, 2, 2, 2], dtype=np.int32)
        expected = ((a_np + b_np) * c_np) - a_np
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_int32_modulo_chain(self, backend):
        """Test chain with modulo operation."""
        # Note: int32 mod may have limited support on some backends
        if backend == cc.Backend.Vulkan:
            pytest.skip("int32 mod shader compilation issue on Vulkan")

        a = cc.Buffer(np.array([10, 20, 30, 40], dtype=np.int32))
        b = cc.Buffer(np.array([3, 7, 8, 9], dtype=np.int32))
        c = cc.Buffer(np.array([2, 2, 2, 2], dtype=np.int32))

        result = cc.mod(a + b, c)

        a_np = np.array([10, 20, 30, 40], dtype=np.int32)
        b_np = np.array([3, 7, 8, 9], dtype=np.int32)
        expected = (a_np + b_np) % 2
        np.testing.assert_array_equal(result.numpy(), expected)


# =============================================================================
# Output Buffer Reuse in Chains
# =============================================================================

class TestOutputBufferReuse:
    """Tests for chains that reuse output buffers."""

    def test_explicit_output_chain(self, backend):
        """Test chain with explicit output buffer allocation."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))

        # Pre-allocate output buffers (size is in bytes: 4 elements * 4 bytes = 16)
        out1 = cc.Buffer(size=16, dtype=np.float32)
        out2 = cc.Buffer(size=16, dtype=np.float32)

        cc.add(a, b, out=out1)
        cc.multiply(out1, b, out=out2)

        expected = (np.array([1.0, 2.0, 3.0, 4.0]) + 0.5) * 0.5
        np.testing.assert_allclose(out2.numpy(), expected, rtol=1e-5)

    def test_buffer_identity_in_chain(self, backend):
        """Verify that output buffer is the same object when passed explicitly."""
        a = cc.Buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = cc.Buffer(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        # size is in bytes: 4 elements * 4 bytes = 16
        out = cc.Buffer(size=16, dtype=np.float32)

        result = cc.add(a, b, out=out)
        assert result is out

        result2 = cc.multiply(result, b, out=out)
        assert result2 is out
