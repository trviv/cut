"""
Tests for the unified CUT compute interface.

These tests verify that all operations work correctly across different backends
(Vulkan and CPU) using the unified cut.compute module.
"""

import numpy as np
import pytest
import cut.compute as cc
from common import cleanup


# =============================================================================
# Test Utilities
# =============================================================================

def get_available_backends():
    """Get list of available backends as pytest parameters."""
    backends = []
    if cc.is_cpu_available():
        backends.append(pytest.param(cc.Backend.CPU, id="cpu"))
    if cc.is_vulkan_available():
        backends.append(pytest.param(cc.Backend.Vulkan, id="vulkan"))
    return backends


# =============================================================================
# Fixtures for Backend Parametrization
# =============================================================================

@pytest.fixture(params=get_available_backends())
def backend(request):
    """Fixture that yields each available backend."""
    backend_type = request.param
    if backend_type == cc.Backend.Vulkan:
        cc.init(backend_type)
    else:
        cc.init(backend_type, simd_mode=cc.SIMDMode.Auto)
    yield backend_type
    cleanup()
    cc.shutdown()


@pytest.fixture
def cpu_backend():
    """Fixture for CPU backend only."""
    if not cc.is_cpu_available():
        pytest.skip("CPU backend not available")
    cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.Auto)
    yield cc.Backend.CPU
    cleanup()
    cc.shutdown()


@pytest.fixture
def vulkan_backend():
    """Fixture for Vulkan backend only."""
    if not cc.is_vulkan_available():
        pytest.skip("Vulkan backend not available")
    cc.init(cc.Backend.Vulkan)
    yield cc.Backend.Vulkan
    cleanup()
    cc.shutdown()


# =============================================================================
# Basic Module Tests
# =============================================================================

class TestModuleBasics:
    """Test basic module functionality."""

    def test_import(self):
        """Test that the module can be imported."""
        assert cc.Backend is not None
        assert cc.SIMDMode is not None

    def test_backend_enum(self):
        """Test Backend enum values."""
        assert hasattr(cc.Backend, "Vulkan")
        assert hasattr(cc.Backend, "CPU")

    def test_simd_enum(self):
        """Test SIMDMode enum values."""
        assert hasattr(cc.SIMDMode, "Scalar")
        assert hasattr(cc.SIMDMode, "SSE")
        assert hasattr(cc.SIMDMode, "AVX")
        assert hasattr(cc.SIMDMode, "Auto")

    def test_operator_enum(self):
        """Test OperatorEnum is accessible."""
        assert hasattr(cc, "OperatorEnum")
        assert hasattr(cc.OperatorEnum, "BinaryVecVecAdd")
        assert hasattr(cc.OperatorEnum, "UnaryNeg")

    def test_shader_enum_alias(self):
        """Test ShaderEnum is an alias for OperatorEnum."""
        assert cc.ShaderEnum is cc.OperatorEnum

    def test_available_backends(self):
        """Test available_backends returns valid list."""
        available = cc.available_backends()
        assert isinstance(available, list)
        assert len(available) > 0

    def test_cpu_available(self):
        """Test CPU is always available."""
        assert cc.is_cpu_available() is True


# =============================================================================
# Backend Initialization Tests
# =============================================================================

class TestBackendInit:
    """Test backend initialization."""

    def test_init_cpu(self):
        """Test CPU backend initialization."""
        if not cc.is_cpu_available():
            pytest.skip("CPU backend not available")

        backend = cc.init(cc.Backend.CPU)
        assert backend == cc.Backend.CPU
        assert cc.current_backend() == cc.Backend.CPU

    def test_init_cpu_with_simd(self):
        """Test CPU backend with SIMD mode."""
        if not cc.is_cpu_available():
            pytest.skip("CPU backend not available")

        cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.Auto)
        assert cc.current_backend() == cc.Backend.CPU
        mode = cc.simd_mode()
        assert mode in (cc.SIMDMode.Scalar, cc.SIMDMode.SSE,
                       cc.SIMDMode.AVX, cc.SIMDMode.Auto)

    def test_init_vulkan(self):
        """Test Vulkan backend initialization."""
        if not cc.is_vulkan_available():
            pytest.skip("Vulkan backend not available")

        backend = cc.init(cc.Backend.Vulkan)
        assert backend == cc.Backend.Vulkan
        assert cc.current_backend() == cc.Backend.Vulkan
        cc.shutdown()

    def test_num_threads_cpu(self, cpu_backend):
        """Test num_threads returns positive value for CPU."""
        threads = cc.num_threads()
        assert threads > 0

    def test_num_threads_vulkan(self, vulkan_backend):
        """Test num_threads returns 0 for Vulkan."""
        threads = cc.num_threads()
        assert threads == 0


# =============================================================================
# Buffer Tests
# =============================================================================

class TestBuffer:
    """Test Buffer class functionality."""

    def test_create_from_array(self, backend):
        """Test creating a buffer from numpy array."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cc.Buffer(data)
        assert buf.size == data.nbytes
        assert buf.handle.valid()

    def test_create_empty(self, backend):
        """Test creating an empty buffer by size."""
        buf = cc.Buffer(size=64)
        assert buf.size == 64
        assert buf.handle.valid()

    def test_create_requires_data_or_size(self, backend):
        """Test that Buffer requires either data or size."""
        with pytest.raises(ValueError, match="Either data or size must be provided"):
            cc.Buffer()

    def test_roundtrip_float32(self, backend):
        """Test data roundtrip with float32."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cc.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_int32(self, backend):
        """Test data roundtrip with int32."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        buf = cc.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_2d_array(self, backend):
        """Test data roundtrip with 2D array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        buf = cc.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_copy_from(self, backend):
        """Test copying new data to existing buffer."""
        initial = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cc.Buffer(initial)

        new_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buf.copy_from(new_data)

        result = buf.numpy()
        np.testing.assert_array_equal(result, new_data)


# =============================================================================
# Shader and Dispatch Tests
# =============================================================================

class TestShaderDispatch:
    """Test Shader and Dispatch classes."""

    def test_create_shader_from_enum(self, backend):
        """Test creating shader from OperatorEnum."""
        shader = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        assert shader.handle.valid()

    def test_create_dispatch(self, backend):
        """Test creating a dispatch."""
        shader = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        dispatch = cc.Dispatch(shader, thread_groups=(1, 1, 1))
        assert dispatch.inner is not None

    def test_dispatch_chaining(self, backend):
        """Test dispatch bind chaining."""
        shader = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf_a = cc.Buffer(data)
        buf_b = cc.Buffer(data)
        buf_out = cc.Buffer(size=data.nbytes)

        dispatch = (cc.Dispatch(shader, thread_groups=(4, 1, 1))
                    .bind(buf_a, 0)
                    .bind(buf_b, 1)
                    .bind(buf_out, 2)
                    .bind(4, 3))

        assert dispatch is not None

    def test_run_dispatch(self, backend):
        """Test running a dispatch."""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buf_a = cc.Buffer(a)
        buf_b = cc.Buffer(b)
        buf_out = cc.Buffer(size=a.nbytes)

        shader = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        dispatch = (cc.Dispatch(shader, thread_groups=(4, 1, 1))
                    .bind(buf_a, 0)
                    .bind(buf_b, 1)
                    .bind(buf_out, 2)
                    .bind(4, 3))

        cc.run(dispatch)

        result = buf_out.numpy()
        expected = a + b
        np.testing.assert_allclose(result, expected)

    def test_shader_caching(self, backend):
        """Test that shaders are cached and reused."""
        cc.clear_shader_cache()

        shader1 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        assert not shader1.cached
        assert shader1.handle.valid()

        shader2 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd)
        assert shader2.cached
        assert shader2.handle.valid()

        stats = cc.get_shader_cache_stats()
        assert stats['size'] >= 1

    def test_shader_cache_different_dtypes(self, backend):
        """Test that different dtypes get different cached shaders."""
        cc.clear_shader_cache()

        shader_f32 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd, dtype=np.float32)
        shader_i32 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd, dtype=np.int32)

        assert not shader_f32.cached
        assert not shader_i32.cached

        stats = cc.get_shader_cache_stats()
        assert stats['size'] == 2

        shader_f32_2 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd, dtype=np.float32)
        shader_i32_2 = cc.Shader(cc.OperatorEnum.BinaryVecVecAdd, dtype=np.int32)
        assert shader_f32_2.cached
        assert shader_i32_2.cached


# =============================================================================
# Binary Vec-Vec Operations (Float32)
# =============================================================================

class TestBinaryVecVecFloat32:
    """Test binary vector-vector operations with float32."""

    @pytest.mark.parametrize("op_name,np_func", [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("minimum", np.minimum),
        ("maximum", np.maximum),
    ])
    def test_binary_op(self, backend, op_name, np_func):
        """Test binary operations."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(a_data, b_data)
        np.testing.assert_allclose(result.numpy(), expected)

    def test_divide(self, backend):
        """Test divide operation."""
        a_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        b_data = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = cc.divide(a, b)
        expected = a_data / b_data
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


# =============================================================================
# Binary Vec-Vec Operations (Int32)
# =============================================================================

class TestBinaryVecVecInt32:
    """Test binary vector-vector operations with int32."""

    @pytest.mark.parametrize("op_name,np_func", [
        ("add", np.add),
        ("subtract", np.subtract),
        ("multiply", np.multiply),
        ("minimum", np.minimum),
        ("maximum", np.maximum),
    ])
    def test_binary_op_int32(self, backend, op_name, np_func):
        """Test binary operations with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        b_data = np.array([5, 6, 7, 8], dtype=np.int32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(a_data, b_data)
        np.testing.assert_array_equal(result.numpy(), expected)


# =============================================================================
# Binary Vec-Scalar Operations
# =============================================================================

class TestBinaryVecScalar:
    """Test binary vector-scalar operations."""

    @pytest.mark.parametrize("op_name,scalar,np_op", [
        ("add_scalar", 10.0, lambda a, s: a + s),
        ("subtract_scalar", 5.0, lambda a, s: a - s),
        ("multiply_scalar", 3.0, lambda a, s: a * s),
        ("minimum_scalar", 4.0, lambda a, s: np.minimum(a, s)),
        ("maximum_scalar", 4.0, lambda a, s: np.maximum(a, s)),
    ])
    def test_vec_scalar_op(self, backend, op_name, scalar, np_op):
        """Test vector-scalar operations."""
        a_data = np.array([1.0, 5.0, 3.0, 8.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = getattr(cc, op_name)(a, scalar)
        expected = np_op(a_data, scalar)
        np.testing.assert_allclose(result.numpy(), expected)

    def test_divide_scalar(self, backend):
        """Test divide_scalar operation."""
        a_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        scalar = 5.0
        result = cc.divide_scalar(a, scalar)
        expected = a_data / scalar
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_add_scalar_int32(self, backend):
        """Test add_scalar with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        a = cc.Buffer(a_data)
        scalar = 10
        result = cc.add_scalar(a, scalar)
        expected = a_data + scalar
        np.testing.assert_array_equal(result.numpy(), expected)


# =============================================================================
# Comparison Operations
# =============================================================================

class TestComparisonOps:
    """Test comparison operations."""

    @pytest.mark.parametrize("op_name,np_func", [
        ("equal", np.equal),
        ("not_equal", np.not_equal),
        ("less", np.less),
        ("less_equal", np.less_equal),
        ("greater", np.greater),
        ("greater_equal", np.greater_equal),
    ])
    def test_comparison_op(self, backend, op_name, np_func):
        """Test comparison operations."""
        a_data = np.array([1.0, 5.0, 3.0, 4.0], dtype=np.float32)
        b_data = np.array([2.0, 4.0, 3.0, 5.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(a_data, b_data).astype(np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_scalar(self, backend):
        """Test equal_scalar comparison."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        scalar = 3.0
        result = cc.equal_scalar(a, scalar)
        expected = np.equal(a_data, scalar).astype(np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)


# =============================================================================
# Unary Operations
# =============================================================================

class TestUnaryOps:
    """Test unary operations."""

    @pytest.mark.parametrize("op_name,np_func,test_data", [
        ("negative", np.negative, np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)),
        ("abs", np.abs, np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)),
        ("floor", np.floor, np.array([1.5, 2.7, -1.5, -2.7], dtype=np.float32)),
        ("ceil", np.ceil, np.array([1.5, 2.7, -1.5, -2.7], dtype=np.float32)),
        ("round", np.round, np.array([1.4, 1.5, 2.5, -1.5], dtype=np.float32)),
        ("sign", np.sign, np.array([-5.0, 0.0, 5.0, -0.0], dtype=np.float32)),
    ])
    def test_unary_exact(self, backend, op_name, np_func, test_data):
        """Test unary operations with exact comparison."""
        a = cc.Buffer(test_data)
        result = getattr(cc, op_name)(a)
        expected = np_func(test_data)
        np.testing.assert_allclose(result.numpy(), expected)

    @pytest.mark.parametrize("op_name,np_func,test_data", [
        ("sqrt", np.sqrt, np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)),
        ("exp", np.exp, np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)),
        ("log", np.log, np.array([1.0, 2.718281828, 10.0, 100.0], dtype=np.float32)),
        ("tanh", np.tanh, np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)),
        ("reciprocal", np.reciprocal, np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)),
        ("square", np.square, np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)),
    ])
    def test_unary_approx(self, backend, op_name, np_func, test_data):
        """Test unary operations with approximate comparison."""
        a = cc.Buffer(test_data)
        result = getattr(cc, op_name)(a)
        expected = np_func(test_data)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_sin(self, backend):
        """Test sin operation."""
        a_data = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = cc.sin(a)
        expected = np.sin(a_data)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_cos(self, backend):
        """Test cos operation."""
        a_data = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = cc.cos(a)
        expected = np.cos(a_data)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# Operator Overloading Tests
# =============================================================================

class TestOperatorOverloading:
    """Test Python operator overloading on Buffers."""

    def test_add_operator(self, backend):
        """Test + operator."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = (a + b).numpy()
        expected = a_data + b_data
        np.testing.assert_allclose(result, expected)

    def test_sub_operator(self, backend):
        """Test - operator."""
        a_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        b_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = (a - b).numpy()
        expected = a_data - b_data
        np.testing.assert_allclose(result, expected)

    def test_mul_operator(self, backend):
        """Test * operator."""
        a_data = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b_data = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = (a * b).numpy()
        expected = a_data * b_data
        np.testing.assert_allclose(result, expected)

    def test_truediv_operator(self, backend):
        """Test / operator."""
        a_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        b_data = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = (a / b).numpy()
        expected = a_data / b_data
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_neg_operator(self, backend):
        """Test unary - operator."""
        a_data = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = (-a).numpy()
        expected = -a_data
        np.testing.assert_allclose(result, expected)

    def test_scalar_add(self, backend):
        """Test + operator with scalar."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = (a + 10.0).numpy()
        expected = a_data + 10.0
        np.testing.assert_allclose(result, expected)

    def test_scalar_mul(self, backend):
        """Test * operator with scalar."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a = cc.Buffer(a_data)
        result = (a * 3.0).numpy()
        expected = a_data * 3.0
        np.testing.assert_allclose(result, expected)


# =============================================================================
# Large Array Tests
# =============================================================================

class TestLargeArrays:
    """Test operations with larger arrays."""

    def test_large_add(self, backend):
        """Test add with large arrays."""
        n = 100000
        a_data = np.random.randn(n).astype(np.float32)
        b_data = np.random.randn(n).astype(np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = cc.add(a, b)
        expected = a_data + b_data
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_large_multiply(self, backend):
        """Test multiply with large arrays."""
        n = 100000
        a_data = np.random.randn(n).astype(np.float32)
        b_data = np.random.randn(n).astype(np.float32)
        a = cc.Buffer(a_data)
        b = cc.Buffer(b_data)
        result = cc.multiply(a, b)
        expected = a_data * b_data
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_large_exp(self, backend):
        """Test exp with large arrays (using safe values to avoid overflow)."""
        n = 100000
        a_data = np.random.uniform(-5, 5, n).astype(np.float32)
        a = cc.Buffer(a_data)
        result = cc.exp(a)
        expected = np.exp(a_data)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-5)


# =============================================================================
# Backend Switching Tests
# =============================================================================

class TestBackendSwitching:
    """Test switching between backends."""

    def test_switch_backends(self):
        """Test switching between CPU and Vulkan backends."""
        if cc.is_cpu_available():
            cc.init(cc.Backend.CPU)
            assert cc.current_backend() == cc.Backend.CPU

            data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            cpu_buf = cc.Buffer(data)
            cpu_result = cc.add(cpu_buf, cpu_buf).numpy()

            del cpu_buf
            cleanup()
            cc.shutdown()

            if cc.is_vulkan_available():
                cc.init(cc.Backend.Vulkan)
                assert cc.current_backend() == cc.Backend.Vulkan

                vk_buf = cc.Buffer(data)
                vk_result = cc.add(vk_buf, vk_buf).numpy()

                np.testing.assert_allclose(cpu_result, vk_result)

                del vk_buf
                cleanup()
                cc.shutdown()

    def test_simd_mode_switching(self, cpu_backend):
        """Test switching SIMD modes on CPU backend."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        cc.set_simd_mode(cc.SIMDMode.Scalar)
        buf1 = cc.Buffer(data)
        result1 = cc.add(buf1, buf1).numpy()

        cc.set_simd_mode(cc.SIMDMode.Auto)
        buf2 = cc.Buffer(data)
        result2 = cc.add(buf2, buf2).numpy()

        np.testing.assert_allclose(result1, result2)
