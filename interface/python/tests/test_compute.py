"""
Tests for the unified CUT compute interface.

These tests verify that all operations work correctly across different backends
(Vulkan and CPU) using the unified cut.compute module.

Note: numpy is used for verification/reference calculations only.
The cut library itself does not depend on numpy.
"""

import numpy as np
import pytest
import cut.compute as cc


# =============================================================================
# Test Utilities
# =============================================================================

def to_numpy(tensor):
    """Convert a Tensor to numpy array for verification."""
    flat = tensor.copy_to()
    arr = np.array(list(flat), dtype=_dtype_to_numpy(tensor.dtype))
    if len(tensor.shape) > 1:
        arr = arr.reshape(tensor.shape)
    return arr


def _dtype_to_numpy(dtype):
    """Convert cut DType to numpy dtype."""
    mapping = {
        'float32': np.float32,
        'float16': np.float16,
        'uint32': np.uint32,
        'int32': np.int32,
    }
    return mapping.get(str(dtype), np.float32)


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
    cc.shutdown()


@pytest.fixture
def cpu_backend():
    """Fixture for CPU backend only."""
    if not cc.is_cpu_available():
        pytest.skip("CPU backend not available")
    cc.init(cc.Backend.CPU, simd_mode=cc.SIMDMode.Auto)
    yield cc.Backend.CPU
    cc.shutdown()


@pytest.fixture
def vulkan_backend():
    """Fixture for Vulkan backend only."""
    if not cc.is_vulkan_available():
        pytest.skip("Vulkan backend not available")
    cc.init(cc.Backend.Vulkan)
    yield cc.Backend.Vulkan
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

    def test_dtype_exports(self):
        """Test DType instances are exported."""
        assert cc.float32 is not None
        assert cc.float16 is not None
        assert cc.int32 is not None
        assert cc.uint32 is not None


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
# Tensor Tests
# =============================================================================

class TestTensor:
    """Test Tensor class functionality."""

    def test_create_from_list(self, backend):
        """Test creating a tensor from Python list."""
        data = [1.0, 2.0, 3.0, 4.0]
        buf = cc.Tensor(data)
        assert buf.size == 16  # 4 floats * 4 bytes
        assert buf.handle.valid()

    def test_create_from_nested_list(self, backend):
        """Test creating a tensor from nested list."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        buf = cc.Tensor(data)
        assert buf.shape == (2, 2)
        assert buf.handle.valid()

    def test_create_empty(self, backend):
        """Test creating an empty tensor by size."""
        buf = cc.Tensor(size=64)
        assert buf.size == 64
        assert buf.handle.valid()

    def test_create_requires_data_or_size(self, backend):
        """Test that Tensor requires either data or size."""
        with pytest.raises(ValueError, match="Either data or size must be provided"):
            cc.Tensor()

    def test_roundtrip_float32(self, backend):
        """Test data roundtrip with float32."""
        data = [1.0, 2.0, 3.0, 4.0]
        buf = cc.Tensor(data)
        result = buf.tolist()
        assert result == data

    def test_roundtrip_int32(self, backend):
        """Test data roundtrip with int32."""
        data = [1, 2, 3, 4, 5]
        buf = cc.Tensor(data, dtype=cc.int32)
        result = buf.tolist()
        assert result == data

    def test_roundtrip_2d_array(self, backend):
        """Test data roundtrip with 2D array."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        buf = cc.Tensor(data)
        result = buf.tolist()
        assert result == data

    def test_copy_from(self, backend):
        """Test copying new data to existing tensor."""
        initial = [1.0, 2.0, 3.0, 4.0]
        buf = cc.Tensor(initial)

        new_data = [5.0, 6.0, 7.0, 8.0]
        buf.copy_from(new_data)

        result = buf.tolist()
        assert result == new_data

    def test_dtype_property(self, backend):
        """Test dtype property."""
        buf = cc.Tensor([1.0, 2.0, 3.0])
        assert buf.dtype == cc.float32

        buf_int = cc.Tensor([1, 2, 3], dtype=cc.int32)
        assert buf_int.dtype == cc.int32


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
        a_data = [1.0, 2.0, 3.0, 4.0]
        b_data = [5.0, 6.0, 7.0, 8.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(np.array(a_data, dtype=np.float32),
                          np.array(b_data, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(result), expected)

    def test_divide(self, backend):
        """Test divide operation."""
        a_data = [10.0, 20.0, 30.0, 40.0]
        b_data = [2.0, 4.0, 5.0, 8.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = cc.divide(a, b)
        expected = np.array(a_data) / np.array(b_data)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)


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
        a_data = [1, 2, 3, 4]
        b_data = [5, 6, 7, 8]
        a = cc.Tensor(a_data, dtype=cc.int32)
        b = cc.Tensor(b_data, dtype=cc.int32)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(np.array(a_data, dtype=np.int32),
                          np.array(b_data, dtype=np.int32))
        np.testing.assert_array_equal(to_numpy(result), expected)


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
        a_data = [1.0, 5.0, 3.0, 8.0]
        a = cc.Tensor(a_data)
        result = getattr(cc, op_name)(a, scalar)
        expected = np_op(np.array(a_data, dtype=np.float32), scalar)
        np.testing.assert_allclose(to_numpy(result), expected)

    def test_divide_scalar(self, backend):
        """Test divide_scalar operation."""
        a_data = [10.0, 20.0, 30.0, 40.0]
        a = cc.Tensor(a_data)
        scalar = 5.0
        result = cc.divide_scalar(a, scalar)
        expected = np.array(a_data) / scalar
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_add_scalar_int32(self, backend):
        """Test add_scalar with int32."""
        a_data = [1, 2, 3, 4]
        a = cc.Tensor(a_data, dtype=cc.int32)
        scalar = 10
        result = cc.add_scalar(a, scalar)
        expected = np.array(a_data, dtype=np.int32) + scalar
        np.testing.assert_array_equal(to_numpy(result), expected)


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
        a_data = [1.0, 5.0, 3.0, 4.0]
        b_data = [2.0, 4.0, 3.0, 5.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = getattr(cc, op_name)(a, b)
        expected = np_func(np.array(a_data), np.array(b_data)).astype(np.float32)
        np.testing.assert_array_equal(to_numpy(result), expected)

    def test_equal_scalar(self, backend):
        """Test equal_scalar comparison."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(a_data)
        scalar = 3.0
        result = cc.equal_scalar(a, scalar)
        expected = np.equal(np.array(a_data), scalar).astype(np.float32)
        np.testing.assert_array_equal(to_numpy(result), expected)


# =============================================================================
# Unary Operations
# =============================================================================

class TestUnaryOps:
    """Test unary operations."""

    @pytest.mark.parametrize("op_name,np_func,test_data", [
        ("negative", np.negative, [1.0, -2.0, 3.0, -4.0]),
        ("abs", np.abs, [-1.0, 2.0, -3.0, 4.0]),
        ("floor", np.floor, [1.5, 2.7, -1.5, -2.7]),
        ("ceil", np.ceil, [1.5, 2.7, -1.5, -2.7]),
        ("round", np.round, [1.4, 1.5, 2.5, -1.5]),
        ("sign", np.sign, [-5.0, 0.0, 5.0, -0.0]),
    ])
    def test_unary_exact(self, backend, op_name, np_func, test_data):
        """Test unary operations with exact comparison."""
        a = cc.Tensor(test_data)
        result = getattr(cc, op_name)(a)
        expected = np_func(np.array(test_data, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(result), expected)

    @pytest.mark.parametrize("op_name,np_func,test_data", [
        ("sqrt", np.sqrt, [1.0, 4.0, 9.0, 16.0]),
        ("exp", np.exp, [0.0, 1.0, 2.0, -1.0]),
        ("log", np.log, [1.0, 2.718281828, 10.0, 100.0]),
        ("tanh", np.tanh, [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("reciprocal", np.reciprocal, [1.0, 2.0, 4.0, 0.5]),
        ("square", np.square, [1.0, 2.0, 3.0, -4.0]),
    ])
    def test_unary_approx(self, backend, op_name, np_func, test_data):
        """Test unary operations with approximate comparison."""
        a = cc.Tensor(test_data)
        result = getattr(cc, op_name)(a)
        expected = np_func(np.array(test_data, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_sin(self, backend):
        """Test sin operation."""
        a_data = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        a = cc.Tensor(a_data)
        result = cc.sin(a)
        expected = np.sin(np.array(a_data, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5, atol=1e-5)

    def test_cos(self, backend):
        """Test cos operation."""
        a_data = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        a = cc.Tensor(a_data)
        result = cc.cos(a)
        expected = np.cos(np.array(a_data, dtype=np.float32))
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# Operator Overloading Tests
# =============================================================================

class TestOperatorOverloading:
    """Test Python operator overloading on Tensors."""

    def test_add_operator(self, backend):
        """Test + operator."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        b_data = [5.0, 6.0, 7.0, 8.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(a + b)
        expected = np.array(a_data) + np.array(b_data)
        np.testing.assert_allclose(result, expected)

    def test_sub_operator(self, backend):
        """Test - operator."""
        a_data = [10.0, 20.0, 30.0, 40.0]
        b_data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(a - b)
        expected = np.array(a_data) - np.array(b_data)
        np.testing.assert_allclose(result, expected)

    def test_mul_operator(self, backend):
        """Test * operator."""
        a_data = [2.0, 3.0, 4.0, 5.0]
        b_data = [3.0, 4.0, 5.0, 6.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(a * b)
        expected = np.array(a_data) * np.array(b_data)
        np.testing.assert_allclose(result, expected)

    def test_truediv_operator(self, backend):
        """Test / operator."""
        a_data = [10.0, 20.0, 30.0, 40.0]
        b_data = [2.0, 4.0, 5.0, 8.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(a / b)
        expected = np.array(a_data) / np.array(b_data)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_neg_operator(self, backend):
        """Test unary - operator."""
        a_data = [1.0, -2.0, 3.0, -4.0]
        a = cc.Tensor(a_data)
        result = to_numpy(-a)
        expected = -np.array(a_data)
        np.testing.assert_allclose(result, expected)

    def test_scalar_add(self, backend):
        """Test + operator with scalar."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(a_data)
        result = to_numpy(a + 10.0)
        expected = np.array(a_data) + 10.0
        np.testing.assert_allclose(result, expected)

    def test_scalar_mul(self, backend):
        """Test * operator with scalar."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(a_data)
        result = to_numpy(a * 3.0)
        expected = np.array(a_data) * 3.0
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
        a = cc.Tensor(a_data.tolist())
        b = cc.Tensor(b_data.tolist())
        result = cc.add(a, b)
        expected = a_data + b_data
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_large_multiply(self, backend):
        """Test multiply with large arrays."""
        n = 100000
        a_data = np.random.randn(n).astype(np.float32)
        b_data = np.random.randn(n).astype(np.float32)
        a = cc.Tensor(a_data.tolist())
        b = cc.Tensor(b_data.tolist())
        result = cc.multiply(a, b)
        expected = a_data * b_data
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_large_exp(self, backend):
        """Test exp with large arrays (using safe values to avoid overflow)."""
        n = 100000
        a_data = np.random.uniform(-5, 5, n).astype(np.float32)
        a = cc.Tensor(a_data.tolist())
        result = cc.exp(a)
        expected = np.exp(a_data)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-4, atol=1e-5)


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

            data = [1.0, 2.0, 3.0, 4.0]
            cpu_buf = cc.Tensor(data)
            cpu_result = to_numpy(cc.add(cpu_buf, cpu_buf))

            del cpu_buf
            cc.shutdown()

            if cc.is_vulkan_available():
                cc.init(cc.Backend.Vulkan)
                assert cc.current_backend() == cc.Backend.Vulkan

                vk_buf = cc.Tensor(data)
                vk_result = to_numpy(cc.add(vk_buf, vk_buf))

                np.testing.assert_allclose(cpu_result, vk_result)

                del vk_buf
                cc.shutdown()

    def test_simd_mode_switching(self, cpu_backend):
        """Test switching SIMD modes on CPU backend."""
        data = [1.0, 2.0, 3.0, 4.0]

        cc.set_simd_mode(cc.SIMDMode.Scalar)
        buf1 = cc.Tensor(data)
        result1 = to_numpy(cc.add(buf1, buf1))

        cc.set_simd_mode(cc.SIMDMode.Auto)
        buf2 = cc.Tensor(data)
        result2 = to_numpy(cc.add(buf2, buf2))

        np.testing.assert_allclose(result1, result2)
