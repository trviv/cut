"""
Tests for the unified CUT compute interface.

These tests verify that all operations work correctly across different backends
(Vulkan and CPU) using the unified cut.compute module.

Note: numpy is used for verification/reference calculations only.
The cut library itself does not depend on numpy.
"""

import builtins
import numpy as np
import pytest
import cut.compute as cc

builtins_max = builtins.max


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
    cc.init(backend_type)
    yield backend_type
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

    def test_backend_enum(self):
        """Test Backend enum values."""
        assert hasattr(cc.Backend, "Vulkan")

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


    def test_init_vulkan(self):
        """Test Vulkan backend initialization."""
        if not cc.is_vulkan_available():
            pytest.skip("Vulkan backend not available")

        backend = cc.init(cc.Backend.Vulkan)
        assert backend == cc.Backend.Vulkan
        assert cc.current_backend() == cc.Backend.Vulkan
        cc.shutdown()



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


# =============================================================================
# Dimension-wise Reduction Tests
# =============================================================================

class TestDimReduce:
    """Test dimension-wise reduction operations."""

    def test_sum_dim0_2d(self, backend):
        """Test sum along dimension 0 of a 2D tensor."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=0)
        expected = data.sum(axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_sum_dim1_2d(self, backend):
        """Test sum along dimension 1 of a 2D tensor."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1)
        expected = data.sum(axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_mean_dim0(self, backend):
        """Test mean along dimension 0."""
        data = np.array([[2, 4], [6, 8], [10, 12]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.mean(t, dim=0)
        expected = data.mean(axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_mean_dim1(self, backend):
        """Test mean along dimension 1."""
        data = np.array([[2, 4], [6, 8], [10, 12]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.mean(t, dim=1)
        expected = data.mean(axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_min_dim0(self, backend):
        """Test min along dimension 0."""
        data = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.min(t, dim=0)
        expected = data.min(axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_max_dim1(self, backend):
        """Test max along dimension 1."""
        data = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.max(t, dim=1)
        expected = data.max(axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_prod_dim0(self, backend):
        """Test prod along dimension 0."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.prod(t, dim=0)
        expected = data.prod(axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_sum_negative_dim(self, backend):
        """Test sum with negative dimension index."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=-1)
        expected = data.sum(axis=-1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_sum_dim_3d(self, backend):
        """Test sum along middle dimension of a 3D tensor."""
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1)
        expected = data.sum(axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_sum_dim_output_shape(self, backend):
        """Test that dim reduce produces correct output shape."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1)
        assert result.shape == (3, 5)

    def test_dim_out_of_range(self, backend):
        """Test that invalid dimension raises ValueError."""
        t = cc.Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            cc.sum(t, dim=3)

    def test_large_dim_reduce(self, backend):
        """Test dim reduce with larger arrays."""
        data = np.random.randn(64, 128).astype(np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1)
        expected = data.sum(axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-4)


# =============================================================================
# Keepdim Tests
# =============================================================================

class TestKeepdim:
    """Test keepdim argument for reduction operations."""

    def test_sum_keepdim_dim0(self, backend):
        """Test sum with keepdim=True along dim 0."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=0, keepdim=True)
        assert result.shape == (1, 3)
        expected = data.sum(axis=0, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_sum_keepdim_dim1(self, backend):
        """Test sum with keepdim=True along dim 1."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1, keepdim=True)
        assert result.shape == (2, 1)
        expected = data.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_mean_keepdim(self, backend):
        """Test mean with keepdim=True."""
        data = np.array([[2, 4], [6, 8], [10, 12]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.mean(t, dim=0, keepdim=True)
        assert result.shape == (1, 2)
        expected = data.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_max_keepdim(self, backend):
        """Test max with keepdim=True."""
        data = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.max(t, dim=1, keepdim=True)
        assert result.shape == (2, 1)
        expected = data.max(axis=1, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_min_keepdim(self, backend):
        """Test min with keepdim=True."""
        data = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.min(t, dim=0, keepdim=True)
        assert result.shape == (1, 3)
        expected = data.min(axis=0, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_prod_keepdim(self, backend):
        """Test prod with keepdim=True."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.prod(t, dim=0, keepdim=True)
        assert result.shape == (1, 2)
        expected = data.prod(axis=0, keepdims=True)
        np.testing.assert_allclose(to_numpy(result).reshape(expected.shape), expected, rtol=1e-5)

    def test_keepdim_3d(self, backend):
        """Test keepdim on a 3D tensor reduces middle dim."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=1, keepdim=True)
        assert result.shape == (3, 1, 5)

    def test_keepdim_false_unchanged(self, backend):
        """Test that keepdim=False (default) matches original behavior."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.sum(t, dim=0, keepdim=False)
        assert result.shape == (3,)
        expected = data.sum(axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_keepdim_values_match(self, backend):
        """Test that keepdim=True produces same values as keepdim=False."""
        data = np.random.randn(4, 5).astype(np.float32)
        t = cc.Tensor(data.tolist())
        result_no_keep = cc.sum(t, dim=1)
        result_keep = cc.sum(t, dim=1, keepdim=True)
        np.testing.assert_allclose(
            to_numpy(result_keep).flatten(),
            to_numpy(result_no_keep),
            rtol=1e-5
        )


# =============================================================================
# Dimension-wise Norm Tests
# =============================================================================

class TestNormDim:
    """Test dimension-wise L2 norm operation."""

    def test_norm_dim0_2d(self, backend):
        """Test L2 norm along dimension 0."""
        data = np.array([[3, 1], [4, 5]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=0)
        expected = np.linalg.norm(data, axis=0)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_norm_dim1_2d(self, backend):
        """Test L2 norm along dimension 1."""
        data = np.array([[3, 4], [5, 12]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=1)
        expected = np.linalg.norm(data, axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_norm_dim_3d(self, backend):
        """Test L2 norm along middle dimension of a 3D tensor."""
        data = np.random.randn(2, 3, 4).astype(np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=1)
        expected = np.linalg.norm(data, axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_norm_dim_output_shape(self, backend):
        """Test that norm dim produces correct output shape."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=2)
        assert result.shape == (3, 4)

    def test_norm_dim_negative_index(self, backend):
        """Test L2 norm with negative dimension index."""
        data = np.array([[3, 4], [5, 12]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=-1)
        expected = np.linalg.norm(data, axis=-1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_norm_dim_single_element(self, backend):
        """Test norm along dimension with size 1."""
        data = np.array([[3], [4], [5]], dtype=np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=1)
        expected = np.linalg.norm(data, axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_norm_dim_large(self, backend):
        """Test dim norm with larger arrays."""
        data = np.random.randn(32, 64).astype(np.float32)
        t = cc.Tensor(data.tolist())
        result = cc.norm(t, dim=1)
        expected = np.linalg.norm(data, axis=1)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_norm_dim_non_l2_raises(self, backend):
        """Test that non-L2 norm with dim raises NotImplementedError."""
        t = cc.Tensor([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError):
            cc.norm(t, p=1, dim=0)


# =============================================================================
# Extended Unary Activation Tests
# =============================================================================

class TestExtendedActivations:
    """Test extended unary activation functions."""

    def test_relu6(self, backend):
        """Test ReLU6: clamp(x, 0, 6)."""
        data = [-2.0, 0.0, 3.0, 7.0, 10.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.relu6(t))
        expected = np.clip(np.array(data, dtype=np.float32), 0.0, 6.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_elu(self, backend):
        """Test ELU: x if x >= 0, exp(x) - 1 if x < 0."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.elu(t))
        arr = np.array(data, dtype=np.float32)
        expected = np.where(arr >= 0, arr, np.exp(arr) - 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_selu(self, backend):
        """Test SELU activation."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.selu(t))
        arr = np.array(data, dtype=np.float32)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        expected = scale * np.where(arr >= 0, arr, alpha * (np.exp(arr) - 1.0))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_celu(self, backend):
        """Test CELU: max(0, x) + min(0, exp(x) - 1)."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.celu(t))
        arr = np.array(data, dtype=np.float32)
        expected = np.maximum(arr, 0.0) + np.minimum(0.0, np.exp(arr) - 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mish(self, backend):
        """Test Mish: x * tanh(softplus(x))."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.mish(t))
        arr = np.array(data, dtype=np.float32)
        expected = arr * np.tanh(np.log(1.0 + np.exp(arr)))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_hardswish(self, backend):
        """Test HardSwish: x * clamp(x + 3, 0, 6) / 6."""
        data = [-4.0, -3.0, 0.0, 3.0, 4.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.hardswish(t))
        arr = np.array(data, dtype=np.float32)
        expected = arr * np.clip(arr + 3.0, 0.0, 6.0) / 6.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardsigmoid(self, backend):
        """Test HardSigmoid: clamp(x/6 + 0.5, 0, 1)."""
        data = [-4.0, -3.0, 0.0, 3.0, 4.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.hardsigmoid(t))
        arr = np.array(data, dtype=np.float32)
        expected = np.clip(arr / 6.0 + 0.5, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardtanh(self, backend):
        """Test HardTanh: clamp(x, -1, 1)."""
        data = [-2.0, -0.5, 0.0, 0.5, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.hardtanh(t))
        expected = np.clip(np.array(data, dtype=np.float32), -1.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_softsign(self, backend):
        """Test Softsign: x / (1 + |x|)."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.softsign(t))
        arr = np.array(data, dtype=np.float32)
        expected = arr / (1.0 + np.abs(arr))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logsigmoid(self, backend):
        """Test LogSigmoid: -log(1 + exp(-x))."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.logsigmoid(t))
        arr = np.array(data, dtype=np.float32)
        expected = -np.log(1.0 + np.exp(-arr))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_tanhshrink(self, backend):
        """Test Tanhshrink: x - tanh(x)."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.tanhshrink(t))
        arr = np.array(data, dtype=np.float32)
        expected = arr - np.tanh(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# =============================================================================
# Extended Binary-Scalar Activation Tests
# =============================================================================

class TestExtendedBinaryScalarActivations:
    """Test extended binary vec-scalar activation functions."""

    def test_prelu(self, backend):
        """Test PReLU: x if x >= 0, weight * x otherwise."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        weight = 0.25
        t = cc.Tensor(data)
        result = to_numpy(cc.prelu(t, weight))
        arr = np.array(data, dtype=np.float32)
        expected = np.where(arr >= 0, arr, weight * arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_hardshrink(self, backend):
        """Test Hardshrink: x if |x| > lambda, 0 otherwise."""
        data = [-2.0, -0.3, 0.0, 0.3, 2.0]
        lambd = 0.5
        t = cc.Tensor(data)
        result = to_numpy(cc.hardshrink(t, lambd))
        arr = np.array(data, dtype=np.float32)
        expected = np.where(np.abs(arr) > lambd, arr, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_softshrink(self, backend):
        """Test Softshrink: sign(x) * max(|x| - lambda, 0)."""
        data = [-2.0, -0.3, 0.0, 0.3, 2.0]
        lambd = 0.5
        t = cc.Tensor(data)
        result = to_numpy(cc.softshrink(t, lambd))
        arr = np.array(data, dtype=np.float32)
        expected = np.sign(arr) * np.maximum(np.abs(arr) - lambd, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# =============================================================================
# Extended Unary Math Tests
# =============================================================================

class TestExtendedUnaryMath:
    """Test extended unary math functions."""

    def test_rsqrt(self, backend):
        """Test reciprocal square root."""
        data = [1.0, 4.0, 9.0, 16.0, 25.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.rsqrt(t))
        expected = 1.0 / np.sqrt(np.array(data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_trunc(self, backend):
        """Test truncation to integer."""
        data = [1.7, -1.7, 2.5, -2.5, 0.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.trunc(t))
        expected = np.trunc(np.array(data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_frac(self, backend):
        """Test fractional part (GLSL fract = x - floor(x), always positive)."""
        data = [1.7, -1.7, 2.5, 0.0, 3.14]
        t = cc.Tensor(data)
        result = to_numpy(cc.frac(t))
        arr = np.array(data, dtype=np.float32)
        expected = arr - np.floor(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_arcsinh(self, backend):
        """Test inverse hyperbolic sine."""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.arcsinh(t))
        expected = np.arcsinh(np.array(data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_arccosh(self, backend):
        """Test inverse hyperbolic cosine."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.arccosh(t))
        expected = np.arccosh(np.array(data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_arctanh(self, backend):
        """Test inverse hyperbolic tangent."""
        data = [-0.9, -0.5, 0.0, 0.5, 0.9]
        t = cc.Tensor(data)
        result = to_numpy(cc.arctanh(t))
        expected = np.arctanh(np.array(data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_isfinite(self, backend):
        """Test isfinite check."""
        data = [1.0, 0.0, -1.0, 2.0, 3.0]
        t = cc.Tensor(data)
        result = to_numpy(cc.isfinite(t))
        expected = np.isfinite(np.array(data, dtype=np.float32)).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# =============================================================================
# Logaddexp Binary Op Tests
# =============================================================================

class TestLogaddexp:
    """Test logaddexp and logaddexp2 operations."""

    def test_logaddexp(self, backend):
        """Test logaddexp: log(exp(a) + exp(b))."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        b_data = [2.0, 3.0, 4.0, 5.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(cc.logaddexp(a, b))
        expected = np.logaddexp(np.array(a_data, dtype=np.float32),
                                np.array(b_data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logaddexp2(self, backend):
        """Test logaddexp2: log2(2^a + 2^b)."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        b_data = [2.0, 3.0, 4.0, 5.0]
        a = cc.Tensor(a_data)
        b = cc.Tensor(b_data)
        result = to_numpy(cc.logaddexp2(a, b))
        expected = np.logaddexp2(np.array(a_data, dtype=np.float32),
                                 np.array(b_data, dtype=np.float32))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logaddexp_scalar(self, backend):
        """Test logaddexp with scalar."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        scalar = 2.0
        a = cc.Tensor(a_data)
        result = to_numpy(cc.logaddexp_scalar(a, scalar))
        expected = np.logaddexp(np.array(a_data, dtype=np.float32), scalar)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_logaddexp2_scalar(self, backend):
        """Test logaddexp2 with scalar."""
        a_data = [1.0, 2.0, 3.0, 4.0]
        scalar = 2.0
        a = cc.Tensor(a_data)
        result = to_numpy(cc.logaddexp2_scalar(a, scalar))
        expected = np.logaddexp2(np.array(a_data, dtype=np.float32), scalar)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# =============================================================================
# Tensor Creation Helper Tests
# =============================================================================

class TestTensorCreationHelpers:
    """Test zeros_like, ones_like, full_like."""

    def test_zeros_like(self, backend):
        """Test zeros_like creates matching shape of zeros."""
        a = cc.Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = cc.zeros_like(a)
        assert result.shape == a.shape
        assert result.dtype == a.dtype
        np.testing.assert_array_equal(to_numpy(result), np.zeros((2, 2), dtype=np.float32))

    def test_ones_like(self, backend):
        """Test ones_like creates matching shape of ones."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = cc.ones_like(a)
        assert result.shape == a.shape
        np.testing.assert_array_equal(to_numpy(result), np.ones(4, dtype=np.float32))

    def test_full_like(self, backend):
        """Test full_like creates matching shape filled with value."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = cc.full_like(a, 3.14)
        assert result.shape == a.shape
        expected = np.full(4, 3.14, dtype=np.float32)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)


# =============================================================================
# Var/Std Tests
# =============================================================================

class TestVarStd:
    """Test variance and standard deviation."""

    def test_var_global(self, backend):
        """Test global variance."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = cc.Tensor(data)
        result = cc.var(a)
        expected = np.var(np.array(data, dtype=np.float32), ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_std_global(self, backend):
        """Test global standard deviation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = cc.Tensor(data)
        result = cc.std(a)
        expected = np.std(np.array(data, dtype=np.float32), ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_var_dim(self, backend):
        """Test variance along dimension 0."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        a = cc.Tensor(data)
        result = to_numpy(cc.var(a, dim=0))
        expected = np.var(np.array(data, dtype=np.float32), axis=0, ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-4)


# =============================================================================
# Softmax Tests
# =============================================================================

class TestSoftmax:
    """Test softmax and log_softmax."""

    def test_softmax_1d(self, backend):
        """Test softmax on 1D input."""
        data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(data)
        result = to_numpy(cc.softmax(a, dim=0))
        arr = np.array(data, dtype=np.float32)
        expected = np.exp(arr) / np.sum(np.exp(arr))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_softmax_2d(self, backend):
        """Test softmax on 2D input along dim=0."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        a = cc.Tensor(data)
        result = cc.softmax(a, dim=0)
        result_flat = list(result.copy_to())
        arr = np.array(data, dtype=np.float32)
        exp_arr = np.exp(arr - arr.max(axis=0, keepdims=True))
        expected = (exp_arr / exp_arr.sum(axis=0, keepdims=True)).flatten()
        np.testing.assert_allclose(result_flat, expected, rtol=1e-5)

    def test_log_softmax_1d(self, backend):
        """Test log_softmax on 1D input."""
        data = [1.0, 2.0, 3.0, 4.0]
        a = cc.Tensor(data)
        result = to_numpy(cc.log_softmax(a, dim=0))
        arr = np.array(data, dtype=np.float32)
        log_sum_exp = np.log(np.sum(np.exp(arr)))
        expected = arr - log_sum_exp
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# =============================================================================
# Loss Function Tests
# =============================================================================

class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self, backend):
        """Test mean squared error loss."""
        pred = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        target = cc.Tensor([1.5, 2.5, 3.5, 4.5])
        result = cc.mse_loss(pred, target)
        expected = np.mean((np.array([1.0, 2.0, 3.0, 4.0]) - np.array([1.5, 2.5, 3.5, 4.5])) ** 2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_l1_loss(self, backend):
        """Test L1 loss."""
        pred = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        target = cc.Tensor([1.5, 2.5, 3.5, 4.5])
        result = cc.l1_loss(pred, target)
        expected = np.mean(np.abs(np.array([1.0, 2.0, 3.0, 4.0]) - np.array([1.5, 2.5, 3.5, 4.5])))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mse_loss_none_reduction(self, backend):
        """Test MSE loss with no reduction."""
        pred = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        target = cc.Tensor([2.0, 3.0, 4.0, 5.0])
        result = to_numpy(cc.mse_loss(pred, target, reduction='none'))
        expected = (np.array([1.0, 2.0, 3.0, 4.0]) - np.array([2.0, 3.0, 4.0, 5.0])) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestShapeOps:
    """Test reshape, view, squeeze, unsqueeze, unflatten."""

    def test_reshape_1d_to_2d(self, backend):
        """Test reshaping a 1D tensor to 2D."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = cc.reshape(a, 2, 3)
        assert result.shape == (2, 3)
        data = list(result.copy_to())
        assert data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_reshape_2d_to_1d(self, backend):
        """Test reshaping a 2D tensor to 1D."""
        a = cc.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = cc.reshape(a, 6)
        assert result.shape == (6,)
        data = list(result.copy_to())
        assert data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_reshape_with_negative_one(self, backend):
        """Test reshape with -1 for inferred dimension."""
        a = cc.Tensor(list(range(12)), shape=(12,))
        result = cc.reshape(a, 3, -1)
        assert result.shape == (3, 4)
        result2 = cc.reshape(a, -1, 6)
        assert result2.shape == (2, 6)

    def test_reshape_tuple_arg(self, backend):
        """Test reshape with a tuple argument."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = cc.reshape(a, (2, 2))
        assert result.shape == (2, 2)

    def test_reshape_invalid(self, backend):
        """Test reshape with mismatched sizes raises error."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            cc.reshape(a, 3, 3)

    def test_reshape_preserves_data(self, backend):
        """Test that reshape is a zero-copy view (same GPU buffer)."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = cc.reshape(a, 2, 3)
        # Verify data is the same
        np.testing.assert_array_equal(list(a.copy_to()), list(b.copy_to()))

    def test_view(self, backend):
        """Test that view is an alias for reshape."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = cc.view(a, 2, 2)
        assert result.shape == (2, 2)

    def test_squeeze_all(self, backend):
        """Test squeezing all size-1 dimensions."""
        a = cc.Tensor([1.0, 2.0, 3.0], shape=(1, 3, 1))
        result = cc.squeeze(a)
        assert result.shape == (3,)
        data = list(result.copy_to())
        assert data == [1.0, 2.0, 3.0]

    def test_squeeze_specific_dim(self, backend):
        """Test squeezing a specific dimension."""
        a = cc.Tensor([1.0, 2.0, 3.0], shape=(1, 3, 1))
        result = cc.squeeze(a, dim=0)
        assert result.shape == (3, 1)
        result2 = cc.squeeze(a, dim=2)
        assert result2.shape == (1, 3)

    def test_squeeze_noop(self, backend):
        """Test squeeze is no-op when dim is not size 1."""
        a = cc.Tensor([1.0, 2.0, 3.0], shape=(1, 3))
        result = cc.squeeze(a, dim=1)
        assert result.shape == (1, 3)

    def test_unsqueeze(self, backend):
        """Test unsqueeze adds a dimension."""
        a = cc.Tensor([1.0, 2.0, 3.0])
        r0 = cc.unsqueeze(a, 0)
        assert r0.shape == (1, 3)
        r1 = cc.unsqueeze(a, 1)
        assert r1.shape == (3, 1)
        r_neg = cc.unsqueeze(a, -1)
        assert r_neg.shape == (3, 1)

    def test_unsqueeze_2d(self, backend):
        """Test unsqueeze on a 2D tensor."""
        a = cc.Tensor([[1.0, 2.0], [3.0, 4.0]])
        r0 = cc.unsqueeze(a, 0)
        assert r0.shape == (1, 2, 2)
        r1 = cc.unsqueeze(a, 1)
        assert r1.shape == (2, 1, 2)
        r2 = cc.unsqueeze(a, 2)
        assert r2.shape == (2, 2, 1)

    def test_unflatten(self, backend):
        """Test unflattening a dimension."""
        a = cc.Tensor(list(range(12)), shape=(12,))
        result = cc.unflatten(a, 0, (3, 4))
        assert result.shape == (3, 4)

    def test_unflatten_middle_dim(self, backend):
        """Test unflattening a middle dimension."""
        a = cc.Tensor(list(range(24)), shape=(2, 12))
        result = cc.unflatten(a, 1, (3, 4))
        assert result.shape == (2, 3, 4)

    def test_unflatten_invalid(self, backend):
        """Test unflatten with mismatched sizes raises error."""
        a = cc.Tensor(list(range(12)), shape=(12,))
        with pytest.raises(ValueError):
            cc.unflatten(a, 0, (3, 5))

    def test_reshape_then_ops(self, backend):
        """Test that reshaped tensors work with operations."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = cc.reshape(a, 2, 3)
        # Sum should still work on the reshaped tensor
        total = cc.sum(b)
        np.testing.assert_allclose(total, 21.0, rtol=1e-5)

    def test_squeeze_unsqueeze_roundtrip(self, backend):
        """Test that squeeze and unsqueeze are inverses."""
        a = cc.Tensor([1.0, 2.0, 3.0])
        b = cc.unsqueeze(a, 0)
        assert b.shape == (1, 3)
        c = cc.squeeze(b, dim=0)
        assert c.shape == (3,)
        np.testing.assert_array_equal(list(a.copy_to()), list(c.copy_to()))


class TestArgmaxArgmin:
    """Test argmax/argmin global and dim-wise reductions."""

    def test_argmax_global(self, backend):
        """Test global argmax."""
        a = cc.Tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = cc.argmax(a)
        assert result == 5  # index of 9.0

    def test_argmin_global(self, backend):
        """Test global argmin."""
        a = cc.Tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = cc.argmin(a)
        assert result == 1  # index of first 1.0

    def test_argmax_global_single(self, backend):
        """Test argmax with a single element."""
        a = cc.Tensor([42.0])
        assert cc.argmax(a) == 0

    def test_argmin_global_negative(self, backend):
        """Test argmin with negative values."""
        a = cc.Tensor([5.0, -3.0, 2.0, -7.0])
        assert cc.argmin(a) == 3  # index of -7.0

    def test_argmax_dim0(self, backend):
        """Test argmax along dim 0."""
        # Shape (3, 4): 3 rows, 4 cols
        a = cc.Tensor([
            1.0, 5.0, 3.0, 2.0,
            4.0, 2.0, 6.0, 1.0,
            3.0, 8.0, 1.0, 7.0,
        ], shape=(3, 4))
        result = cc.argmax(a, dim=0)
        # For each column, which row has the max?
        # col 0: max is 4.0 at row 1
        # col 1: max is 8.0 at row 2
        # col 2: max is 6.0 at row 1
        # col 3: max is 7.0 at row 2
        expected = [1.0, 2.0, 1.0, 2.0]
        np.testing.assert_array_equal(list(result.copy_to()), expected)

    def test_argmin_dim0(self, backend):
        """Test argmin along dim 0."""
        a = cc.Tensor([
            1.0, 5.0, 3.0, 2.0,
            4.0, 2.0, 6.0, 1.0,
            3.0, 8.0, 1.0, 7.0,
        ], shape=(3, 4))
        result = cc.argmin(a, dim=0)
        # col 0: min is 1.0 at row 0
        # col 1: min is 2.0 at row 1
        # col 2: min is 1.0 at row 2
        # col 3: min is 1.0 at row 1
        expected = [0.0, 1.0, 2.0, 1.0]
        np.testing.assert_array_equal(list(result.copy_to()), expected)

    def test_argmax_large(self, backend):
        """Test argmax with larger tensor."""
        import random
        random.seed(42)
        data = [random.random() for _ in range(1024)]
        a = cc.Tensor(data)
        result = cc.argmax(a)
        expected = data.index(builtins_max(data))
        assert result == expected


class TestCumsumCumprod:
    """Test cumulative sum and product."""

    def test_cumsum_1d(self, backend):
        """Test cumulative sum on 1D tensor."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = list(cc.cumsum(a).copy_to())
        expected = [1.0, 3.0, 6.0, 10.0]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumprod_1d(self, backend):
        """Test cumulative product on 1D tensor."""
        a = cc.Tensor([1.0, 2.0, 3.0, 4.0])
        result = list(cc.cumprod(a).copy_to())
        expected = [1.0, 2.0, 6.0, 24.0]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumsum_2d_dim0(self, backend):
        """Test cumsum along dim 0 on a 2D tensor."""
        a = cc.Tensor([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ], shape=(3, 4))
        result = list(cc.cumsum(a, dim=0).copy_to())
        # Row 0: [1, 2, 3, 4]
        # Row 1: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
        # Row 2: [6+9, 8+10, 10+11, 12+12] = [15, 18, 21, 24]
        expected = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumprod_2d_dim0(self, backend):
        """Test cumprod along dim 0 on a 2D tensor."""
        a = cc.Tensor([
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 1.0, 2.0,
        ], shape=(2, 4))
        result = list(cc.cumprod(a, dim=0).copy_to())
        expected = [1.0, 2.0, 3.0, 4.0, 2.0, 6.0, 3.0, 8.0]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumsum_preserves_shape(self, backend):
        """Test that cumsum output has same shape as input."""
        a = cc.Tensor(list(range(1, 13)), shape=(3, 4))
        result = cc.cumsum(a, dim=0)
        assert result.shape == a.shape

    def test_cumsum_single_element(self, backend):
        """Test cumsum with single element."""
        a = cc.Tensor([5.0])
        result = list(cc.cumsum(a).copy_to())
        assert result == [5.0]

    def test_cumsum_2d_dim1(self, backend):
        """Test cumsum along dim 1 on a 2D tensor."""
        a = cc.Tensor([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ], shape=(3, 4))
        result = list(cc.cumsum(a, dim=1).copy_to())
        expected = list(np.cumsum(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ], dtype=np.float32), axis=1).flatten())
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumprod_2d_dim1(self, backend):
        """Test cumprod along dim 1 on a 2D tensor."""
        a = cc.Tensor([
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 1.0, 2.0,
        ], shape=(2, 4))
        result = list(cc.cumprod(a, dim=1).copy_to())
        expected = list(np.cumprod(np.array([
            [1, 2, 3, 4],
            [2, 3, 1, 2],
        ], dtype=np.float32), axis=1).flatten())
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cumsum_large(self, backend):
        """Test cumsum on a large 1D tensor."""
        data = list(range(1, 1025))
        a = cc.Tensor([float(x) for x in data])
        result = list(cc.cumsum(a).copy_to())
        expected = list(np.cumsum(np.array(data, dtype=np.float32)))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_cumsum_3d(self, backend):
        """Test cumsum on a 3D tensor along each dimension."""
        data = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        a = cc.Tensor(data.flatten().tolist(), shape=(2, 3, 4))
        for dim in range(3):
            result = list(cc.cumsum(a, dim=dim).copy_to())
            expected = list(np.cumsum(data, axis=dim).flatten())
            np.testing.assert_allclose(result, expected, rtol=1e-5,
                                       err_msg=f"cumsum failed for dim={dim}")
