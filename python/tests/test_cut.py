"""Tests for the CUT Python bindings."""

import numpy as np
import pytest
import cut


class TestImport:
    """Test basic module import and initialization."""

    def test_import(self):
        """Test that the module can be imported."""
        assert cut.__version__ == "0.1.0"

    def test_operator_enum(self):
        """Test OperatorEnum is accessible."""
        assert hasattr(cut, "OperatorEnum")
        assert hasattr(cut.OperatorEnum, "BinaryVecVecAdd")

    def test_shader_enum_alias(self):
        """Test ShaderEnum is an alias for OperatorEnum (backward compatibility)."""
        assert hasattr(cut, "ShaderEnum")
        assert hasattr(cut.ShaderEnum, "BinaryVecVecAdd")
        # ShaderEnum should be the same as OperatorEnum
        assert cut.ShaderEnum is cut.OperatorEnum

    def test_get_interface(self):
        """Test get_interface returns a valid interface."""
        interface = cut.get_interface()
        assert interface is not None


class TestBuffer:
    """Test Buffer class functionality."""

    def test_create_from_array(self):
        """Test creating a buffer from numpy array."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(data)
        assert buf.size == data.nbytes
        assert buf.handle.valid()

    def test_create_empty(self):
        """Test creating an empty buffer by size."""
        buf = cut.Buffer(size=64)
        assert buf.size == 64
        assert buf.handle.valid()

    def test_create_requires_data_or_size(self):
        """Test that Buffer requires either data or size."""
        with pytest.raises(ValueError, match="Either data or size must be provided"):
            cut.Buffer()

    def test_roundtrip_float32(self):
        """Test data roundtrip with float32."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_int32(self):
        """Test data roundtrip with int32."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        buf = cut.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_2d_array(self):
        """Test data roundtrip with 2D array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        buf = cut.Buffer(data)
        result = buf.numpy()
        np.testing.assert_array_equal(result, data)

    def test_copy_from(self):
        """Test copying new data to existing buffer."""
        initial = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(initial)

        new_data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buf.copy_from(new_data)

        result = buf.numpy()
        np.testing.assert_array_equal(result, new_data)

    def test_copy_to_preallocated(self):
        """Test copying to a preallocated array."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(data)

        out = np.zeros(4, dtype=np.float32)
        result = buf.copy_to(out)

        assert result is out
        np.testing.assert_array_equal(result, data)

    def test_uniform_buffer(self):
        """Test creating a uniform buffer."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(data, is_uniform=True)
        assert buf.handle.valid()


class TestShader:
    """Test Shader class functionality."""

    def test_create_from_enum(self):
        """Test creating shader from ShaderEnum."""
        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        assert shader.handle.valid()

    def test_create_from_spirv(self):
        """Test creating shader from SPIR-V bytecode."""
        from cut import _cut_core
        spirv = _cut_core.get_shader(cut.ShaderEnum.BinaryVecVecAdd, _cut_core.ScalarDataType.Float)
        shader = cut.Shader(spirv)
        assert shader.handle.valid()


class TestDispatch:
    """Test Dispatch class functionality."""

    def test_create_dispatch(self):
        """Test creating a dispatch."""
        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        dispatch = cut.Dispatch(shader, thread_groups=(1, 1, 1))
        assert dispatch.inner is not None

    def test_bind_buffer(self):
        """Test binding a buffer."""
        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        dispatch = cut.Dispatch(shader, thread_groups=(1, 1, 1))

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = cut.Buffer(data)

        result = dispatch.bind(buf, 0)
        assert result is dispatch  # Returns self for chaining

    def test_bind_push_constant(self):
        """Test binding push constant data."""
        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        dispatch = cut.Dispatch(shader, thread_groups=(1, 1, 1))

        push_data = np.array([4], dtype=np.uint32)
        result = dispatch.bind(push_data, 3)
        assert result is dispatch

    def test_chained_bindings(self):
        """Test chaining multiple binds."""
        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf_a = cut.Buffer(data)
        buf_b = cut.Buffer(data)
        buf_out = cut.Buffer(size=data.nbytes)

        dispatch = (cut.Dispatch(shader, thread_groups=(1, 1, 1))
                    .bind(buf_a, 0)
                    .bind(buf_b, 1)
                    .bind(buf_out, 2))

        assert dispatch is not None


class TestRun:
    """Test the run function for executing dispatches."""

    def test_run_single_dispatch(self):
        """Test running a single dispatch using the add shader."""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        buf_a = cut.Buffer(a)
        buf_b = cut.Buffer(b)
        buf_out = cut.Buffer(size=a.nbytes)

        shader = cut.Shader(cut.ShaderEnum.BinaryVecVecAdd)
        num_elements = a.size

        dispatch = (cut.Dispatch(shader, thread_groups=(num_elements, 1, 1))
                    .bind(buf_a, 0)
                    .bind(buf_b, 1)
                    .bind(buf_out, 2)
                    .bind(num_elements, 3))

        cut.run(dispatch)

        result = buf_out.copy_to(np.zeros(4, dtype=np.float32))
        expected = a + b
        np.testing.assert_allclose(result, expected)


class TestThreadSize:
    """Test ThreadSize from core module."""

    def test_create_default(self):
        """Test creating with default values."""
        from cut import _cut_core
        tgs = _cut_core.ThreadSize()
        assert tgs.x == 0
        assert tgs.y == 0
        assert tgs.z == 0

    def test_create_with_values(self):
        """Test creating with specific values."""
        from cut import _cut_core
        tgs = _cut_core.ThreadSize(64, 1, 1)
        assert tgs.x == 64
        assert tgs.y == 1
        assert tgs.z == 1

    def test_modify_values(self):
        """Test modifying values."""
        from cut import _cut_core
        tgs = _cut_core.ThreadSize(1, 1, 1)
        tgs.x = 128
        tgs.y = 2
        tgs.z = 4
        assert tgs.x == 128
        assert tgs.y == 2
        assert tgs.z == 4


class TestComputeHandle:
    """Test ComputeHandle from core module."""

    def test_handle_bool(self):
        """Test handle bool conversion."""
        data = np.array([1.0], dtype=np.float32)
        buf = cut.Buffer(data)
        assert buf.handle  # Should be truthy
        assert buf.handle.valid()


class TestBinaryVecOpsInt32:
    """Test binary vector-vector operations with int32 datatype."""

    def test_add_int32(self):
        """Test add operation with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        b_data = np.array([5, 6, 7, 8], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.add(a, b)
        expected = a_data + b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_subtract_int32(self):
        """Test subtract operation with int32."""
        a_data = np.array([10, 20, 30, 40], dtype=np.int32)
        b_data = np.array([1, 2, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.subtract(a, b)
        expected = a_data - b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply_int32(self):
        """Test multiply operation with int32."""
        a_data = np.array([2, 3, 4, 5], dtype=np.int32)
        b_data = np.array([3, 4, 5, 6], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.multiply(a, b)
        expected = a_data * b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_divide_int32(self):
        """Test divide operation with int32."""
        a_data = np.array([10, 20, 30, 40], dtype=np.int32)
        b_data = np.array([2, 4, 5, 8], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.divide(a, b)
        expected = (a_data // b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    @pytest.mark.xfail(reason="Int32 mod shader not yet implemented")
    def test_mod_int32(self):
        """Test mod operation with int32."""
        a_data = np.array([10, 21, 35, 47], dtype=np.int32)
        b_data = np.array([3, 4, 6, 9], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.mod(a, b)
        expected = a_data % b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    @pytest.mark.xfail(reason="Int32 floor_divide shader not yet implemented")
    def test_floor_divide_int32(self):
        """Test floor_divide operation with int32."""
        a_data = np.array([10, 21, 35, 47], dtype=np.int32)
        b_data = np.array([3, 4, 6, 9], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.floor_divide(a, b)
        expected = a_data // b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_minimum_int32(self):
        """Test minimum operation with int32."""
        a_data = np.array([1, 5, 3, 8], dtype=np.int32)
        b_data = np.array([2, 4, 6, 7], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.minimum(a, b)
        expected = np.minimum(a_data, b_data)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_maximum_int32(self):
        """Test maximum operation with int32."""
        a_data = np.array([1, 5, 3, 8], dtype=np.int32)
        b_data = np.array([2, 4, 6, 7], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.maximum(a, b)
        expected = np.maximum(a_data, b_data)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_int32(self):
        """Test equal comparison with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        b_data = np.array([1, 3, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.equal(a, b)
        expected = np.equal(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_int32(self):
        """Test not_equal comparison with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        b_data = np.array([1, 3, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.not_equal(a, b)
        expected = np.not_equal(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_int32(self):
        """Test less comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        b_data = np.array([2, 4, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.less(a, b)
        expected = np.less(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal_int32(self):
        """Test less_equal comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        b_data = np.array([2, 4, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.less_equal(a, b)
        expected = np.less_equal(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_int32(self):
        """Test greater comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        b_data = np.array([2, 4, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.greater(a, b)
        expected = np.greater(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal_int32(self):
        """Test greater_equal comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        b_data = np.array([2, 4, 3, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.greater_equal(a, b)
        expected = np.greater_equal(a_data, b_data).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_add_int32_negative(self):
        """Test add operation with negative int32 values."""
        a_data = np.array([-10, -5, 3, 8], dtype=np.int32)
        b_data = np.array([5, -3, -7, 2], dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.add(a, b)
        expected = a_data + b_data
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply_int32_large(self):
        """Test multiply operation with larger int32 arrays."""
        a_data = np.arange(100, dtype=np.int32)
        b_data = np.arange(100, 200, dtype=np.int32)
        a = cut.Buffer(a_data)
        b = cut.Buffer(b_data)
        result = cut.multiply(a, b)
        expected = a_data * b_data
        np.testing.assert_array_equal(result.numpy(), expected)


class TestBinaryScalarOpsInt32:
    """Test binary vector-scalar operations with int32 datatype."""

    def test_add_scalar_int32(self):
        """Test add_scalar operation with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 10
        result = cut.add_scalar(a, scalar)
        expected = a_data + scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_subtract_scalar_int32(self):
        """Test subtract_scalar operation with int32."""
        a_data = np.array([10, 20, 30, 40], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 5
        result = cut.subtract_scalar(a, scalar)
        expected = a_data - scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply_scalar_int32(self):
        """Test multiply_scalar operation with int32."""
        a_data = np.array([2, 3, 4, 5], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 3
        result = cut.multiply_scalar(a, scalar)
        expected = a_data * scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_divide_scalar_int32(self):
        """Test divide_scalar operation with int32."""
        a_data = np.array([10, 20, 30, 40], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 5
        result = cut.divide_scalar(a, scalar)
        expected = (a_data // scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    @pytest.mark.xfail(reason="Int32 mod_scalar shader not yet implemented")
    def test_mod_scalar_int32(self):
        """Test mod_scalar operation with int32."""
        a_data = np.array([10, 21, 35, 47], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 7
        result = cut.mod_scalar(a, scalar)
        expected = a_data % scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    @pytest.mark.xfail(reason="Int32 floor_divide_scalar shader not yet implemented")
    def test_floor_divide_scalar_int32(self):
        """Test floor_divide_scalar operation with int32."""
        a_data = np.array([10, 21, 35, 47], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 7
        result = cut.floor_divide_scalar(a, scalar)
        expected = a_data // scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_minimum_scalar_int32(self):
        """Test minimum_scalar operation with int32."""
        a_data = np.array([1, 5, 3, 8], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 4
        result = cut.minimum_scalar(a, scalar)
        expected = np.minimum(a_data, scalar)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_maximum_scalar_int32(self):
        """Test maximum_scalar operation with int32."""
        a_data = np.array([1, 5, 3, 8], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 4
        result = cut.maximum_scalar(a, scalar)
        expected = np.maximum(a_data, scalar)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_scalar_int32(self):
        """Test equal_scalar comparison with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 3
        result = cut.equal_scalar(a, scalar)
        expected = np.equal(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_scalar_int32(self):
        """Test not_equal_scalar comparison with int32."""
        a_data = np.array([1, 2, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 3
        result = cut.not_equal_scalar(a, scalar)
        expected = np.not_equal(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_scalar_int32(self):
        """Test less_scalar comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 4
        result = cut.less_scalar(a, scalar)
        expected = np.less(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal_scalar_int32(self):
        """Test less_equal_scalar comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 4
        result = cut.less_equal_scalar(a, scalar)
        expected = np.less_equal(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_scalar_int32(self):
        """Test greater_scalar comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 3
        result = cut.greater_scalar(a, scalar)
        expected = np.greater(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal_scalar_int32(self):
        """Test greater_equal_scalar comparison with int32."""
        a_data = np.array([1, 5, 3, 4], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 3
        result = cut.greater_equal_scalar(a, scalar)
        expected = np.greater_equal(a_data, scalar).astype(np.int32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_add_scalar_int32_negative(self):
        """Test add_scalar with negative scalar and int32."""
        a_data = np.array([10, 20, 30, 40], dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = -15
        result = cut.add_scalar(a, scalar)
        expected = a_data + scalar
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiply_scalar_int32_large(self):
        """Test multiply_scalar with larger int32 arrays."""
        a_data = np.arange(100, dtype=np.int32)
        a = cut.Buffer(a_data)
        scalar = 7
        result = cut.multiply_scalar(a, scalar)
        expected = a_data * scalar
        np.testing.assert_array_equal(result.numpy(), expected)
