"""Tests for the CUT Python bindings."""

import numpy as np
import pytest
import cut


class TestImport:
    """Test basic module import and initialization."""

    def test_import(self):
        """Test that the module can be imported."""
        assert cut.__version__ == "0.1.0"

    def test_shader_enum(self):
        """Test ShaderEnum is accessible."""
        assert hasattr(cut, "ShaderEnum")
        assert hasattr(cut.ShaderEnum, "BinaryVecVecAdd")

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
        spirv = _cut_core.get_shader(cut.ShaderEnum.BinaryVecVecAdd)
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
