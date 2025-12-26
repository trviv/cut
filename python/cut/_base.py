"""
Shared base classes for CUT backends.

This module provides common implementations that work across all backends
(Vulkan, CPU) to avoid code duplication.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Callable, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OperatorMixin(ABC):
    """
    Mixin class providing operator overloads for Buffer classes.

    This mixin expects the class to have:
    - _get_module(): Returns the module containing operation functions
    - _is_scalar(other): Returns True if other is a scalar value
    """

    def _is_scalar(self, other) -> bool:
        """Check if other is a scalar (int or float)."""
        return isinstance(other, (int, float)) and not isinstance(other, bool)

    @abstractmethod
    def _get_module(self):
        """Get the module containing operation functions. Must be implemented by subclass."""
        pass

    def _get_op(self, name: str):
        """Get an operation function from the module."""
        return getattr(self._get_module(), name)

    # =========================================================================
    # Arithmetic operators
    # =========================================================================

    def __add__(self, other) -> "OperatorMixin":
        """Add: self + other (element-wise for Buffer, scalar broadcast for numbers)."""
        if self._is_scalar(other):
            return self._get_op('add_scalar')(self, other)
        return self._get_op('add')(self, other)

    def __radd__(self, other) -> "OperatorMixin":
        """Right add: other + self."""
        if self._is_scalar(other):
            return self._get_op('add_scalar')(self, other)
        return self._get_op('add')(other, self)

    def __sub__(self, other) -> "OperatorMixin":
        """Subtract: self - other."""
        if self._is_scalar(other):
            return self._get_op('subtract_scalar')(self, other)
        return self._get_op('subtract')(self, other)

    def __rsub__(self, other) -> "OperatorMixin":
        """Right subtract: other - self."""
        if self._is_scalar(other):
            # other - self = -self + other
            neg_self = self._get_op('negative')(self)
            return self._get_op('add_scalar')(neg_self, other)
        return self._get_op('subtract')(other, self)

    def __mul__(self, other) -> "OperatorMixin":
        """Multiply: self * other."""
        if self._is_scalar(other):
            return self._get_op('multiply_scalar')(self, other)
        return self._get_op('multiply')(self, other)

    def __rmul__(self, other) -> "OperatorMixin":
        """Right multiply: other * self."""
        if self._is_scalar(other):
            return self._get_op('multiply_scalar')(self, other)
        return self._get_op('multiply')(other, self)

    def __truediv__(self, other) -> "OperatorMixin":
        """Divide: self / other."""
        if self._is_scalar(other):
            return self._get_op('divide_scalar')(self, other)
        return self._get_op('divide')(self, other)

    def __rtruediv__(self, other) -> "OperatorMixin":
        """Right divide: other / self."""
        if self._is_scalar(other):
            # other / self = other * (1/self)
            recip = self._get_op('reciprocal')(self)
            return self._get_op('multiply_scalar')(recip, other)
        return self._get_op('divide')(other, self)

    def __floordiv__(self, other) -> "OperatorMixin":
        """Floor divide: self // other."""
        if self._is_scalar(other):
            return self._get_op('floor_divide_scalar')(self, other)
        return self._get_op('floor_divide')(self, other)

    def __rfloordiv__(self, other) -> "OperatorMixin":
        """Right floor divide: other // self."""
        if self._is_scalar(other):
            # other // self = floor(other / self) = floor(other * reciprocal(self))
            recip = self._get_op('reciprocal')(self)
            scaled = self._get_op('multiply_scalar')(recip, other)
            return self._get_op('floor')(scaled)
        return self._get_op('floor_divide')(other, self)

    def __mod__(self, other) -> "OperatorMixin":
        """Modulo: self % other."""
        if self._is_scalar(other):
            return self._get_op('mod_scalar')(self, other)
        return self._get_op('mod')(self, other)

    def __rmod__(self, other) -> "OperatorMixin":
        """Right modulo: other % self."""
        raise NotImplementedError("scalar % Buffer not supported; use Buffer % scalar instead")

    def __pow__(self, other) -> "OperatorMixin":
        """Power: self ** other."""
        if self._is_scalar(other):
            return self._get_op('power_scalar')(self, other)
        return self._get_op('power')(self, other)

    def __rpow__(self, other) -> "OperatorMixin":
        """Right power: other ** self."""
        raise NotImplementedError("scalar ** Buffer not supported; use Buffer ** scalar instead")

    # =========================================================================
    # Unary operators
    # =========================================================================

    def __neg__(self) -> "OperatorMixin":
        """Negate: -self."""
        return self._get_op('negative')(self)

    def __abs__(self) -> "OperatorMixin":
        """Absolute value: abs(self)."""
        return self._get_op('abs')(self)

    def __pos__(self) -> "OperatorMixin":
        """Positive: +self (returns self)."""
        return self

    # =========================================================================
    # Comparison operators (return Buffer with 1.0 for True, 0.0 for False)
    # =========================================================================

    def __eq__(self, other) -> "OperatorMixin":
        """Equal: self == other."""
        if self._is_scalar(other):
            return self._get_op('equal_scalar')(self, other)
        return self._get_op('equal')(self, other)

    def __ne__(self, other) -> "OperatorMixin":
        """Not equal: self != other."""
        if self._is_scalar(other):
            return self._get_op('not_equal_scalar')(self, other)
        return self._get_op('not_equal')(self, other)

    def __lt__(self, other) -> "OperatorMixin":
        """Less than: self < other."""
        if self._is_scalar(other):
            return self._get_op('less_scalar')(self, other)
        return self._get_op('less')(self, other)

    def __le__(self, other) -> "OperatorMixin":
        """Less than or equal: self <= other."""
        if self._is_scalar(other):
            return self._get_op('less_equal_scalar')(self, other)
        return self._get_op('less_equal')(self, other)

    def __gt__(self, other) -> "OperatorMixin":
        """Greater than: self > other."""
        if self._is_scalar(other):
            return self._get_op('greater_scalar')(self, other)
        return self._get_op('greater')(self, other)

    def __ge__(self, other) -> "OperatorMixin":
        """Greater than or equal: self >= other."""
        if self._is_scalar(other):
            return self._get_op('greater_equal_scalar')(self, other)
        return self._get_op('greater_equal')(self, other)


# Type alias for numpy dtype to backend DataType mapping
DTypeMap = dict


def get_dtype_map(backend_module) -> DTypeMap:
    """
    Get the numpy dtype to backend DataType mapping.

    Args:
        backend_module: The backend module (_cut_core or _cut_cpu)

    Returns:
        Dictionary mapping numpy dtypes to backend DataType enum values
    """
    return {
        np.float32: backend_module.DataType.Float32,
        np.float16: backend_module.DataType.Float16,
        np.uint32: backend_module.DataType.UInt32,
        np.int32: backend_module.DataType.Int32,
    }


def create_buffer_from_data(
    interface,
    data: np.ndarray,
    is_uniform: bool = False
) -> tuple:
    """
    Create a buffer from numpy array data.

    Args:
        interface: The backend compute interface
        data: NumPy array to initialize buffer with
        is_uniform: If True, create a uniform buffer

    Returns:
        Tuple of (handle, size, dtype, shape)
    """
    data = np.ascontiguousarray(data)
    handle = interface.create_buffer(data, is_uniform)
    return handle, data.nbytes, data.dtype, data.shape


def create_buffer_empty(
    interface,
    backend_module,
    size: int,
    dtype: Optional[np.dtype] = None,
    shape: Optional[tuple] = None,
    is_uniform: bool = False
) -> tuple:
    """
    Create an empty buffer of given size.

    Args:
        interface: The backend compute interface
        backend_module: The backend module for DataType enum
        size: Buffer size in bytes
        dtype: Data type for the buffer (default: float32)
        shape: Shape for the buffer
        is_uniform: If True, create a uniform buffer

    Returns:
        Tuple of (handle, size, dtype, shape)
    """
    if dtype is None:
        dtype = np.float32

    element_size = np.dtype(dtype).itemsize
    num_elements = size // element_size
    buffer_shape = list(shape) if shape is not None else [num_elements]

    dtype_map = get_dtype_map(backend_module)
    cut_dtype = dtype_map.get(np.dtype(dtype).type, backend_module.DataType.Float32)

    handle = interface.create_buffer_empty(buffer_shape, cut_dtype, is_uniform)
    return handle, size, dtype, tuple(buffer_shape)


class BaseBuffer(OperatorMixin):
    """
    Base buffer class with common functionality for all backends.

    Subclasses must implement:
    - _get_module(): Return the module containing operation functions
    - _get_interface(): Return the compute interface
    """

    _handle: Any
    _size: int
    _dtype: Optional[np.dtype]
    _shape: Optional[tuple]

    def _init_from_data(self, interface, data: np.ndarray, is_uniform: bool = False):
        """Initialize buffer from numpy array data."""
        self._handle, self._size, self._dtype, self._shape = create_buffer_from_data(
            interface, data, is_uniform
        )

    def _init_empty(self, interface, backend_module, size: int,
                    dtype: Optional[np.dtype] = None, shape: Optional[tuple] = None,
                    is_uniform: bool = False):
        """Initialize empty buffer of given size."""
        self._handle, self._size, self._dtype, self._shape = create_buffer_empty(
            interface, backend_module, size, dtype, shape, is_uniform
        )

    @property
    def handle(self):
        """Get the underlying compute handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._size

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Get the buffer's data type."""
        return self._dtype

    @property
    def shape(self) -> Optional[tuple]:
        """Get the buffer's shape."""
        return self._shape

    @abstractmethod
    def _get_interface(self):
        """Get the compute interface. Must be implemented by subclass."""
        pass

    def copy_from(self, data: np.ndarray):
        """Copy data from numpy array to buffer."""
        data = np.ascontiguousarray(data)
        self._get_interface().copy_to_buffer(self._handle, data)

    def copy_to(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Copy data from buffer to numpy array.

        Args:
            out: Output array (created if not provided)

        Returns:
            NumPy array with buffer contents
        """
        if out is None:
            if self._dtype is not None and self._shape is not None:
                out = np.empty(self._shape, dtype=self._dtype)
            else:
                out = np.empty(self._size, dtype=np.uint8)
        out = np.ascontiguousarray(out)
        self._get_interface().copy_from_buffer(self._handle, out)
        return out

    def numpy(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        return self.copy_to()


class BaseDispatch:
    """
    Base dispatch class with common functionality for all backends.
    """

    def __init__(self, dispatch_obj, thread_size_class):
        """
        Initialize dispatch wrapper.

        Args:
            dispatch_obj: The backend-specific dispatch object
            thread_size_class: The ThreadSize class from the backend module
        """
        self._dispatch = dispatch_obj
        self._thread_size_class = thread_size_class
        self._bindings = []

    def set_workgroup_size(self, thread_groups: tuple):
        """Set the workgroup size."""
        self._dispatch.set_workgroup_size(
            self._thread_size_class(thread_groups[0], thread_groups[1], thread_groups[2])
        )

    def bind(self, resource, binding: int, buffer_class=None) -> "BaseDispatch":
        """
        Bind a resource to a binding point.

        Args:
            resource: Buffer, numpy array, int (as uint32), or float (as float32)
            binding: Binding index
            buffer_class: The Buffer class to check isinstance against

        Returns:
            self for chaining
        """
        if buffer_class is not None and isinstance(resource, buffer_class):
            self._dispatch.bind_resource(resource.handle, binding)
        elif isinstance(resource, np.ndarray):
            data = np.ascontiguousarray(resource)
            self._dispatch.bind_data(data, binding)
            self._bindings.append(data)  # Keep reference alive
        elif isinstance(resource, int):
            self._dispatch.bind_uint(resource, binding)
        elif isinstance(resource, float):
            self._dispatch.bind_float(resource, binding)
        else:
            # Assume it's a buffer-like object with a handle
            self._dispatch.bind_resource(resource.handle, binding)
        return self

    @property
    def inner(self):
        """Get the underlying dispatch object."""
        return self._dispatch
