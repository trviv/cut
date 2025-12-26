"""
Shared operation definitions for CUT backends.

This module provides operation function generators that work with any backend
(Vulkan or CPU) by accepting the backend-specific components as parameters.
"""

from typing import Optional, Callable, Any
import numpy as np


# Operation name to enum suffix mapping
BINARY_VEC_VEC_OPS = {
    # Arithmetic
    "add": "BinaryVecVecAdd",
    "subtract": "BinaryVecVecSub",
    "multiply": "BinaryVecVecMul",
    "divide": "BinaryVecVecDiv",
    "mod": "BinaryVecVecMod",
    "power": "BinaryVecVecPow",
    "floor_divide": "BinaryVecVecFloorDiv",
    # Comparison
    "equal": "BinaryVecVecEqual",
    "not_equal": "BinaryVecVecNotEqual",
    "less": "BinaryVecVecLess",
    "less_equal": "BinaryVecVecLessEqual",
    "greater": "BinaryVecVecGreater",
    "greater_equal": "BinaryVecVecGreaterEqual",
    # Min/Max
    "minimum": "BinaryVecVecMin",
    "maximum": "BinaryVecVecMax",
}

BINARY_VEC_SCALAR_OPS = {
    # Arithmetic
    "add_scalar": "BinaryVecScalarAdd",
    "subtract_scalar": "BinaryVecScalarSub",
    "multiply_scalar": "BinaryVecScalarMul",
    "divide_scalar": "BinaryVecScalarDiv",
    "mod_scalar": "BinaryVecScalarMod",
    "power_scalar": "BinaryVecScalarPow",
    "floor_divide_scalar": "BinaryVecScalarFloorDiv",
    # Comparison
    "equal_scalar": "BinaryVecScalarEqual",
    "not_equal_scalar": "BinaryVecScalarNotEqual",
    "less_scalar": "BinaryVecScalarLess",
    "less_equal_scalar": "BinaryVecScalarLessEqual",
    "greater_scalar": "BinaryVecScalarGreater",
    "greater_equal_scalar": "BinaryVecScalarGreaterEqual",
    # Min/Max
    "minimum_scalar": "BinaryVecScalarMin",
    "maximum_scalar": "BinaryVecScalarMax",
}

UNARY_OPS = {
    "negative": "UnaryNeg",
    "abs": "UnaryAbs",
    "sqrt": "UnarySqrt",
    "exp": "UnaryExp",
    "log": "UnaryLog",
    "log2": "UnaryLog2",
    "log10": "UnaryLog10",
    "sin": "UnarySin",
    "cos": "UnaryCos",
    "tan": "UnaryTan",
    "arcsin": "UnaryAsin",
    "arccos": "UnaryAcos",
    "arctan": "UnaryAtan",
    "sinh": "UnarySinh",
    "cosh": "UnaryCosh",
    "tanh": "UnaryTanh",
    "floor": "UnaryFloor",
    "ceil": "UnaryCeil",
    "round": "UnaryRound",
    "sign": "UnarySign",
    "reciprocal": "UnaryReciprocal",
    "square": "UnarySquare",
}

# Docstrings for operations
BINARY_VEC_VEC_DOCS = {
    "add": "Add two buffers element-wise.",
    "subtract": "Subtract two buffers element-wise.",
    "multiply": "Multiply two buffers element-wise.",
    "divide": "Divide two buffers element-wise.",
    "mod": "Modulo of two buffers element-wise.",
    "power": "Power of two buffers element-wise.",
    "floor_divide": "Floor division of two buffers element-wise.",
    "equal": "Element-wise equality comparison. Returns 1.0 for True, 0.0 for False.",
    "not_equal": "Element-wise inequality comparison. Returns 1.0 for True, 0.0 for False.",
    "less": "Element-wise less-than comparison. Returns 1.0 for True, 0.0 for False.",
    "less_equal": "Element-wise less-than-or-equal comparison. Returns 1.0 for True, 0.0 for False.",
    "greater": "Element-wise greater-than comparison. Returns 1.0 for True, 0.0 for False.",
    "greater_equal": "Element-wise greater-than-or-equal comparison. Returns 1.0 for True, 0.0 for False.",
    "minimum": "Element-wise minimum of two buffers.",
    "maximum": "Element-wise maximum of two buffers.",
}

BINARY_VEC_SCALAR_DOCS = {
    "add_scalar": "Add a scalar to each element of a buffer.",
    "subtract_scalar": "Subtract a scalar from each element of a buffer.",
    "multiply_scalar": "Multiply each element of a buffer by a scalar.",
    "divide_scalar": "Divide each element of a buffer by a scalar.",
    "mod_scalar": "Modulo of each element of a buffer by a scalar.",
    "power_scalar": "Raise each element of a buffer to a scalar power.",
    "floor_divide_scalar": "Floor division of each element of a buffer by a scalar.",
    "equal_scalar": "Element-wise equality comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "not_equal_scalar": "Element-wise inequality comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "less_scalar": "Element-wise less-than comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "less_equal_scalar": "Element-wise less-than-or-equal comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "greater_scalar": "Element-wise greater-than comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "greater_equal_scalar": "Element-wise greater-than-or-equal comparison with a scalar. Returns 1.0 for True, 0.0 for False.",
    "minimum_scalar": "Element-wise minimum of buffer elements and a scalar.",
    "maximum_scalar": "Element-wise maximum of buffer elements and a scalar.",
}

UNARY_DOCS = {
    "negative": "Negate buffer element-wise.",
    "abs": "Absolute value element-wise.",
    "sqrt": "Square root element-wise.",
    "exp": "Exponential element-wise.",
    "log": "Natural logarithm element-wise.",
    "log2": "Base-2 logarithm element-wise.",
    "log10": "Base-10 logarithm element-wise.",
    "sin": "Sine element-wise.",
    "cos": "Cosine element-wise.",
    "tan": "Tangent element-wise.",
    "arcsin": "Inverse sine element-wise.",
    "arccos": "Inverse cosine element-wise.",
    "arctan": "Inverse tangent element-wise.",
    "sinh": "Hyperbolic sine element-wise.",
    "cosh": "Hyperbolic cosine element-wise.",
    "tanh": "Hyperbolic tangent element-wise.",
    "floor": "Floor element-wise.",
    "ceil": "Ceiling element-wise.",
    "round": "Round element-wise.",
    "sign": "Sign element-wise (-1, 0, or 1).",
    "reciprocal": "Reciprocal (1/x) element-wise.",
    "square": "Square (x*x) element-wise.",
}


def create_binary_op(
    op_enum,
    buffer_class,
    shader_or_kernel_class,
    dispatch_class,
    run_func: Callable,
    ensure_init: Callable,
):
    """
    Create a binary vec-vec operation function.

    Args:
        op_enum: The operator enum value (e.g., ShaderEnum.BinaryVecVecAdd)
        buffer_class: The Buffer class for this backend
        shader_or_kernel_class: The Shader/Kernel class for this backend
        dispatch_class: The Dispatch class for this backend
        run_func: The run function for this backend
        ensure_init: The initialization function for this backend
    """
    def binary_op(a, b, out=None):
        ensure_init()

        if a.size != b.size:
            raise ValueError(f"Size mismatch: {a.size} vs {b.size}")

        if out is None:
            out = buffer_class(size=a.size, dtype=a._dtype, shape=a._shape)

        # Calculate num_elements based on dtype itemsize
        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader_or_kernel = shader_or_kernel_class(op_enum, dtype=a._dtype)
        dispatch = dispatch_class(shader_or_kernel, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(b, 1)
        dispatch.bind(out, 2)
        dispatch.bind(num_elements, 3)

        run_func(dispatch)

        return out

    return binary_op


def create_unary_op(
    op_enum,
    buffer_class,
    shader_or_kernel_class,
    dispatch_class,
    run_func: Callable,
    ensure_init: Callable,
):
    """
    Create a unary operation function.

    Args:
        op_enum: The operator enum value (e.g., ShaderEnum.UnaryNeg)
        buffer_class: The Buffer class for this backend
        shader_or_kernel_class: The Shader/Kernel class for this backend
        dispatch_class: The Dispatch class for this backend
        run_func: The run function for this backend
        ensure_init: The initialization function for this backend
    """
    def unary_op(a, out=None):
        ensure_init()

        if out is None:
            out = buffer_class(size=a.size, dtype=a._dtype, shape=a._shape)

        # Calculate num_elements based on dtype itemsize
        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader_or_kernel = shader_or_kernel_class(op_enum, dtype=a._dtype)
        dispatch = dispatch_class(shader_or_kernel, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(out, 1)
        dispatch.bind(num_elements, 2)

        run_func(dispatch)

        return out

    return unary_op


def create_binary_vec_scalar_op(
    op_enum,
    buffer_class,
    shader_or_kernel_class,
    dispatch_class,
    run_func: Callable,
    ensure_init: Callable,
):
    """
    Create a binary vec-scalar operation function.

    Args:
        op_enum: The operator enum value (e.g., ShaderEnum.BinaryVecScalarAdd)
        buffer_class: The Buffer class for this backend
        shader_or_kernel_class: The Shader/Kernel class for this backend
        dispatch_class: The Dispatch class for this backend
        run_func: The run function for this backend
        ensure_init: The initialization function for this backend
    """
    def vec_scalar_op(a, scalar, out=None):
        ensure_init()

        if out is None:
            out = buffer_class(size=a.size, dtype=a._dtype, shape=a._shape)

        # Calculate num_elements based on dtype itemsize
        itemsize = np.dtype(a._dtype).itemsize if a._dtype is not None else 4
        num_elements = a.size // itemsize

        shader_or_kernel = shader_or_kernel_class(op_enum, dtype=a._dtype)
        dispatch = dispatch_class(shader_or_kernel, (num_elements, 1, 1))
        dispatch.bind(a, 0)
        dispatch.bind(out, 1)
        # Pack push constants as numpy array: uint32 + scalar (type matches buffer dtype)
        dtype = a._dtype if a._dtype is not None else np.float32
        if dtype == np.int32:
            push_constants = np.array([num_elements, int(scalar)], dtype=np.int32)
        elif dtype == np.uint32:
            push_constants = np.array([num_elements, int(scalar)], dtype=np.uint32)
        else:
            push_constants = np.array([num_elements, 0], dtype=np.uint32)
            push_constants.view(np.float32)[1] = float(scalar)
        dispatch.bind(push_constants, 2)

        run_func(dispatch)

        return out

    return vec_scalar_op


def register_operations(
    module_dict: dict,
    enum_module: Any,
    buffer_class,
    shader_or_kernel_class,
    dispatch_class,
    run_func: Callable,
    ensure_init: Callable,
    backend_name: str = "",
):
    """
    Register all operations into a module's namespace.

    Args:
        module_dict: The module's __dict__ or globals() to add functions to
        enum_module: Module containing the operator enums (e.g., _cut_core or _cut_cpu)
        buffer_class: The Buffer class for this backend
        shader_or_kernel_class: The Shader/Kernel class for this backend
        dispatch_class: The Dispatch class for this backend
        run_func: The run function for this backend
        ensure_init: The initialization function for this backend
        backend_name: Optional backend name for docstrings (e.g., "on GPU", "on CPU")
    """
    # Use ShaderEnum for Vulkan, OperatorEnum for CPU
    if hasattr(enum_module, 'ShaderEnum'):
        op_enum = enum_module.ShaderEnum
    else:
        op_enum = enum_module.OperatorEnum

    suffix = f" {backend_name}" if backend_name else ""

    # Register binary vec-vec operations
    for op_name, enum_name in BINARY_VEC_VEC_OPS.items():
        enum_value = getattr(op_enum, enum_name)
        func = create_binary_op(
            enum_value, buffer_class, shader_or_kernel_class,
            dispatch_class, run_func, ensure_init
        )
        func.__name__ = op_name
        func.__doc__ = BINARY_VEC_VEC_DOCS.get(op_name, "") + suffix
        module_dict[op_name] = func

    # Register binary vec-scalar operations
    for op_name, enum_name in BINARY_VEC_SCALAR_OPS.items():
        enum_value = getattr(op_enum, enum_name)
        func = create_binary_vec_scalar_op(
            enum_value, buffer_class, shader_or_kernel_class,
            dispatch_class, run_func, ensure_init
        )
        func.__name__ = op_name
        func.__doc__ = BINARY_VEC_SCALAR_DOCS.get(op_name, "") + suffix
        module_dict[op_name] = func

    # Register unary operations
    for op_name, enum_name in UNARY_OPS.items():
        enum_value = getattr(op_enum, enum_name)
        func = create_unary_op(
            enum_value, buffer_class, shader_or_kernel_class,
            dispatch_class, run_func, ensure_init
        )
        func.__name__ = op_name
        func.__doc__ = UNARY_DOCS.get(op_name, "") + suffix
        module_dict[op_name] = func


# List of all operation names for __all__ exports
ALL_OPERATION_NAMES = (
    list(BINARY_VEC_VEC_OPS.keys()) +
    list(BINARY_VEC_SCALAR_OPS.keys()) +
    list(UNARY_OPS.keys())
)
