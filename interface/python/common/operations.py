"""
Operation definitions for benchmarks and tests.

This module defines the operations tested across all backends.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any

from .test_data import TestData


# Operation definition type: (name, numpy_function, numpy_args_from_data)
OperationDef = Tuple[str, Callable, Callable[[TestData], tuple]]


# Binary arithmetic operations
BINARY_ARITHMETIC_OPS: List[Tuple[str, Callable, str, str]] = [
    ("add", np.add, "a", "b"),
    ("subtract", np.subtract, "a", "b"),
    ("multiply", np.multiply, "a", "b"),
    ("divide", np.divide, "a", "b_pos"),
    ("mod", np.mod, "a_pos", "b_pos"),
    ("power", np.power, "a_pos", "b_small"),
    ("floor_divide", np.floor_divide, "a", "b_pos"),
]

# Binary comparison operations
COMPARISON_OPS: List[Tuple[str, str, str]] = [
    ("equal", "a", "b"),
    ("not_equal", "a", "b"),
    ("less", "a", "b"),
    ("less_equal", "a", "b"),
    ("greater", "a", "b"),
    ("greater_equal", "a", "b"),
]

# Binary min/max operations
MINMAX_OPS: List[Tuple[str, Callable, str, str]] = [
    ("minimum", np.minimum, "a", "b"),
    ("maximum", np.maximum, "a", "b"),
]

# Unary operations: (name, numpy_function, input_array_name)
UNARY_OPS: List[Tuple[str, Callable, str]] = [
    ("negative", np.negative, "a"),
    ("abs", np.abs, "a"),
    ("sqrt", np.sqrt, "a_pos"),
    ("exp", np.exp, "a_div10"),
    ("log", np.log, "a_pos"),
    ("log2", np.log2, "a_pos"),
    ("log10", np.log10, "a_pos"),
    ("sin", np.sin, "a"),
    ("cos", np.cos, "a"),
    ("tan", np.tan, "a_tan_safe"),
    ("arcsin", np.arcsin, "a_unit"),
    ("arccos", np.arccos, "a_unit"),
    ("arctan", np.arctan, "a"),
    ("sinh", np.sinh, "a_div10"),
    ("cosh", np.cosh, "a_div10"),
    ("tanh", np.tanh, "a"),
    ("floor", np.floor, "a"),
    ("ceil", np.ceil, "a"),
    ("round", np.round, "a"),
    ("sign", np.sign, "a"),
    ("reciprocal", np.reciprocal, "a_pos"),
    ("square", np.square, "a"),
]

# Vec-scalar arithmetic operations
VEC_SCALAR_ARITHMETIC_OPS: List[Tuple[str, str, float]] = [
    ("add_scalar", "a", 2.5),
    ("subtract_scalar", "a", 2.5),
    ("multiply_scalar", "a", 2.5),
    ("divide_scalar", "a", 2.5),
    ("mod_scalar", "a_pos", 2.5),
    ("power_scalar", "a_pos", 2.0),
    ("floor_divide_scalar", "a", 2.5),
]

# Vec-scalar comparison operations
VEC_SCALAR_COMPARISON_OPS: List[Tuple[str, str, float]] = [
    ("equal_scalar", "a", 0.0),
    ("not_equal_scalar", "a", 0.0),
    ("less_scalar", "a", 0.0),
    ("less_equal_scalar", "a", 0.0),
    ("greater_scalar", "a", 0.0),
    ("greater_equal_scalar", "a", 0.0),
]

# Vec-scalar min/max operations
VEC_SCALAR_MINMAX_OPS: List[Tuple[str, str, float]] = [
    ("minimum_scalar", "a", 0.5),
    ("maximum_scalar", "a", -0.5),
]

# New PyTorch-like operations
TENSOR_CREATION_OPS: List[Tuple[str, int]] = [
    ("arange", 1000),  # (name, num_elements)
    ("linspace", 1000),
    ("zeros", 10000),
    ("ones", 10000),
    ("full", 10000),
]

# Conditional selection operations
CONDITIONAL_OPS: List[Tuple[str, str, str, str]] = [
    ("where", "a_pos", "a", "b"),  # (name, condition_array, x_array, y_array)
]

# Norm operations
NORM_OPS: List[Tuple[str, str, int]] = [
    ("norm_l2", "a", 2),  # (name, input_array, p)
    ("norm_l1", "a", 1),
]


def _make_comparison_func(op_name: str) -> Callable:
    """Create a comparison function that returns float32."""
    np_func = getattr(np, op_name)
    return lambda x, y: np_func(x, y).astype(np.float32)


def get_operations(data: TestData) -> Dict[str, List[Tuple[str, Callable, tuple]]]:
    """
    Get all benchmark operations organized by category.

    Args:
        data: TestData instance with generated arrays

    Returns:
        Dictionary mapping category names to lists of (name, np_func, args) tuples
    """
    operations = {}

    # Binary Arithmetic
    operations["Binary Arithmetic"] = [
        (name, np_func, (getattr(data, arg1), getattr(data, arg2)))
        for name, np_func, arg1, arg2 in BINARY_ARITHMETIC_OPS
    ]

    # Binary Comparison
    operations["Binary Comparison"] = [
        (name, _make_comparison_func(name), (getattr(data, arg1), getattr(data, arg2)))
        for name, arg1, arg2 in COMPARISON_OPS
    ]

    # Binary Min/Max
    operations["Binary Min/Max"] = [
        (name, np_func, (getattr(data, arg1), getattr(data, arg2)))
        for name, np_func, arg1, arg2 in MINMAX_OPS
    ]

    # Unary Operations
    operations["Unary Operations"] = [
        (name, np_func, (getattr(data, arr_name),))
        for name, np_func, arr_name in UNARY_OPS
    ]

    # Vec-Scalar Arithmetic
    operations["Vec-Scalar Arithmetic"] = [
        (name, lambda x, s, f=getattr(np, name.replace('_scalar', '')): f(x, s),
         (getattr(data, arr_name), scalar))
        for name, arr_name, scalar in VEC_SCALAR_ARITHMETIC_OPS
    ]

    # Vec-Scalar Comparison
    operations["Vec-Scalar Comparison"] = [
        (name, lambda x, s, n=name.replace('_scalar', ''): getattr(np, n)(x, s).astype(np.float32),
         (getattr(data, arr_name), scalar))
        for name, arr_name, scalar in VEC_SCALAR_COMPARISON_OPS
    ]

    # Vec-Scalar Min/Max
    operations["Vec-Scalar Min/Max"] = [
        (name, lambda x, s, f=getattr(np, name.replace('_scalar', '')): f(x, s),
         (getattr(data, arr_name), scalar))
        for name, arr_name, scalar in VEC_SCALAR_MINMAX_OPS
    ]

    # Tensor Creation Operations
    operations["Tensor Creation"] = []
    for name, size in TENSOR_CREATION_OPS:
        if name == "arange":
            operations["Tensor Creation"].append(
                (name, np.arange, (0, size, 1))
            )
        elif name == "linspace":
            operations["Tensor Creation"].append(
                (name, np.linspace, (0, size-1, size))
            )
        elif name == "zeros":
            operations["Tensor Creation"].append(
                (name, np.zeros, (size,))
            )
        elif name == "ones":
            operations["Tensor Creation"].append(
                (name, np.ones, (size,))
            )
        elif name == "full":
            operations["Tensor Creation"].append(
                (name, lambda size, val=3.14: np.full(size, val), (size,))
            )

    # Conditional Selection
    operations["Conditional Selection"] = [
        (name, np.where, (getattr(data, cond_arr), getattr(data, x_arr), getattr(data, y_arr)))
        for name, cond_arr, x_arr, y_arr in CONDITIONAL_OPS
    ]

    # Norm Operations
    operations["Norm Operations"] = []
    for name, arr_name, p in NORM_OPS:
        if name == "norm_l2":
            operations["Norm Operations"].append(
                ("norm", lambda x: np.array([np.linalg.norm(x, ord=2)], dtype=np.float32), (getattr(data, arr_name),))
            )
        elif name == "norm_l1":
            operations["Norm Operations"].append(
                ("norm", lambda x: np.array([np.linalg.norm(x, ord=1)], dtype=np.float32), (getattr(data, arr_name),))
            )

    return operations


def get_simple_operations(data: TestData) -> Dict[str, List[Tuple[str, Callable, Callable, tuple]]]:
    """
    Get simplified operations for the basic benchmark.py file.

    This version returns (name, cut_func, np_func, args) tuples that work
    with the simpler benchmark.py structure.

    Args:
        data: TestData instance with generated arrays

    Returns:
        Dictionary mapping category names to operation lists
    """
    operations = {}

    # Binary Arithmetic
    operations["Binary Arithmetic"] = [
        (name, np_func, getattr(data, arg1), getattr(data, arg2))
        for name, np_func, arg1, arg2 in BINARY_ARITHMETIC_OPS
    ]

    # Binary Comparison (with float32 casting for numpy)
    operations["Binary Comparison"] = [
        (name, _make_comparison_func(name), getattr(data, arg1), getattr(data, arg2))
        for name, arg1, arg2 in COMPARISON_OPS
    ]

    # Binary Min/Max
    operations["Binary Min/Max"] = [
        (name, np_func, getattr(data, arg1), getattr(data, arg2))
        for name, np_func, arg1, arg2 in MINMAX_OPS
    ]

    # Unary Operations
    operations["Unary Operations"] = [
        (name, np_func, getattr(data, arr_name))
        for name, np_func, arr_name in UNARY_OPS
    ]

    return operations
