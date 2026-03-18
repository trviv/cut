"""
Operation definitions for benchmarks and tests.

This module defines the operations tested across all backends.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional

from .test_data import TestData


# Operation definition type: (name, numpy_function, numpy_args_from_data)
OperationDef = Tuple[str, Callable, Callable[[TestData], tuple]]

# Try importing scipy for special functions; provide fallbacks
try:
    from scipy import special as _sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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

# Extended binary operations
BINARY_MATH_OPS: List[Tuple[str, Callable, str, str]] = [
    ("arctan2", np.arctan2, "a", "b"),
    ("hypot", np.hypot, "a", "b"),
    ("copysign", np.copysign, "a", "b"),
    ("fmod", np.fmod, "a", "b_pos"),
    ("logaddexp", np.logaddexp, "a_div10", "a_div10"),
    ("logaddexp2", np.logaddexp2, "a_div10", "a_div10"),
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

# Extended unary math operations
UNARY_MATH_OPS: List[Tuple[str, Callable, str]] = [
    ("expm1", np.expm1, "a_div10"),
    ("log1p", np.log1p, "a_pos"),
    ("cbrt", np.cbrt, "a"),
    ("exp2", np.exp2, "a_div10"),
    ("degrees", np.degrees, "a"),
    ("radians", np.radians, "a"),
    ("rsqrt", lambda x: 1.0 / np.sqrt(x), "a_pos"),
    ("trunc", np.trunc, "a"),
    ("frac", lambda x: x - np.trunc(x), "a"),
    ("arcsinh", np.arcsinh, "a"),
    ("arccosh", np.arccosh, "a_pos"),
    ("arctanh", np.arctanh, "a_unit"),
]

# Activation functions (unary)
ACTIVATION_OPS: List[Tuple[str, Callable, str]] = [
    ("relu", lambda x: np.maximum(x, 0), "a"),
    ("relu6", lambda x: np.clip(x, 0, 6), "a"),
    ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x)), "a"),
    ("gelu", lambda x: x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))), "a"),
    ("silu", lambda x: x / (1.0 + np.exp(-x)), "a"),
    ("softplus", lambda x: np.log1p(np.exp(np.clip(x, -20, 20))), "a"),
    ("elu", lambda x: np.where(x >= 0, x, np.exp(x) - 1), "a"),
    ("mish", lambda x: x * np.tanh(np.log1p(np.exp(np.clip(x, -20, 20)))), "a"),
    ("hardswish", lambda x: x * np.clip(x + 3, 0, 6) / 6.0, "a"),
    ("hardsigmoid", lambda x: np.clip(x / 6.0 + 0.5, 0, 1), "a"),
    ("hardtanh", lambda x: np.clip(x, -1, 1), "a"),
    ("softsign", lambda x: x / (1.0 + np.abs(x)), "a"),
    ("logsigmoid", lambda x: -np.log1p(np.exp(-x)), "a"),
    ("tanhshrink", lambda x: x - np.tanh(x), "a"),
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


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """NumPy softmax along last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    """NumPy log_softmax along last axis."""
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m - np.log(np.sum(e, axis=-1, keepdims=True))).astype(np.float32)


def _layer_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """NumPy layer normalization over last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)).astype(np.float32)


def _rms_norm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """NumPy RMS normalization over last axis."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms).astype(np.float32)


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

    # Binary Math (arctan2, hypot, etc.)
    operations["Binary Math"] = [
        (name, np_func, (getattr(data, arg1), getattr(data, arg2)))
        for name, np_func, arg1, arg2 in BINARY_MATH_OPS
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

    # Extended Unary Math
    operations["Unary Math (Extended)"] = [
        (name, np_func, (getattr(data, arr_name),))
        for name, np_func, arr_name in UNARY_MATH_OPS
    ]

    # Activation Functions
    operations["Activation Functions"] = [
        (name, np_func, (getattr(data, arr_name),))
        for name, np_func, arr_name in ACTIVATION_OPS
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

    # Reduction Operations
    operations["Reduction Operations"] = [
        ("sum", lambda x: np.array([np.sum(x)], dtype=np.float32), (data.a,)),
        ("mean", lambda x: np.array([np.mean(x)], dtype=np.float32), (data.a,)),
        ("min", lambda x: np.array([np.min(x)], dtype=np.float32), (data.a,)),
        ("max", lambda x: np.array([np.max(x)], dtype=np.float32), (data.a,)),
        ("prod", lambda x: np.array([np.prod(x[:100])], dtype=np.float32), (data.a_pos[:100],)),
    ]

    # Statistical Operations
    operations["Statistical Operations"] = [
        ("var", lambda x: np.array([np.var(x, ddof=1)], dtype=np.float32), (data.a,)),
        ("std", lambda x: np.array([np.std(x, ddof=1)], dtype=np.float32), (data.a,)),
    ]

    # Cumulative Operations
    operations["Cumulative Operations"] = [
        ("cumsum", lambda x: np.cumsum(x).astype(np.float32), (data.a,)),
        ("cumprod", lambda x: np.cumprod(x[:100]).astype(np.float32), (data.a_pos[:100],)),
    ]

    # Matrix Operations (use mat_a, mat_b)
    if data.mat_a is not None and data.mat_b is not None:
        operations["Matrix Operations"] = [
            ("matmul", lambda a, b: (a @ b).astype(np.float32), (data.mat_a, data.mat_b)),
            ("transpose", lambda a: a.T.astype(np.float32), (data.mat_a,)),
        ]

    # Normalization Operations (use mat_2d)
    if data.mat_2d is not None:
        operations["Normalization"] = [
            ("softmax", _softmax_np, (data.mat_2d,)),
            ("log_softmax", _log_softmax_np, (data.mat_2d,)),
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

    # Loss Functions
    operations["Loss Functions"] = [
        ("mse_loss", lambda x, y: np.array([np.mean((x - y) ** 2)], dtype=np.float32), (data.a, data.b)),
        ("l1_loss", lambda x, y: np.array([np.mean(np.abs(x - y))], dtype=np.float32), (data.a, data.b)),
    ]

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
