"""
Operation definitions for CUT compute interface.

This module contains the operation name to enum mappings and docstrings
used by the compute module.
"""

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

# List of all operation names for __all__ exports
ALL_OPERATION_NAMES = (
    list(BINARY_VEC_VEC_OPS.keys()) +
    list(BINARY_VEC_SCALAR_OPS.keys()) +
    list(UNARY_OPS.keys())
)
