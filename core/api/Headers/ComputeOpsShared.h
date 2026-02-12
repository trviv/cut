// Shared enum definitions for C++ and GLSL
// This file uses syntax compatible with both languages
//
// For C++: #define GLSL_SHARED_HEADER before including
// For GLSL: Include directly

#ifndef COMPUTE_OPS_SHARED
#define COMPUTE_OPS_SHARED

// Scalar data types
#define DTYPE_FLOAT 0
#define DTYPE_HALF 1
#define DTYPE_UINT 2
#define DTYPE_INT 3

// =============================================================================
// Binary vec-vec operations (0-29)
// =============================================================================

// Binary vec-vec arithmetic operations
#define OP_BINARY_VEC_VEC_ADD 0
#define OP_BINARY_VEC_VEC_SUB 1
#define OP_BINARY_VEC_VEC_MUL 2
#define OP_BINARY_VEC_VEC_DIV 3
#define OP_BINARY_VEC_VEC_MOD 4
#define OP_BINARY_VEC_VEC_POW 5
#define OP_BINARY_VEC_VEC_FLOOR_DIV 6

// Binary vec-vec comparison operations
#define OP_BINARY_VEC_VEC_EQUAL 7
#define OP_BINARY_VEC_VEC_NOT_EQUAL 8
#define OP_BINARY_VEC_VEC_LESS 9
#define OP_BINARY_VEC_VEC_LESS_EQUAL 10
#define OP_BINARY_VEC_VEC_GREATER 11
#define OP_BINARY_VEC_VEC_GREATER_EQUAL 12

// Binary vec-vec min/max operations
#define OP_BINARY_VEC_VEC_MIN 13
#define OP_BINARY_VEC_VEC_MAX 14

// Binary vec-vec bitwise operations
#define OP_BINARY_VEC_VEC_BITWISE_AND 15
#define OP_BINARY_VEC_VEC_BITWISE_OR 16
#define OP_BINARY_VEC_VEC_BITWISE_XOR 17
#define OP_BINARY_VEC_VEC_LEFT_SHIFT 18
#define OP_BINARY_VEC_VEC_RIGHT_SHIFT 19

// Binary vec-vec logical operations
#define OP_BINARY_VEC_VEC_LOGICAL_AND 20
#define OP_BINARY_VEC_VEC_LOGICAL_OR 21
#define OP_BINARY_VEC_VEC_LOGICAL_XOR 22

// Binary vec-vec math operations
#define OP_BINARY_VEC_VEC_ATAN2 23
#define OP_BINARY_VEC_VEC_HYPOT 24
#define OP_BINARY_VEC_VEC_COPYSIGN 25
#define OP_BINARY_VEC_VEC_FMOD 26

// =============================================================================
// Binary vec-scalar operations (30-59)
// =============================================================================

// Binary vec-scalar arithmetic operations
#define OP_BINARY_VEC_SCALAR_ADD 30
#define OP_BINARY_VEC_SCALAR_SUB 31
#define OP_BINARY_VEC_SCALAR_MUL 32
#define OP_BINARY_VEC_SCALAR_DIV 33
#define OP_BINARY_VEC_SCALAR_MOD 34
#define OP_BINARY_VEC_SCALAR_POW 35
#define OP_BINARY_VEC_SCALAR_FLOOR_DIV 36

// Binary vec-scalar comparison operations
#define OP_BINARY_VEC_SCALAR_EQUAL 37
#define OP_BINARY_VEC_SCALAR_NOT_EQUAL 38
#define OP_BINARY_VEC_SCALAR_LESS 39
#define OP_BINARY_VEC_SCALAR_LESS_EQUAL 40
#define OP_BINARY_VEC_SCALAR_GREATER 41
#define OP_BINARY_VEC_SCALAR_GREATER_EQUAL 42

// Binary vec-scalar min/max operations
#define OP_BINARY_VEC_SCALAR_MIN 43
#define OP_BINARY_VEC_SCALAR_MAX 44

// Binary vec-scalar bitwise operations
#define OP_BINARY_VEC_SCALAR_BITWISE_AND 45
#define OP_BINARY_VEC_SCALAR_BITWISE_OR 46
#define OP_BINARY_VEC_SCALAR_BITWISE_XOR 47
#define OP_BINARY_VEC_SCALAR_LEFT_SHIFT 48
#define OP_BINARY_VEC_SCALAR_RIGHT_SHIFT 49

// Binary vec-scalar logical operations
#define OP_BINARY_VEC_SCALAR_LOGICAL_AND 50
#define OP_BINARY_VEC_SCALAR_LOGICAL_OR 51
#define OP_BINARY_VEC_SCALAR_LOGICAL_XOR 52

// Binary vec-scalar math operations
#define OP_BINARY_VEC_SCALAR_ATAN2 53
#define OP_BINARY_VEC_SCALAR_HYPOT 54
#define OP_BINARY_VEC_SCALAR_COPYSIGN 55
#define OP_BINARY_VEC_SCALAR_FMOD 56

// Binary vec-scalar activation operations
#define OP_BINARY_VEC_SCALAR_LEAKY_RELU 57

// =============================================================================
// Unary operations (60-89)
// =============================================================================

// Unary basic operations
#define OP_UNARY_NEG 60
#define OP_UNARY_ABS 61
#define OP_UNARY_SQRT 62
#define OP_UNARY_SQUARE 63
#define OP_UNARY_RECIPROCAL 64
#define OP_UNARY_SIGN 65

// Unary exponential/logarithmic operations
#define OP_UNARY_EXP 66
#define OP_UNARY_EXP2 67
#define OP_UNARY_EXPM1 68
#define OP_UNARY_LOG 69
#define OP_UNARY_LOG2 70
#define OP_UNARY_LOG10 71
#define OP_UNARY_LOG1P 72

// Unary trigonometric operations
#define OP_UNARY_SIN 73
#define OP_UNARY_COS 74
#define OP_UNARY_TAN 75
#define OP_UNARY_ASIN 76
#define OP_UNARY_ACOS 77
#define OP_UNARY_ATAN 78

// Unary hyperbolic operations
#define OP_UNARY_SINH 79
#define OP_UNARY_COSH 80
#define OP_UNARY_TANH 81

// Unary rounding operations
#define OP_UNARY_FLOOR 82
#define OP_UNARY_CEIL 83
#define OP_UNARY_ROUND 84

// Unary special math operations
#define OP_UNARY_CBRT 85
#define OP_UNARY_DEGREES 86
#define OP_UNARY_RADIANS 87

// Unary logical/bitwise operations
#define OP_UNARY_LOGICAL_NOT 88
#define OP_UNARY_BITWISE_NOT 89

// Unary activation functions
#define OP_UNARY_RELU 90
#define OP_UNARY_SIGMOID 91
#define OP_UNARY_GELU 92
#define OP_UNARY_SILU 93
#define OP_UNARY_SOFTPLUS 94

// Unary check operations
#define OP_UNARY_ISNAN 95
#define OP_UNARY_ISINF 96

// =============================================================================
// Ternary operations (100-109)
// =============================================================================

#define OP_TERNARY_CLAMP 100
#define OP_TERNARY_SELECT 101

// =============================================================================
// Reduction operations (110-119)
// =============================================================================

#define OP_REDUCE_SUM 110
#define OP_REDUCE_MEAN 111
#define OP_REDUCE_MIN 112
#define OP_REDUCE_MAX 113
#define OP_REDUCE_PROD 114
#define OP_REDUCE_ANY 115
#define OP_REDUCE_ALL 116
#define OP_REDUCE_DIM_SUM 117
#define OP_REDUCE_DIM_MEAN 118
#define OP_REDUCE_DIM_MIN 119
#define OP_REDUCE_DIM_MAX 123
#define OP_REDUCE_DIM_PROD 124
#define OP_REDUCE_DIM_ANY 125
#define OP_REDUCE_DIM_ALL 126

// =============================================================================
// Matrix operations (120-122)
// =============================================================================

#define OP_MATMUL 120
#define OP_TRANSPOSE 121
#define OP_DOT 122

// =============================================================================
// Tensor manipulation operations (130-139)
// =============================================================================

#define OP_CONCAT 130
#define OP_STACK 131
#define OP_FLATTEN 132

// =============================================================================
// Norm operations (140-149)
// =============================================================================

#define OP_NORM 140

// =============================================================================
// Tensor creation operations (150-159)
// =============================================================================

#define OP_ARANGE 150
#define OP_LINSPACE 151
#define OP_ZEROS 152
#define OP_ONES 153
#define OP_FULL 154

#endif // COMPUTE_OPS_SHARED
