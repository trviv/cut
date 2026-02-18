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
// Binary vec-vec operations (0-28)
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
#define OP_BINARY_VEC_VEC_LOGADDEXP 27
#define OP_BINARY_VEC_VEC_LOGADDEXP2 28

// =============================================================================
// Binary vec-scalar operations (29-61)
// =============================================================================

// Binary vec-scalar arithmetic operations
#define OP_BINARY_VEC_SCALAR_ADD 29
#define OP_BINARY_VEC_SCALAR_SUB 30
#define OP_BINARY_VEC_SCALAR_MUL 31
#define OP_BINARY_VEC_SCALAR_DIV 32
#define OP_BINARY_VEC_SCALAR_MOD 33
#define OP_BINARY_VEC_SCALAR_POW 34
#define OP_BINARY_VEC_SCALAR_FLOOR_DIV 35

// Binary vec-scalar comparison operations
#define OP_BINARY_VEC_SCALAR_EQUAL 36
#define OP_BINARY_VEC_SCALAR_NOT_EQUAL 37
#define OP_BINARY_VEC_SCALAR_LESS 38
#define OP_BINARY_VEC_SCALAR_LESS_EQUAL 39
#define OP_BINARY_VEC_SCALAR_GREATER 40
#define OP_BINARY_VEC_SCALAR_GREATER_EQUAL 41

// Binary vec-scalar min/max operations
#define OP_BINARY_VEC_SCALAR_MIN 42
#define OP_BINARY_VEC_SCALAR_MAX 43

// Binary vec-scalar bitwise operations
#define OP_BINARY_VEC_SCALAR_BITWISE_AND 44
#define OP_BINARY_VEC_SCALAR_BITWISE_OR 45
#define OP_BINARY_VEC_SCALAR_BITWISE_XOR 46
#define OP_BINARY_VEC_SCALAR_LEFT_SHIFT 47
#define OP_BINARY_VEC_SCALAR_RIGHT_SHIFT 48

// Binary vec-scalar logical operations
#define OP_BINARY_VEC_SCALAR_LOGICAL_AND 49
#define OP_BINARY_VEC_SCALAR_LOGICAL_OR 50
#define OP_BINARY_VEC_SCALAR_LOGICAL_XOR 51

// Binary vec-scalar math operations
#define OP_BINARY_VEC_SCALAR_ATAN2 52
#define OP_BINARY_VEC_SCALAR_HYPOT 53
#define OP_BINARY_VEC_SCALAR_COPYSIGN 54
#define OP_BINARY_VEC_SCALAR_FMOD 55

// Binary vec-scalar activation operations
#define OP_BINARY_VEC_SCALAR_LEAKY_RELU 56
#define OP_BINARY_VEC_SCALAR_PRELU 57
#define OP_BINARY_VEC_SCALAR_HARDSHRINK 58
#define OP_BINARY_VEC_SCALAR_SOFTSHRINK 59
#define OP_BINARY_VEC_SCALAR_LOGADDEXP 60
#define OP_BINARY_VEC_SCALAR_LOGADDEXP2 61

// =============================================================================
// Ternary operations (62-63)
// =============================================================================

#define OP_TERNARY_CLAMP 62
#define OP_TERNARY_SELECT 63

// =============================================================================
// Matrix operations (64-66)
// =============================================================================

#define OP_MATMUL 64
#define OP_TRANSPOSE 65
#define OP_DOT 66

// =============================================================================
// Tensor manipulation operations (67-69)
// =============================================================================

#define OP_CONCAT 67
#define OP_STACK 68
#define OP_FLATTEN 69

// =============================================================================
// Norm operations (70-71)
// =============================================================================

#define OP_NORM 70
#define OP_NORM_DIM 71

// =============================================================================
// Tensor creation operations (72-77)
// =============================================================================

#define OP_ARANGE 72
#define OP_LINSPACE 73
#define OP_ZEROS 74
#define OP_ONES 75
#define OP_FULL 76
#define OP_COPY 77

// Matmul variants (for benchmarking different strategies)
#define OP_MATMUL_NAIVE 78
#define OP_MATMUL_REG_TILED 79
#define OP_MATMUL_TILED_2X2 80
#define OP_MATMUL_T8_R2X2 81
#define OP_MATMUL_T8_R4X4 82
#define OP_MATMUL_T16_R4X4 83
#define OP_MATMUL_T16_R8X8 84
#define OP_MATMUL_T32_R2X2 85
#define OP_MATMUL_SIMD_R4X4 86
#define OP_MATMUL_SIMD_R4X8 87
#define OP_MATMUL_SIMD_R8X8 88

// =============================================================================
// Unary operations (100-154)
// =============================================================================

// Unary basic operations
#define OP_UNARY_NEG 100
#define OP_UNARY_ABS 101
#define OP_UNARY_SQRT 102
#define OP_UNARY_SQUARE 103
#define OP_UNARY_RECIPROCAL 104
#define OP_UNARY_SIGN 105

// Unary exponential/logarithmic operations
#define OP_UNARY_EXP 106
#define OP_UNARY_EXP2 107
#define OP_UNARY_EXPM1 108
#define OP_UNARY_LOG 109
#define OP_UNARY_LOG2 110
#define OP_UNARY_LOG10 111
#define OP_UNARY_LOG1P 112

// Unary trigonometric operations
#define OP_UNARY_SIN 113
#define OP_UNARY_COS 114
#define OP_UNARY_TAN 115
#define OP_UNARY_ASIN 116
#define OP_UNARY_ACOS 117
#define OP_UNARY_ATAN 118

// Unary hyperbolic operations
#define OP_UNARY_SINH 119
#define OP_UNARY_COSH 120
#define OP_UNARY_TANH 121

// Unary rounding operations
#define OP_UNARY_FLOOR 122
#define OP_UNARY_CEIL 123
#define OP_UNARY_ROUND 124

// Unary special math operations
#define OP_UNARY_CBRT 125
#define OP_UNARY_DEGREES 126
#define OP_UNARY_RADIANS 127

// Unary logical/bitwise operations
#define OP_UNARY_LOGICAL_NOT 128
#define OP_UNARY_BITWISE_NOT 129

// Unary activation functions
#define OP_UNARY_RELU 130
#define OP_UNARY_SIGMOID 131
#define OP_UNARY_GELU 132
#define OP_UNARY_SILU 133
#define OP_UNARY_SOFTPLUS 134

// Unary check operations
#define OP_UNARY_ISNAN 135
#define OP_UNARY_ISINF 136

// Unary extended activation functions
#define OP_UNARY_RELU6 137
#define OP_UNARY_ELU 138
#define OP_UNARY_SELU 139
#define OP_UNARY_CELU 140
#define OP_UNARY_MISH 141
#define OP_UNARY_HARDSWISH 142
#define OP_UNARY_HARDSIGMOID 143
#define OP_UNARY_HARDTANH 144
#define OP_UNARY_SOFTSIGN 145
#define OP_UNARY_LOGSIGMOID 146
#define OP_UNARY_TANHSHRINK 147

// Unary extended math operations
#define OP_UNARY_RSQRT 148
#define OP_UNARY_TRUNC 149
#define OP_UNARY_FRAC 150
#define OP_UNARY_ASINH 151
#define OP_UNARY_ACOSH 152
#define OP_UNARY_ATANH 153
#define OP_UNARY_ISFINITE 154

// =============================================================================
// Reduction operations (200-215)
// =============================================================================

#define OP_REDUCE_SUM 200
#define OP_REDUCE_MEAN 201
#define OP_REDUCE_MIN 202
#define OP_REDUCE_MAX 203
#define OP_REDUCE_PROD 204
#define OP_REDUCE_ANY 205
#define OP_REDUCE_ALL 206
#define OP_REDUCE_ARGMAX 214
#define OP_REDUCE_ARGMIN 215

// =============================================================================
// Cumulative/scan operations (240-241)
// =============================================================================

#define OP_CUMSUM 240
#define OP_CUMPROD 241

// =============================================================================
// Prefix scan operations (260-261)
// =============================================================================

#define OP_PREFIX_SCAN_EXCLUSIVE_SUM 260
#define OP_PREFIX_SCAN_INCLUSIVE_SUM 261

// =============================================================================
// Sort operations (270-271)
// =============================================================================

#define OP_SORT_BITONIC 270
#define OP_SORT_RADIX 271

// =============================================================================
// Dispatcher internal shader templates (280-291)
// =============================================================================

// Multi-workgroup reduce
#define OP_INTERNAL_PARTIAL_REDUCE 280
#define OP_INTERNAL_FINAL_REDUCE 281

// Prefix scan (three-pass)
#define OP_INTERNAL_SCAN_PER_WG 282
#define OP_INTERNAL_SCAN_PARTIAL_SUMS 283
#define OP_INTERNAL_SCAN_PROPAGATE 284

// Bitonic sort
#define OP_INTERNAL_BITONIC_STEP 285
#define OP_INTERNAL_BITONIC_PAD_INIT 286
#define OP_INTERNAL_BITONIC_COPY_BACK 287

// Radix sort
#define OP_INTERNAL_RADIX_HISTOGRAM 288
#define OP_INTERNAL_RADIX_SCATTER 289

// Utility
#define OP_INTERNAL_FILL_UINT 290
#define OP_INTERNAL_SCAN_UINT 291

// =============================================================================
// Convolution operations (300-301)
// =============================================================================

#define OP_CONV1D 300
#define OP_CONV2D 301

// =============================================================================
// Pooling operations (302-303)
// =============================================================================

#define OP_MAX_POOL2D 302
#define OP_AVG_POOL2D 303

// =============================================================================
// Normalization operations (310-311)
// =============================================================================

#define OP_LAYER_NORM 310
#define OP_BATCH_NORM 311

// =============================================================================
// Embedding operations (320)
// =============================================================================

#define OP_EMBEDDING 320

// =============================================================================
// Padding operations (330)
// =============================================================================

#define OP_PAD 330

#endif // COMPUTE_OPS_SHARED
