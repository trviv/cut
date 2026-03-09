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
#define DTYPE_INT8 4

// =============================================================================
// Binary operations (0-32) - unified for all binary variants
// =============================================================================

// Binary arithmetic operations
#define OP_BINARY_ADD 0
#define OP_BINARY_SUB 1
#define OP_BINARY_MUL 2
#define OP_BINARY_DIV 3
#define OP_BINARY_MOD 4
#define OP_BINARY_POW 5
#define OP_BINARY_FLOOR_DIV 6

// Binary comparison operations
#define OP_BINARY_EQUAL 7
#define OP_BINARY_NOT_EQUAL 8
#define OP_BINARY_LESS 9
#define OP_BINARY_LESS_EQUAL 10
#define OP_BINARY_GREATER 11
#define OP_BINARY_GREATER_EQUAL 12

// Binary min/max operations
#define OP_BINARY_MIN 13
#define OP_BINARY_MAX 14

// Binary bitwise operations
#define OP_BINARY_BITWISE_AND 15
#define OP_BINARY_BITWISE_OR 16
#define OP_BINARY_BITWISE_XOR 17
#define OP_BINARY_LEFT_SHIFT 18
#define OP_BINARY_RIGHT_SHIFT 19

// Binary logical operations
#define OP_BINARY_LOGICAL_AND 20
#define OP_BINARY_LOGICAL_OR 21
#define OP_BINARY_LOGICAL_XOR 22

// Binary math operations
#define OP_BINARY_ATAN2 23
#define OP_BINARY_HYPOT 24
#define OP_BINARY_COPYSIGN 25
#define OP_BINARY_FMOD 26

// Binary activation operations
#define OP_BINARY_LEAKY_RELU 27
#define OP_BINARY_PRELU 28
#define OP_BINARY_HARDSHRINK 29
#define OP_BINARY_SOFTSHRINK 30
#define OP_BINARY_LOGADDEXP 31
#define OP_BINARY_LOGADDEXP2 32

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
#define OP_CAST 78

// Matmul variants are defined in matmul_variants.json and selected by index
// at runtime rather than by individual op codes.

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
#define OP_REDUCE_VARIANCE 210
#define OP_REDUCE_RMS 211
#define OP_REDUCE_LOGSUMEXP 212
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
// Normalization operations (310-314)
// =============================================================================

#define OP_LAYER_NORM 310
#define OP_BATCH_NORM 311
#define OP_RMS_NORM 312
#define OP_EXTENDED_RMS_NORM 313
#define OP_SOFTMAX 315
#define OP_LOG_SOFTMAX 316

// =============================================================================
// Embedding operations (320)
// =============================================================================

#define OP_EMBEDDING 320

// =============================================================================
// Padding operations (330)
// =============================================================================

#define OP_PAD 330

// =============================================================================
// Expand operations (340)
// =============================================================================

#define OP_EXPAND 340

// =============================================================================
// RoPE operations (350)
// =============================================================================

#define OP_ROPE 350

// =============================================================================
// Attention operations (360-361)
// =============================================================================

#define OP_CACHE_WRITE 360
#define OP_ATTENTION 361

// =============================================================================
// Quantized matmul operations (371)
// =============================================================================

#define OP_MATMUL_Q8 371

#define OP_MATMUL_Q4 374

// =============================================================================
// Fused binary operations (380)
// =============================================================================

#define OP_FUSED_BINARY 380

// =============================================================================
// Dequantization operations (390)
// =============================================================================

#define OP_DEQUANTIZE 390

// =============================================================================
// Sampling operations (400)
// =============================================================================

#define OP_REPETITION_PENALTY 400

// =============================================================================
// Q4 transpose operations (401)
// =============================================================================

#define OP_TRANSPOSE_Q4 401

#endif // COMPUTE_OPS_SHARED
