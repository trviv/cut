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

// Binary vec-scalar arithmetic operations
#define OP_BINARY_VEC_SCALAR_ADD 15
#define OP_BINARY_VEC_SCALAR_SUB 16
#define OP_BINARY_VEC_SCALAR_MUL 17
#define OP_BINARY_VEC_SCALAR_DIV 18
#define OP_BINARY_VEC_SCALAR_MOD 19
#define OP_BINARY_VEC_SCALAR_POW 20
#define OP_BINARY_VEC_SCALAR_FLOOR_DIV 21

// Binary vec-scalar comparison operations
#define OP_BINARY_VEC_SCALAR_EQUAL 22
#define OP_BINARY_VEC_SCALAR_NOT_EQUAL 23
#define OP_BINARY_VEC_SCALAR_LESS 24
#define OP_BINARY_VEC_SCALAR_LESS_EQUAL 25
#define OP_BINARY_VEC_SCALAR_GREATER 26
#define OP_BINARY_VEC_SCALAR_GREATER_EQUAL 27

// Binary vec-scalar min/max operations
#define OP_BINARY_VEC_SCALAR_MIN 28
#define OP_BINARY_VEC_SCALAR_MAX 29

// Unary operations
#define OP_UNARY_NEG 30
#define OP_UNARY_ABS 31
#define OP_UNARY_SQRT 32
#define OP_UNARY_EXP 33
#define OP_UNARY_LOG 34
#define OP_UNARY_LOG2 35
#define OP_UNARY_LOG10 36
#define OP_UNARY_SIN 37
#define OP_UNARY_COS 38
#define OP_UNARY_TAN 39
#define OP_UNARY_ASIN 40
#define OP_UNARY_ACOS 41
#define OP_UNARY_ATAN 42
#define OP_UNARY_SINH 43
#define OP_UNARY_COSH 44
#define OP_UNARY_TANH 45
#define OP_UNARY_FLOOR 46
#define OP_UNARY_CEIL 47
#define OP_UNARY_ROUND 48
#define OP_UNARY_SIGN 49
#define OP_UNARY_RECIPROCAL 50
#define OP_UNARY_SQUARE 51

#endif // COMPUTE_OPS_SHARED
