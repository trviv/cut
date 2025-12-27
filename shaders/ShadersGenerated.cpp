
#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>
#include <unordered_map>

namespace cut {

/// Enable/disable caching for generated SPIR-V shaders
constexpr bool kEnableShaderCache = true;

/// Cache for generated SPIR-V shaders, keyed by (OperatorEnum, ScalarDataType)
static std::unordered_map<uint64_t, std::vector<uint32_t>> shaderCache;

/// Creates a cache key from operator enum and datatype
static uint64_t makeCacheKey(OperatorEnum shader, ScalarDataType datatype) {
  return (static_cast<uint64_t>(shader) << 32) |
         static_cast<uint64_t>(datatype);
}

// Template for binary operations using an operator (e.g., +, -, *, /)
static const char *binaryVecVecShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = dataA[index] %OP% dataB[index];
}
)";

// Template for binary operations using a function (e.g., pow, min, max)
static const char *binaryVecVecFuncShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %FUNC%(dataA[index], dataB[index]);
}
)";

// Template for binary comparison operations (returns 1.0 or 0.0)
static const char *binaryVecVecCompareShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %VEC_DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %VEC_DTYPE%(%EXPR%);
}
)";

// Template for binary vec-scalar operations using an operator (e.g., +, -, *,
// /) Scalar is passed via push constants
static const char *binaryVecScalarShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
    %SCALAR_DTYPE% scalar;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = dataA[index] %OP% %VEC_DTYPE%(scalar);
}
)";

// Template for binary vec-scalar operations using a function (e.g., pow, min,
// max) Scalar is passed via push constants
static const char *binaryVecScalarFuncShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
    %SCALAR_DTYPE% scalar;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %FUNC%(dataA[index], %VEC_DTYPE%(scalar));
}
)";

// Template for binary vec-scalar comparison operations (returns 1.0 or 0.0)
// Scalar is passed via push constants
static const char *binaryVecScalarCompareShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
    %SCALAR_DTYPE% scalar;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %VEC_DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %VEC_DTYPE%(%EXPR%);
}
)";

// Template for unary operations (e.g., sqrt, sin, cos)
static const char *unaryShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferIn {
    %VEC_DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %VEC_DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %EXPR%;
}
)";

static std::string replaceAll(const std::string &str,
                              const std::string &from,
                              const std::string &to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.length(), to);
    pos += to.length();
  }
  return result;
}

static const char *getGLSLType(ScalarDataType datatype) {
  switch (datatype) {
  case ScalarDataType::Float:
    return "vec4";
  case ScalarDataType::Half:
    return "mediump vec4";
  case ScalarDataType::UInt:
    return "uvec4";
  case ScalarDataType::Int:
    return "ivec4";
  default:
    return "vec4";
  }
}

static const char *getGLSLScalarType(ScalarDataType datatype) {
  switch (datatype) {
  case ScalarDataType::Float:
    return "float";
  case ScalarDataType::Half:
    return "mediump float";
  case ScalarDataType::UInt:
    return "uint";
  case ScalarDataType::Int:
    return "int";
  default:
    return "float";
  }
}

static std::string generateBinaryVecVecShader(const char *op,
                                              ScalarDataType datatype) {
  std::string shader = binaryVecVecShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%OP%", op);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecVecFuncShader(const char *func,
                                                  ScalarDataType datatype) {
  std::string shader = binaryVecVecFuncShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%FUNC%", func);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecVecCompareShader(const char *expr,
                                                     ScalarDataType datatype) {
  std::string shader = binaryVecVecCompareShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%EXPR%", expr);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateUnaryShader(const char *expr,
                                       ScalarDataType datatype) {
  std::string shader = unaryShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%EXPR%", expr);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecScalarShader(const char *op,
                                                 ScalarDataType datatype) {
  std::string shader = binaryVecScalarShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%SCALAR_DTYPE%", getGLSLScalarType(datatype));
  shader = replaceAll(shader, "%OP%", op);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecScalarFuncShader(const char *func,
                                                     ScalarDataType datatype) {
  std::string shader = binaryVecScalarFuncShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%SCALAR_DTYPE%", getGLSLScalarType(datatype));
  shader = replaceAll(shader, "%FUNC%", func);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string
generateBinaryVecScalarCompareShader(const std::string &expr,
                                     ScalarDataType datatype) {
  std::string shader = binaryVecScalarCompareShaderTemplate;
  shader = replaceAll(shader, "%VEC_DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%SCALAR_DTYPE%", getGLSLScalarType(datatype));
  shader = replaceAll(shader, "%EXPR%", expr);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

std::optional<std::vector<uint32_t>>
getGeneratedShader(const OperatorEnum shader, const ScalarDataType datatype) {
  // Check cache first
  if constexpr (kEnableShaderCache) {
    uint64_t cacheKey = makeCacheKey(shader, datatype);
    auto it = shaderCache.find(cacheKey);
    if (it != shaderCache.end()) {
      return it->second;
    }
  }

  std::string shaderSource;
  std::string shaderName = "generated_shader";

  switch (shader) {
  // Binary arithmetic operations
  case BinaryVecVecAdd:
    shaderSource = generateBinaryVecVecShader("+", datatype);
    shaderName = "binary_vec_vec_add";
    break;
  case BinaryVecVecSub:
    shaderSource = generateBinaryVecVecShader("-", datatype);
    shaderName = "binary_vec_vec_sub";
    break;
  case BinaryVecVecMul:
    shaderSource = generateBinaryVecVecShader("*", datatype);
    shaderName = "binary_vec_vec_mul";
    break;
  case BinaryVecVecDiv:
    shaderSource = generateBinaryVecVecShader("/", datatype);
    shaderName = "binary_vec_vec_div";
    break;
  case BinaryVecVecMod:
    // GLSL uses mod() instead of fmod()
    shaderSource = generateBinaryVecVecFuncShader("mod", datatype);
    shaderName = "binary_vec_vec_mod";
    break;
  case BinaryVecVecPow:
    shaderSource = generateBinaryVecVecFuncShader("pow", datatype);
    shaderName = "binary_vec_vec_pow";
    break;
  case BinaryVecVecFloorDiv:
    // floor(a / b) - integer division rounded down
    shaderSource = generateBinaryVecVecShader("/", datatype);
    shaderSource = replaceAll(shaderSource, "dataA[index] / dataB[index]",
                              "floor(dataA[index] / dataB[index])");
    shaderName = "binary_vec_vec_floor_div";
    break;

  // Binary comparison operations
  case BinaryVecVecEqual:
    shaderSource = generateBinaryVecVecCompareShader(
        "equal(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_equal";
    break;
  case BinaryVecVecNotEqual:
    shaderSource = generateBinaryVecVecCompareShader(
        "notEqual(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_not_equal";
    break;
  case BinaryVecVecLess:
    shaderSource = generateBinaryVecVecCompareShader(
        "lessThan(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_less";
    break;
  case BinaryVecVecLessEqual:
    shaderSource = generateBinaryVecVecCompareShader(
        "lessThanEqual(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_less_equal";
    break;
  case BinaryVecVecGreater:
    shaderSource = generateBinaryVecVecCompareShader(
        "greaterThan(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_greater";
    break;
  case BinaryVecVecGreaterEqual:
    shaderSource = generateBinaryVecVecCompareShader(
        "greaterThanEqual(dataA[index], dataB[index])", datatype);
    shaderName = "binary_vec_vec_greater_equal";
    break;

  // Binary min/max operations
  case BinaryVecVecMin:
    shaderSource = generateBinaryVecVecFuncShader("min", datatype);
    shaderName = "binary_vec_vec_min";
    break;
  case BinaryVecVecMax:
    shaderSource = generateBinaryVecVecFuncShader("max", datatype);
    shaderName = "binary_vec_vec_max";
    break;

  // Binary vec-scalar arithmetic operations
  case BinaryVecScalarAdd:
    shaderSource = generateBinaryVecScalarShader("+", datatype);
    shaderName = "binary_vec_scalar_add";
    break;
  case BinaryVecScalarSub:
    shaderSource = generateBinaryVecScalarShader("-", datatype);
    shaderName = "binary_vec_scalar_sub";
    break;
  case BinaryVecScalarMul:
    shaderSource = generateBinaryVecScalarShader("*", datatype);
    shaderName = "binary_vec_scalar_mul";
    break;
  case BinaryVecScalarDiv:
    shaderSource = generateBinaryVecScalarShader("/", datatype);
    shaderName = "binary_vec_scalar_div";
    break;
  case BinaryVecScalarMod:
    shaderSource = generateBinaryVecScalarFuncShader("mod", datatype);
    shaderName = "binary_vec_scalar_mod";
    break;
  case BinaryVecScalarPow:
    shaderSource = generateBinaryVecScalarFuncShader("pow", datatype);
    shaderName = "binary_vec_scalar_pow";
    break;
  case BinaryVecScalarFloorDiv:
    shaderSource = generateBinaryVecScalarShader("/", datatype);
    shaderSource = replaceAll(
        shaderSource,
        "dataA[index] / " + std::string(getGLSLType(datatype)) + "(scalar)",
        "floor(dataA[index] / " + std::string(getGLSLType(datatype)) +
            "(scalar))");
    shaderName = "binary_vec_scalar_floor_div";
    break;

  // Binary vec-scalar comparison operations
  case BinaryVecScalarEqual:
    shaderSource = generateBinaryVecScalarCompareShader(
        "equal(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_equal";
    break;
  case BinaryVecScalarNotEqual:
    shaderSource = generateBinaryVecScalarCompareShader(
        "notEqual(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_not_equal";
    break;
  case BinaryVecScalarLess:
    shaderSource = generateBinaryVecScalarCompareShader(
        "lessThan(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_less";
    break;
  case BinaryVecScalarLessEqual:
    shaderSource = generateBinaryVecScalarCompareShader(
        "lessThanEqual(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_less_equal";
    break;
  case BinaryVecScalarGreater:
    shaderSource = generateBinaryVecScalarCompareShader(
        "greaterThan(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_greater";
    break;
  case BinaryVecScalarGreaterEqual:
    shaderSource = generateBinaryVecScalarCompareShader(
        "greaterThanEqual(dataA[index], " + std::string(getGLSLType(datatype)) +
            "(scalar))",
        datatype);
    shaderName = "binary_vec_scalar_greater_equal";
    break;

  // Binary vec-scalar min/max operations
  case BinaryVecScalarMin:
    shaderSource = generateBinaryVecScalarFuncShader("min", datatype);
    shaderName = "binary_vec_scalar_min";
    break;
  case BinaryVecScalarMax:
    shaderSource = generateBinaryVecScalarFuncShader("max", datatype);
    shaderName = "binary_vec_scalar_max";
    break;

  // Unary operations
  case UnaryNeg:
    shaderSource = generateUnaryShader("-dataIn[index]", datatype);
    shaderName = "unary_neg";
    break;
  case UnaryAbs:
    shaderSource = generateUnaryShader("abs(dataIn[index])", datatype);
    shaderName = "unary_abs";
    break;
  case UnarySqrt:
    shaderSource = generateUnaryShader("sqrt(dataIn[index])", datatype);
    shaderName = "unary_sqrt";
    break;
  case UnaryExp:
    shaderSource = generateUnaryShader("exp(dataIn[index])", datatype);
    shaderName = "unary_exp";
    break;
  case UnaryLog:
    shaderSource = generateUnaryShader("log(dataIn[index])", datatype);
    shaderName = "unary_log";
    break;
  case UnaryLog2:
    shaderSource = generateUnaryShader("log2(dataIn[index])", datatype);
    shaderName = "unary_log2";
    break;
  case UnaryLog10:
    // GLSL doesn't have log10, use log(x) / log(10) = log(x) * 0.4342944819
    shaderSource =
        generateUnaryShader("log(dataIn[index]) * 0.4342944819", datatype);
    shaderName = "unary_log10";
    break;
  case UnarySin:
    shaderSource = generateUnaryShader("sin(dataIn[index])", datatype);
    shaderName = "unary_sin";
    break;
  case UnaryCos:
    shaderSource = generateUnaryShader("cos(dataIn[index])", datatype);
    shaderName = "unary_cos";
    break;
  case UnaryTan:
    shaderSource = generateUnaryShader("tan(dataIn[index])", datatype);
    shaderName = "unary_tan";
    break;
  case UnaryAsin:
    shaderSource = generateUnaryShader("asin(dataIn[index])", datatype);
    shaderName = "unary_asin";
    break;
  case UnaryAcos:
    shaderSource = generateUnaryShader("acos(dataIn[index])", datatype);
    shaderName = "unary_acos";
    break;
  case UnaryAtan:
    shaderSource = generateUnaryShader("atan(dataIn[index])", datatype);
    shaderName = "unary_atan";
    break;
  case UnarySinh:
    shaderSource = generateUnaryShader("sinh(dataIn[index])", datatype);
    shaderName = "unary_sinh";
    break;
  case UnaryCosh:
    shaderSource = generateUnaryShader("cosh(dataIn[index])", datatype);
    shaderName = "unary_cosh";
    break;
  case UnaryTanh:
    shaderSource = generateUnaryShader("tanh(dataIn[index])", datatype);
    shaderName = "unary_tanh";
    break;
  case UnaryFloor:
    shaderSource = generateUnaryShader("floor(dataIn[index])", datatype);
    shaderName = "unary_floor";
    break;
  case UnaryCeil:
    shaderSource = generateUnaryShader("ceil(dataIn[index])", datatype);
    shaderName = "unary_ceil";
    break;
  case UnaryRound:
    shaderSource = generateUnaryShader("round(dataIn[index])", datatype);
    shaderName = "unary_round";
    break;
  case UnarySign:
    shaderSource = generateUnaryShader("sign(dataIn[index])", datatype);
    shaderName = "unary_sign";
    break;
  case UnaryReciprocal:
    shaderSource = generateUnaryShader("1.0 / dataIn[index]", datatype);
    shaderName = "unary_reciprocal";
    break;
  case UnarySquare:
    shaderSource =
        generateUnaryShader("dataIn[index] * dataIn[index]", datatype);
    shaderName = "unary_square";
    break;

  default:
    return std::nullopt;
  }

  auto spirv =
      compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
  if constexpr (kEnableShaderCache) {
    shaderCache[makeCacheKey(shader, datatype)] = spirv;
  }
  return spirv;
}

} // namespace cut
