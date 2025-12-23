
#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>

namespace cut {

// Template for binary operations using an operator (e.g., +, -, *, /)
static const char *binaryVecVecShaderTemplate = R"(
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint dtype_vec_size = %DTYPE_SIZE%;

layout(push_constant) uniform PushConstants {
    uint numElements;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer BufferA {
    %DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %DTYPE% dataOut[];
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
    %DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %DTYPE% dataOut[];
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
    %DTYPE% dataA[];
};

layout(set = 0, binding = 1, std430) restrict readonly buffer BufferB {
    %DTYPE% dataB[];
};

layout(set = 0, binding = 2, std430) restrict writeonly buffer BufferOutput {
    %DTYPE% dataOut[];
};

void main() {
    const uint index = gl_GlobalInvocationID.x;

    if (index * dtype_vec_size >= numElements) {
        return;
    }

    dataOut[index] = %DTYPE%(%EXPR%);
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
    %DTYPE% dataIn[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer BufferOutput {
    %DTYPE% dataOut[];
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

static std::string generateBinaryVecVecShader(const char *op,
                                              ScalarDataType datatype) {
  std::string shader = binaryVecVecShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%OP%", op);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecVecFuncShader(const char *func,
                                                  ScalarDataType datatype) {
  std::string shader = binaryVecVecFuncShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%FUNC%", func);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateBinaryVecVecCompareShader(const char *expr,
                                                     ScalarDataType datatype) {
  std::string shader = binaryVecVecCompareShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%EXPR%", expr);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

static std::string generateUnaryShader(const char *expr,
                                       ScalarDataType datatype) {
  std::string shader = unaryShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", getGLSLType(datatype));
  shader = replaceAll(shader, "%EXPR%", expr);
  shader = replaceAll(shader, "%DTYPE_SIZE%", "4");
  return shader;
}

std::optional<std::vector<uint32_t>>
getGeneratedShader(const ShaderEnum shader, const ScalarDataType datatype) {
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

  return compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
}

} // namespace cut
