
#include <ComputeCommon.h>
#include <Shaders.h>
#include <optional>

namespace cut {

// Template for binary operations using an operator (e.g., +, -, *, /)
static const char *binaryVecVecShaderTemplate = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    uint index = gl_GlobalInvocationID.x;

    if (index >= numElements) {
        return;
    }

    dataOut[index] = dataA[index] %OP% dataB[index];
}
)";

// Template for binary operations using a function (e.g., pow, min, max)
static const char *binaryVecVecFuncShaderTemplate = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    uint index = gl_GlobalInvocationID.x;

    if (index >= numElements) {
        return;
    }

    dataOut[index] = %FUNC%(dataA[index], dataB[index]);
}
)";

// Template for binary comparison operations (returns 1.0 or 0.0)
static const char *binaryVecVecCompareShaderTemplate = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    uint index = gl_GlobalInvocationID.x;

    if (index >= numElements) {
        return;
    }

    dataOut[index] = (dataA[index] %OP% dataB[index]) ? 1.0 : 0.0;
}
)";

// Template for unary operations (e.g., sqrt, sin, cos)
static const char *unaryShaderTemplate = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    uint index = gl_GlobalInvocationID.x;

    if (index >= numElements) {
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

static std::string generateBinaryVecVecShader(const char *op,
                                              const char *dtype = "float") {
  std::string shader = binaryVecVecShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", dtype);
  shader = replaceAll(shader, "%OP%", op);
  return shader;
}

static std::string generateBinaryVecVecFuncShader(const char *func,
                                                  const char *dtype = "float") {
  std::string shader = binaryVecVecFuncShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", dtype);
  shader = replaceAll(shader, "%FUNC%", func);
  return shader;
}

static std::string
generateBinaryVecVecCompareShader(const char *op, const char *dtype = "float") {
  std::string shader = binaryVecVecCompareShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", dtype);
  shader = replaceAll(shader, "%OP%", op);
  return shader;
}

static std::string generateUnaryShader(const char *expr,
                                       const char *dtype = "float") {
  std::string shader = unaryShaderTemplate;
  shader = replaceAll(shader, "%DTYPE%", dtype);
  shader = replaceAll(shader, "%EXPR%", expr);
  return shader;
}

std::optional<std::vector<uint32_t>>
getGeneratedShader(const ShaderEnum shader) {
  std::string shaderSource;
  std::string shaderName = "generated_shader";

  switch (shader) {
  // Binary arithmetic operations
  case BinaryVecVecAdd:
    shaderSource = generateBinaryVecVecShader("+");
    shaderName = "binary_vec_vec_add";
    break;
  case BinaryVecVecSub:
    shaderSource = generateBinaryVecVecShader("-");
    shaderName = "binary_vec_vec_sub";
    break;
  case BinaryVecVecMul:
    shaderSource = generateBinaryVecVecShader("*");
    shaderName = "binary_vec_vec_mul";
    break;
  case BinaryVecVecDiv:
    shaderSource = generateBinaryVecVecShader("/");
    shaderName = "binary_vec_vec_div";
    break;
  case BinaryVecVecMod:
    // GLSL uses mod() instead of fmod()
    shaderSource = generateBinaryVecVecFuncShader("mod");
    shaderName = "binary_vec_vec_mod";
    break;
  case BinaryVecVecPow:
    shaderSource = generateBinaryVecVecFuncShader("pow");
    shaderName = "binary_vec_vec_pow";
    break;
  case BinaryVecVecFloorDiv:
    // floor(a / b) - integer division rounded down
    shaderSource = generateBinaryVecVecShader("/");
    shaderSource = replaceAll(shaderSource, "dataA[index] / dataB[index]",
                              "floor(dataA[index] / dataB[index])");
    shaderName = "binary_vec_vec_floor_div";
    break;

  // Binary comparison operations
  case BinaryVecVecEqual:
    shaderSource = generateBinaryVecVecCompareShader("==");
    shaderName = "binary_vec_vec_equal";
    break;
  case BinaryVecVecNotEqual:
    shaderSource = generateBinaryVecVecCompareShader("!=");
    shaderName = "binary_vec_vec_not_equal";
    break;
  case BinaryVecVecLess:
    shaderSource = generateBinaryVecVecCompareShader("<");
    shaderName = "binary_vec_vec_less";
    break;
  case BinaryVecVecLessEqual:
    shaderSource = generateBinaryVecVecCompareShader("<=");
    shaderName = "binary_vec_vec_less_equal";
    break;
  case BinaryVecVecGreater:
    shaderSource = generateBinaryVecVecCompareShader(">");
    shaderName = "binary_vec_vec_greater";
    break;
  case BinaryVecVecGreaterEqual:
    shaderSource = generateBinaryVecVecCompareShader(">=");
    shaderName = "binary_vec_vec_greater_equal";
    break;

  // Binary min/max operations
  case BinaryVecVecMin:
    shaderSource = generateBinaryVecVecFuncShader("min");
    shaderName = "binary_vec_vec_min";
    break;
  case BinaryVecVecMax:
    shaderSource = generateBinaryVecVecFuncShader("max");
    shaderName = "binary_vec_vec_max";
    break;

  // Unary operations
  case UnaryNeg:
    shaderSource = generateUnaryShader("-dataIn[index]");
    shaderName = "unary_neg";
    break;
  case UnaryAbs:
    shaderSource = generateUnaryShader("abs(dataIn[index])");
    shaderName = "unary_abs";
    break;
  case UnarySqrt:
    shaderSource = generateUnaryShader("sqrt(dataIn[index])");
    shaderName = "unary_sqrt";
    break;
  case UnaryExp:
    shaderSource = generateUnaryShader("exp(dataIn[index])");
    shaderName = "unary_exp";
    break;
  case UnaryLog:
    shaderSource = generateUnaryShader("log(dataIn[index])");
    shaderName = "unary_log";
    break;
  case UnaryLog2:
    shaderSource = generateUnaryShader("log2(dataIn[index])");
    shaderName = "unary_log2";
    break;
  case UnaryLog10:
    // GLSL doesn't have log10, use log(x) / log(10) = log(x) * 0.4342944819
    shaderSource = generateUnaryShader("log(dataIn[index]) * 0.4342944819");
    shaderName = "unary_log10";
    break;
  case UnarySin:
    shaderSource = generateUnaryShader("sin(dataIn[index])");
    shaderName = "unary_sin";
    break;
  case UnaryCos:
    shaderSource = generateUnaryShader("cos(dataIn[index])");
    shaderName = "unary_cos";
    break;
  case UnaryTan:
    shaderSource = generateUnaryShader("tan(dataIn[index])");
    shaderName = "unary_tan";
    break;
  case UnaryAsin:
    shaderSource = generateUnaryShader("asin(dataIn[index])");
    shaderName = "unary_asin";
    break;
  case UnaryAcos:
    shaderSource = generateUnaryShader("acos(dataIn[index])");
    shaderName = "unary_acos";
    break;
  case UnaryAtan:
    shaderSource = generateUnaryShader("atan(dataIn[index])");
    shaderName = "unary_atan";
    break;
  case UnarySinh:
    shaderSource = generateUnaryShader("sinh(dataIn[index])");
    shaderName = "unary_sinh";
    break;
  case UnaryCosh:
    shaderSource = generateUnaryShader("cosh(dataIn[index])");
    shaderName = "unary_cosh";
    break;
  case UnaryTanh:
    shaderSource = generateUnaryShader("tanh(dataIn[index])");
    shaderName = "unary_tanh";
    break;
  case UnaryFloor:
    shaderSource = generateUnaryShader("floor(dataIn[index])");
    shaderName = "unary_floor";
    break;
  case UnaryCeil:
    shaderSource = generateUnaryShader("ceil(dataIn[index])");
    shaderName = "unary_ceil";
    break;
  case UnaryRound:
    shaderSource = generateUnaryShader("round(dataIn[index])");
    shaderName = "unary_round";
    break;
  case UnarySign:
    shaderSource = generateUnaryShader("sign(dataIn[index])");
    shaderName = "unary_sign";
    break;
  case UnaryReciprocal:
    shaderSource = generateUnaryShader("1.0 / dataIn[index]");
    shaderName = "unary_reciprocal";
    break;
  case UnarySquare:
    shaderSource = generateUnaryShader("dataIn[index] * dataIn[index]");
    shaderName = "unary_square";
    break;

  default:
    return std::nullopt;
  }

  return compileShaderToSpirv(shaderSource, shaderName, ShaderLanguage::GLSL);
}

} // namespace cut
