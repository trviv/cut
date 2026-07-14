#pragma once

#include <ComputeHandle.h>
#include <string>
#include <vector>

namespace cut {

/**
 * Logs a formatted message to the output.
 * @param format Printf-style format string.
 * @param ... Variable arguments matching the format string.
 */
extern void logMsg(const char *format, ...);

/**
 * Logs a message with a header followed by multiple lines.
 * @param header The header text to display before the lines.
 * @param lines A vector of C-strings to log after the header.
 */
extern void logMsg(const char *header, const std::vector<const char *> &lines);

/**
 * Logs a message with a header followed by multiple lines.
 * @param header The header text to display before the lines.
 * @param lines A vector of strings to log after the header.
 */
extern void logMsg(const char *header, const std::vector<std::string> &lines);

/**
 * Logs a formatted error message to the error output.
 * @param format Printf-style format string.
 * @param ... Variable arguments matching the format string.
 */
extern void logErr(const char *format, ...);

/**
 * Reads the contents of a shader file into a byte buffer.
 * @param filename Path to the shader file to read.
 * @return A vector containing the raw bytes of the shader file.
 */
static std::vector<char> readShaderFile(const std::string &filename);

/**
 * Shader source language for compilation.
 */
enum class ShaderLanguage {
  GLSL, ///< OpenGL Shading Language
  HLSL, ///< High Level Shading Language (DirectX)
};

/**
 * Compiles compute shader source code to SPIR-V bytecode at runtime.
 * Uses the shaderc library for compilation.
 * @param source The shader source code string.
 * @param filename Optional filename for error messages (defaults to "shader").
 * @param language Shader source language (defaults to GLSL).
 * @return A vector containing the compiled SPIR-V bytecode as uint32_t words.
 * @throws std::runtime_error if compilation fails.
 */
std::vector<uint32_t>
compileShaderToSpirv(const std::string &source,
                     const std::string &filename = "shader",
                     ShaderLanguage language = ShaderLanguage::GLSL);

/**
 * Compiles a GLSL compute shader file to SPIR-V bytecode at runtime.
 * Reads the file and compiles it using shaderc.
 * @param filepath Path to the shader source file.
 * @return A vector containing the compiled SPIR-V bytecode as uint32_t words.
 * @throws std::runtime_error if the file cannot be read or compilation fails.
 */
std::vector<uint32_t> compileShaderFileToSpirv(const std::string &filepath);

/**
 * Per-device capabilities populated by the backend during initialization.
 * Owned by each ComputeInterface instance and accessed via
 * ComputeInterface::caps() (or TensorStore::caps() from operator code).
 */
struct DeviceCaps {
  bool cooperativeMatrix = false; ///< VK_KHR_cooperative_matrix
  bool integerDotProduct = false; ///< VK_KHR_shader_integer_dot_product
  uint32_t subgroupSize = 32;     ///< Device subgroup/warp size
};

/**
 * Represents thread dimensions (workgroup or dispatch size).
 */
struct ThreadSize {
  uint32_t x = 0; ///< Size in the X dimension.
  uint32_t y = 0; ///< Size in the Y dimension.
  uint32_t z = 0; ///< Size in the Z dimension.
};

/**
 * Data type for buffer elements.
 */
enum class DataType {
  Float32, ///< 32-bit floating point (float).
  Float16, ///< 16-bit floating point (half).
  UInt32,  ///< 32-bit unsigned integer.
  Int32,   ///< 32-bit signed integer.
  Int8,    ///< 8-bit signed integer (for quantized weights).
};

/**
 * Returns the size in bytes for a given DataType.
 * @param dtype The data type.
 * @return Size in bytes.
 */
size_t dataTypeSize(DataType dtype);

/**
 * Returns a string name for a given DataType.
 * @param dtype The data type.
 * @return String representation of the data type.
 */
const char *dataTypeName(DataType dtype);

/// Widens a type to the next higher precision.
/// Float16 → Float32, Int8 → Float32. All others unchanged.
DataType widenPrecision(DataType dt);

/**
 * A lightweight wrapper for referencing raw data with size information.
 * Used to pass data to compute operations without copying.
 */
struct DataReference final {
  /**
   * Constructs a DataReference from any typed object.
   * @tparam T The type of the referenced data.
   * @param dataRef Reference to the data object.
   */
  template <typename T>
  DataReference(const T &dataRef) : ptr(&dataRef), size(sizeof(dataRef)) {}

  /**
   * Constructs a DataReference from a raw pointer and size.
   * @param dataPtr Pointer to the data.
   * @param size Size of the data in bytes.
   */
  DataReference(const void *dataPtr, uint32_t size)
      : ptr(dataPtr), size(size) {}

  const void *ptr;     ///< Pointer to the referenced data.
  const uint32_t size; ///< Size of the data in bytes.
};

/**
 * Describes the type of a shader resource binding.
 */
enum class BindingType {
  Sampler,       ///< Sampler for texture sampling.
  UniformBuffer, ///< Uniform buffer (constant data).
  StorageBuffer, ///< Storage buffer (read/write data).
  SampledImage,  ///< Sampled image (texture).
  StorageImage,  ///< Storage image (read/write texture).
  PushConstant,  ///< Push constant data.
};

/**
 * Describes the access mode for a shader resource binding.
 */
enum class BindingAccess {
  ReadOnly,  ///< Resource is only read from.
  WriteOnly, ///< Resource is only written to.
  ReadWrite, ///< Resource is both read from and written to.
};

/**
 * Describes a single shader resource binding extracted from SPIR-V reflection.
 */
struct BindingInfo {
  uint32_t binding;     ///< Binding index in the shader.
  uint32_t set;         ///< Descriptor set number.
  BindingType type;     ///< Type of the binding.
  BindingAccess access; ///< Access mode (read/write/readwrite).
};

/**
 * Contains all binding information extracted from a SPIR-V shader module.
 */
struct ShaderReflection {
  std::vector<BindingInfo> bindings; ///< All resource bindings.
  ThreadSize tgSize;         ///< Workgroup size from OpExecutionMode LocalSize.
  uint32_t dtypeVecSize;     ///< Vector size for dtype (from spec constant).
  uint32_t pushConstantSize; ///< Size of push constants in bytes.
};

/**
 * Reflects SPIR-V bytecode to extract binding information.
 * Parses the SPIR-V binary to identify all resource bindings including
 * uniforms, storage buffers, textures, samplers, and push constants.
 * @param spirvCode Vector containing the SPIR-V bytecode.
 * @return ShaderReflection containing all binding information.
 */
ShaderReflection reflectSpirvBindings(const std::vector<uint32_t> &spirvCode);

} // namespace cut
