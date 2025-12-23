#include <ComputeCommon.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>

#include <shaderc/shaderc.h>

namespace cut {

#define WRITE_MSG(prefix)                                                      \
  va_list args;                                                                \
  va_start(args, format);                                                      \
  /* Copy a prefix to the buffer */                                            \
  snprintf(msg, sizeof(msg), prefix);                                          \
  /* Copy args to the buffer */                                                \
  vsnprintf(&msg[std::strlen(prefix)], sizeof(msg) - std::strlen(prefix),      \
            format, args);                                                     \
  va_end(args);

void logMsg(const char *format, ...) {
  char msg[256];
  WRITE_MSG("Info: ")
  printf("\n%s\n", msg);
}

void logMsg(const char *header, const std::vector<const char *> &lines) {
  std::string msg(header);
  msg += ":";
  for (const auto line : lines) {
    msg += std::string("\n") + line;
  }
  printf("\n%s\n", msg.c_str());
}

void logMsg(const char *header, const std::vector<std::string> &lines) {
  std::string msg(header);
  msg += ":";
  for (const auto &line : lines) {
    msg += "\n" + line;
  }
  printf("\n%s\n", msg.c_str());
}

void logErr(const char *format, ...) {
  char msg[256];
  WRITE_MSG("Error: ")
  throw std::runtime_error(msg);
}

std::vector<char> readShaderFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    logErr("Failed to open shader file: %s", filename.c_str());
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

std::vector<uint32_t> compileShaderToSpirv(const std::string &source,
                                           const std::string &filename,
                                           ShaderLanguage language) {
  // Create compiler and options
  shaderc_compiler_t compiler = shaderc_compiler_initialize();
  if (!compiler) {
    logErr("Failed to initialize shaderc compiler");
  }

  shaderc_compile_options_t options = shaderc_compile_options_initialize();
  if (!options) {
    shaderc_compiler_release(compiler);
    logErr("Failed to initialize shaderc compile options");
  }

  // Set compilation options
  shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_0);
  shaderc_compile_options_set_optimization_level(
      options, shaderc_optimization_level_performance);

  // Set source language
  if (language == ShaderLanguage::HLSL) {
    shaderc_compile_options_set_source_language(options,
                                                shaderc_source_language_hlsl);
  } else {
    shaderc_compile_options_set_source_language(options,
                                                shaderc_source_language_glsl);
  }

  // Compile to SPIR-V (compute shader only)
  shaderc_compilation_result_t result = shaderc_compile_into_spv(
      compiler, source.c_str(), source.size(), shaderc_compute_shader,
      filename.c_str(), "main", options);

  // Check compilation status
  shaderc_compilation_status status =
      shaderc_result_get_compilation_status(result);

  if (status != shaderc_compilation_status_success) {
    const char *errorMsg = shaderc_result_get_error_message(result);
    std::string errorStr = errorMsg ? errorMsg : "Unknown compilation error";

    shaderc_result_release(result);
    shaderc_compile_options_release(options);
    shaderc_compiler_release(compiler);

    logErr("Shader compilation failed for '%s': %s", filename.c_str(),
           errorStr.c_str());
  }

  // Get the compiled SPIR-V bytecode
  size_t spirvSize = shaderc_result_get_length(result);
  const char *spirvBytes =
      reinterpret_cast<const char *>(shaderc_result_get_bytes(result));

  // Copy to vector of uint32_t (SPIR-V is always 4-byte aligned)
  std::vector<uint32_t> spirvCode(spirvSize / sizeof(uint32_t));
  std::memcpy(spirvCode.data(), spirvBytes, spirvSize);

  // Log compilation info
  size_t numWarnings = shaderc_result_get_num_warnings(result);
  if (numWarnings > 0) {
    logMsg("Shader '%s' compiled with %zu warning(s)", filename.c_str(),
           numWarnings);
  }

  // Cleanup
  shaderc_result_release(result);
  shaderc_compile_options_release(options);
  shaderc_compiler_release(compiler);

  logMsg("Shader '%s' compiled successfully: %zu bytes of SPIR-V",
         filename.c_str(), spirvSize);

  return spirvCode;
}

std::vector<uint32_t> compileShaderFileToSpirv(const std::string &filepath) {
  // Read the shader source file
  std::ifstream file(filepath);
  if (!file.is_open()) {
    logErr("Failed to open shader file: %s", filepath.c_str());
  }

  std::string source((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  file.close();

  // Extract filename for error messages
  size_t slashPos = filepath.rfind('/');
  std::string filename = (slashPos != std::string::npos)
                             ? filepath.substr(slashPos + 1)
                             : filepath;

  return compileShaderToSpirv(source, filename);
}

// SPIR-V constants
namespace spirv {
constexpr uint32_t MagicNumber = 0x07230203;

// Opcodes
constexpr uint32_t OpDecorate = 71;
constexpr uint32_t OpMemberDecorate = 72;
constexpr uint32_t OpExecutionMode = 16;
constexpr uint32_t OpTypeInt = 21;
constexpr uint32_t OpTypeFloat = 22;
constexpr uint32_t OpTypeVector = 23;
constexpr uint32_t OpTypeMatrix = 24;
constexpr uint32_t OpTypeStruct = 30;
constexpr uint32_t OpTypePointer = 32;
constexpr uint32_t OpVariable = 59;
constexpr uint32_t OpTypeImage = 25;
constexpr uint32_t OpTypeSampler = 26;
constexpr uint32_t OpTypeSampledImage = 27;

// Decorations
constexpr uint32_t DecorationBinding = 33;
constexpr uint32_t DecorationDescriptorSet = 34;
constexpr uint32_t DecorationNonWritable = 24;
constexpr uint32_t DecorationNonReadable = 25;
constexpr uint32_t DecorationBlock = 2;
constexpr uint32_t DecorationBufferBlock = 3;
constexpr uint32_t DecorationOffset = 35;

// Execution modes
constexpr uint32_t ExecutionModeLocalSize = 17;

// Storage classes
constexpr uint32_t StorageClassUniform = 2;
constexpr uint32_t StorageClassUniformConstant = 0;
constexpr uint32_t StorageClassStorageBuffer = 12;
constexpr uint32_t StorageClassPushConstant = 9;
} // namespace spirv

ShaderReflection reflectSpirvBindings(const std::vector<uint32_t> &spirvCode) {
  ShaderReflection reflection{};
  reflection.pushConstantSize = 0;
  reflection.tgSize = {0, 0, 0};

  if (spirvCode.size() < 5 || spirvCode[0] != spirv::MagicNumber) {
    logErr("Invalid SPIR-V: bad magic number or too small");
    return reflection;
  }

  // Maps to track decorations and types
  std::unordered_map<uint32_t, uint32_t> idToBinding;
  std::unordered_map<uint32_t, uint32_t> idToDescriptorSet;
  std::unordered_map<uint32_t, bool> idIsNonWritable;
  std::unordered_map<uint32_t, bool> idIsNonReadable;
  std::unordered_map<uint32_t, bool> idIsBufferBlock;
  std::unordered_map<uint32_t, uint32_t> pointerToPointedType;
  std::unordered_map<uint32_t, uint32_t> idToStorageClass;
  std::unordered_map<uint32_t, bool> idIsImage;
  std::unordered_map<uint32_t, bool> idIsSampler;
  std::unordered_map<uint32_t, bool> idIsSampledImage;

  // Maps for type size calculation
  std::unordered_map<uint32_t, uint32_t>
      idToTypeSize; // Type ID -> size in bytes
  std::unordered_map<uint32_t, std::vector<uint32_t>>
      structMembers; // Struct ID -> member type IDs
  // Map: struct ID -> (member index -> offset)
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
      memberOffsets;

  // First pass: collect decorations and type info
  size_t i = 5; // Skip header
  while (i < spirvCode.size()) {
    uint32_t instruction = spirvCode[i];
    uint32_t opcode = instruction & 0xFFFF;
    uint32_t wordCount = instruction >> 16;

    if (wordCount == 0 || i + wordCount > spirvCode.size()) {
      break;
    }

    switch (opcode) {
    case spirv::OpDecorate: {
      if (wordCount >= 3) {
        uint32_t targetId = spirvCode[i + 1];
        uint32_t decoration = spirvCode[i + 2];

        if (decoration == spirv::DecorationBinding && wordCount >= 4) {
          idToBinding[targetId] = spirvCode[i + 3];
        } else if (decoration == spirv::DecorationDescriptorSet &&
                   wordCount >= 4) {
          idToDescriptorSet[targetId] = spirvCode[i + 3];
        } else if (decoration == spirv::DecorationNonWritable) {
          idIsNonWritable[targetId] = true;
        } else if (decoration == spirv::DecorationNonReadable) {
          idIsNonReadable[targetId] = true;
        } else if (decoration == spirv::DecorationBufferBlock) {
          idIsBufferBlock[targetId] = true;
        }
      }
      break;
    }
    case spirv::OpMemberDecorate: {
      // OpMemberDecorate %structType member decoration [operands...]
      if (wordCount >= 4) {
        uint32_t structId = spirvCode[i + 1];
        uint32_t memberIndex = spirvCode[i + 2];
        uint32_t decoration = spirvCode[i + 3];

        if (decoration == spirv::DecorationOffset && wordCount >= 5) {
          memberOffsets[structId][memberIndex] = spirvCode[i + 4];
        }
      }
      break;
    }
    case spirv::OpExecutionMode: {
      // OpExecutionMode %entryPoint mode [operands...]
      // For LocalSize: OpExecutionMode %entryPoint LocalSize x y z
      if (wordCount >= 3) {
        uint32_t mode = spirvCode[i + 2];
        if (mode == spirv::ExecutionModeLocalSize && wordCount >= 6) {
          reflection.tgSize.x = spirvCode[i + 3];
          reflection.tgSize.y = spirvCode[i + 4];
          reflection.tgSize.z = spirvCode[i + 5];
        }
      }
      break;
    }
    case spirv::OpTypeInt:
    case spirv::OpTypeFloat: {
      // OpTypeInt/OpTypeFloat %result width [signedness]
      if (wordCount >= 3) {
        uint32_t resultId = spirvCode[i + 1];
        uint32_t width = spirvCode[i + 2]; // Width in bits
        idToTypeSize[resultId] = width / 8;
      }
      break;
    }
    case spirv::OpTypeVector: {
      // OpTypeVector %result %componentType componentCount
      if (wordCount >= 4) {
        uint32_t resultId = spirvCode[i + 1];
        uint32_t componentType = spirvCode[i + 2];
        uint32_t componentCount = spirvCode[i + 3];
        if (idToTypeSize.count(componentType) > 0) {
          idToTypeSize[resultId] = idToTypeSize[componentType] * componentCount;
        }
      }
      break;
    }
    case spirv::OpTypeMatrix: {
      // OpTypeMatrix %result %columnType columnCount
      if (wordCount >= 4) {
        uint32_t resultId = spirvCode[i + 1];
        uint32_t columnType = spirvCode[i + 2];
        uint32_t columnCount = spirvCode[i + 3];
        if (idToTypeSize.count(columnType) > 0) {
          idToTypeSize[resultId] = idToTypeSize[columnType] * columnCount;
        }
      }
      break;
    }
    case spirv::OpTypeStruct: {
      // OpTypeStruct %result [memberTypes...]
      if (wordCount >= 2) {
        uint32_t resultId = spirvCode[i + 1];
        std::vector<uint32_t> members;
        for (uint32_t m = 2; m < wordCount; ++m) {
          members.emplace_back(spirvCode[i + m]);
        }
        structMembers[resultId] = members;
      }
      break;
    }
    case spirv::OpTypePointer: {
      if (wordCount >= 4) {
        uint32_t resultId = spirvCode[i + 1];
        uint32_t storageClass = spirvCode[i + 2];
        uint32_t pointedType = spirvCode[i + 3];
        pointerToPointedType[resultId] = pointedType;
        idToStorageClass[resultId] = storageClass;
      }
      break;
    }
    case spirv::OpTypeImage: {
      if (wordCount >= 2) {
        idIsImage[spirvCode[i + 1]] = true;
      }
      break;
    }
    case spirv::OpTypeSampler: {
      if (wordCount >= 2) {
        idIsSampler[spirvCode[i + 1]] = true;
      }
      break;
    }
    case spirv::OpTypeSampledImage: {
      if (wordCount >= 2) {
        idIsSampledImage[spirvCode[i + 1]] = true;
      }
      break;
    }
    default:
      break;
    }
    i += wordCount;
  }

  // Helper lambda to calculate struct size from member offsets and sizes
  auto calculateStructSize = [&](uint32_t structId) -> uint32_t {
    if (structMembers.count(structId) == 0) {
      return 0;
    }

    const auto &members = structMembers[structId];
    if (members.empty()) {
      return 0;
    }

    // If we have offset decorations, use offset + size of last member
    if (memberOffsets.count(structId) > 0) {
      const auto &offsets = memberOffsets[structId];
      uint32_t maxOffset = 0;
      uint32_t lastMemberSize = 0;
      uint32_t lastMemberIndex = 0;

      for (const auto &[memberIdx, offset] : offsets) {
        if (offset >= maxOffset) {
          maxOffset = offset;
          lastMemberIndex = memberIdx;
        }
      }

      // Get size of the last member
      if (lastMemberIndex < members.size()) {
        uint32_t lastMemberType = members[lastMemberIndex];
        if (idToTypeSize.count(lastMemberType) > 0) {
          lastMemberSize = idToTypeSize[lastMemberType];
        } else {
          // Default to 4 bytes if unknown
          lastMemberSize = 4;
        }
      }

      return maxOffset + lastMemberSize;
    }

    // Fallback: sum up member sizes
    uint32_t totalSize = 0;
    for (uint32_t memberType : members) {
      if (idToTypeSize.count(memberType) > 0) {
        totalSize += idToTypeSize[memberType];
      } else {
        totalSize += 4; // Default
      }
    }
    return totalSize;
  };

  // Second pass: find variables
  i = 5;
  while (i < spirvCode.size()) {
    uint32_t instruction = spirvCode[i];
    uint32_t opcode = instruction & 0xFFFF;
    uint32_t wordCount = instruction >> 16;

    if (wordCount == 0 || i + wordCount > spirvCode.size()) {
      break;
    }

    if (opcode == spirv::OpVariable && wordCount >= 4) {
      uint32_t pointerTypeId = spirvCode[i + 1];
      uint32_t resultId = spirvCode[i + 2];
      uint32_t storageClass = spirvCode[i + 3];

      // Handle push constants
      if (storageClass == spirv::StorageClassPushConstant) {
        // Push constants don't have binding/set decorations
        BindingInfo info{};
        info.binding = 0;
        info.set = 0;
        info.type = BindingType::PushConstant;
        info.access = BindingAccess::ReadOnly;
        reflection.bindings.emplace_back(info);

        // Calculate push constant size from the pointed struct type
        if (pointerToPointedType.count(pointerTypeId) > 0) {
          uint32_t structTypeId = pointerToPointedType[pointerTypeId];
          reflection.pushConstantSize = calculateStructSize(structTypeId);
        }
      }
      // Handle bound resources
      else if (idToBinding.count(resultId) > 0) {
        BindingInfo info{};
        info.binding = idToBinding[resultId];
        info.set = idToDescriptorSet.count(resultId) > 0
                       ? idToDescriptorSet[resultId]
                       : 0;

        // Determine type based on storage class and pointed type
        uint32_t pointedType = pointerToPointedType.count(pointerTypeId) > 0
                                   ? pointerToPointedType[pointerTypeId]
                                   : 0;

        if (storageClass == spirv::StorageClassStorageBuffer) {
          info.type = BindingType::StorageBuffer;
        } else if (storageClass == spirv::StorageClassUniform) {
          // Check if the pointed type has BufferBlock decoration (SSBO in GLSL
          // 4.30 / SPIR-V 1.0)
          if (pointedType != 0 && idIsBufferBlock.count(pointedType) > 0) {
            info.type = BindingType::StorageBuffer;
          } else {
            info.type = BindingType::UniformBuffer;
          }
        } else if (storageClass == spirv::StorageClassUniformConstant) {
          // Could be sampler, image, or sampled image
          if (idIsSampler.count(pointedType) > 0) {
            info.type = BindingType::Sampler;
          } else if (idIsSampledImage.count(pointedType) > 0) {
            info.type = BindingType::SampledImage;
          } else if (idIsImage.count(pointedType) > 0) {
            info.type = BindingType::StorageImage;
          } else {
            info.type =
                BindingType::SampledImage; // Default for uniform constant
          }
        } else {
          info.type = BindingType::StorageBuffer; // Default
        }

        // Determine access mode
        bool nonWritable = idIsNonWritable.count(resultId) > 0;
        bool nonReadable = idIsNonReadable.count(resultId) > 0;

        if (nonWritable && !nonReadable) {
          info.access = BindingAccess::ReadOnly;
        } else if (nonReadable && !nonWritable) {
          info.access = BindingAccess::WriteOnly;
        } else if (!nonWritable && !nonReadable) {
          // Default: uniform buffers are read-only, storage buffers are
          // read-write
          if (info.type == BindingType::UniformBuffer ||
              info.type == BindingType::Sampler ||
              info.type == BindingType::SampledImage) {
            info.access = BindingAccess::ReadOnly;
          } else {
            info.access = BindingAccess::ReadWrite;
          }
        } else {
          info.access = BindingAccess::ReadWrite;
        }

        reflection.bindings.emplace_back(info);
      }
    }
    i += wordCount;
  }

  // Sort bindings by set, then by binding index
  std::sort(reflection.bindings.begin(), reflection.bindings.end(),
            [](const BindingInfo &a, const BindingInfo &b) {
              if (a.set != b.set) {
                return a.set < b.set;
              }
              return a.binding < b.binding;
            });

  return reflection;
}

} // namespace cut
