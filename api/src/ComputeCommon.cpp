#include <ComputeCommon.h>

#include <fstream>
#include <string>
#include <unordered_map>

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

// SPIR-V constants
namespace spirv {
constexpr uint32_t MagicNumber = 0x07230203;

// Opcodes
constexpr uint32_t OpDecorate = 71;
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

// Storage classes
constexpr uint32_t StorageClassUniform = 2;
constexpr uint32_t StorageClassUniformConstant = 0;
constexpr uint32_t StorageClassStorageBuffer = 12;
constexpr uint32_t StorageClassPushConstant = 9;
} // namespace spirv

ShaderReflection reflectSpirvBindings(const std::vector<uint32_t> &spirvCode) {
  ShaderReflection reflection{};
  reflection.pushConstantSize = 0;

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
        reflection.bindings.push_back(info);
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

        reflection.bindings.push_back(info);
      }
    }
    i += wordCount;
  }

  return reflection;
}

} // namespace cut
