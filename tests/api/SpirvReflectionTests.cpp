#include <gtest/gtest.h>

#include <ComputeCommon.h>

using namespace cut;

// SPIR-V constants for building test binaries
namespace spirv {
constexpr uint32_t MagicNumber = 0x07230203;
constexpr uint32_t Version = 0x00010000; // SPIR-V 1.0

// Opcodes (word count << 16 | opcode)
constexpr uint32_t makeOp(uint32_t wordCount, uint32_t opcode) {
  return (wordCount << 16) | opcode;
}

// Opcodes
constexpr uint32_t OpCapability = 17;
constexpr uint32_t OpMemoryModel = 14;
constexpr uint32_t OpEntryPoint = 15;
constexpr uint32_t OpDecorate = 71;
constexpr uint32_t OpTypeVoid = 19;
constexpr uint32_t OpTypeInt = 21;
constexpr uint32_t OpTypeFloat = 22;
constexpr uint32_t OpTypeStruct = 30;
constexpr uint32_t OpTypePointer = 32;
constexpr uint32_t OpTypeImage = 25;
constexpr uint32_t OpTypeSampler = 26;
constexpr uint32_t OpTypeSampledImage = 27;
constexpr uint32_t OpVariable = 59;
constexpr uint32_t OpTypeFunction = 33;
constexpr uint32_t OpFunction = 54;
constexpr uint32_t OpFunctionEnd = 56;
constexpr uint32_t OpLabel = 248;
constexpr uint32_t OpReturn = 253;

// Decorations
constexpr uint32_t DecorationBinding = 33;
constexpr uint32_t DecorationDescriptorSet = 34;
constexpr uint32_t DecorationNonWritable = 24;
constexpr uint32_t DecorationNonReadable = 25;
constexpr uint32_t DecorationBlock = 2;
constexpr uint32_t DecorationBufferBlock = 3;

// Storage classes
constexpr uint32_t StorageClassUniformConstant = 0;
constexpr uint32_t StorageClassUniform = 2;
constexpr uint32_t StorageClassStorageBuffer = 12;
constexpr uint32_t StorageClassPushConstant = 9;
constexpr uint32_t StorageClassFunction = 7;

// Capabilities
constexpr uint32_t CapabilityShader = 1;

// Addressing/Memory models
constexpr uint32_t AddressingModelLogical = 0;
constexpr uint32_t MemoryModelGLSL450 = 1;

// Execution model
constexpr uint32_t ExecutionModelGLCompute = 5;

// Image dimensionality
constexpr uint32_t Dim2D = 1;
} // namespace spirv

class SpirvReflectionTest : public ::testing::Test {
protected:
  // Helper to build a minimal SPIR-V header
  std::vector<uint32_t> makeHeader(uint32_t idBound) {
    return {
        spirv::MagicNumber,
        spirv::Version,
        0, // Generator
        idBound,
        0 // Reserved
    };
  }

  // Helper to add OpCapability Shader
  void addCapability(std::vector<uint32_t> &code) {
    code.push_back(spirv::makeOp(2, spirv::OpCapability));
    code.push_back(spirv::CapabilityShader);
  }

  // Helper to add OpMemoryModel
  void addMemoryModel(std::vector<uint32_t> &code) {
    code.push_back(spirv::makeOp(3, spirv::OpMemoryModel));
    code.push_back(spirv::AddressingModelLogical);
    code.push_back(spirv::MemoryModelGLSL450);
  }

  // Helper to add OpDecorate with binding
  void addBindingDecoration(std::vector<uint32_t> &code, uint32_t targetId,
                            uint32_t binding) {
    code.push_back(spirv::makeOp(4, spirv::OpDecorate));
    code.push_back(targetId);
    code.push_back(spirv::DecorationBinding);
    code.push_back(binding);
  }

  // Helper to add OpDecorate with descriptor set
  void addDescriptorSetDecoration(std::vector<uint32_t> &code,
                                  uint32_t targetId, uint32_t set) {
    code.push_back(spirv::makeOp(4, spirv::OpDecorate));
    code.push_back(targetId);
    code.push_back(spirv::DecorationDescriptorSet);
    code.push_back(set);
  }

  // Helper to add NonWritable decoration
  void addNonWritableDecoration(std::vector<uint32_t> &code,
                                uint32_t targetId) {
    code.push_back(spirv::makeOp(3, spirv::OpDecorate));
    code.push_back(targetId);
    code.push_back(spirv::DecorationNonWritable);
  }

  // Helper to add NonReadable decoration
  void addNonReadableDecoration(std::vector<uint32_t> &code,
                                uint32_t targetId) {
    code.push_back(spirv::makeOp(3, spirv::OpDecorate));
    code.push_back(targetId);
    code.push_back(spirv::DecorationNonReadable);
  }

  // Helper to add Block decoration
  void addBlockDecoration(std::vector<uint32_t> &code, uint32_t targetId) {
    code.push_back(spirv::makeOp(3, spirv::OpDecorate));
    code.push_back(targetId);
    code.push_back(spirv::DecorationBlock);
  }

  // Helper to add OpTypeVoid
  void addTypeVoid(std::vector<uint32_t> &code, uint32_t resultId) {
    code.push_back(spirv::makeOp(2, spirv::OpTypeVoid));
    code.push_back(resultId);
  }

  // Helper to add OpTypeInt
  void addTypeInt(std::vector<uint32_t> &code, uint32_t resultId,
                  uint32_t width = 32, uint32_t signedness = 0) {
    code.push_back(spirv::makeOp(4, spirv::OpTypeInt));
    code.push_back(resultId);
    code.push_back(width);
    code.push_back(signedness);
  }

  // Helper to add OpTypeFloat
  void addTypeFloat(std::vector<uint32_t> &code, uint32_t resultId,
                    uint32_t width = 32) {
    code.push_back(spirv::makeOp(3, spirv::OpTypeFloat));
    code.push_back(resultId);
    code.push_back(width);
  }

  // Helper to add OpTypeStruct
  void addTypeStruct(std::vector<uint32_t> &code, uint32_t resultId,
                     const std::vector<uint32_t> &memberTypes) {
    code.push_back(
        spirv::makeOp(2 + memberTypes.size(), spirv::OpTypeStruct));
    code.push_back(resultId);
    for (uint32_t member : memberTypes) {
      code.push_back(member);
    }
  }

  // Helper to add OpTypePointer
  void addTypePointer(std::vector<uint32_t> &code, uint32_t resultId,
                      uint32_t storageClass, uint32_t pointedType) {
    code.push_back(spirv::makeOp(4, spirv::OpTypePointer));
    code.push_back(resultId);
    code.push_back(storageClass);
    code.push_back(pointedType);
  }

  // Helper to add OpTypeSampler
  void addTypeSampler(std::vector<uint32_t> &code, uint32_t resultId) {
    code.push_back(spirv::makeOp(2, spirv::OpTypeSampler));
    code.push_back(resultId);
  }

  // Helper to add OpTypeImage
  void addTypeImage(std::vector<uint32_t> &code, uint32_t resultId,
                    uint32_t sampledType) {
    // OpTypeImage %result %sampledType Dim Depth Arrayed MS Sampled Format
    code.push_back(spirv::makeOp(9, spirv::OpTypeImage));
    code.push_back(resultId);
    code.push_back(sampledType);   // Sampled type (e.g., float)
    code.push_back(spirv::Dim2D);  // Dim
    code.push_back(0);             // Depth (0 = not depth)
    code.push_back(0);             // Arrayed (0 = not arrayed)
    code.push_back(0);             // MS (0 = single-sampled)
    code.push_back(1);             // Sampled (1 = used with sampler)
    code.push_back(0);             // Format (Unknown)
  }

  // Helper to add OpTypeSampledImage
  void addTypeSampledImage(std::vector<uint32_t> &code, uint32_t resultId,
                           uint32_t imageType) {
    code.push_back(spirv::makeOp(3, spirv::OpTypeSampledImage));
    code.push_back(resultId);
    code.push_back(imageType);
  }

  // Helper to add OpVariable
  void addVariable(std::vector<uint32_t> &code, uint32_t pointerTypeId,
                   uint32_t resultId, uint32_t storageClass) {
    code.push_back(spirv::makeOp(4, spirv::OpVariable));
    code.push_back(pointerTypeId);
    code.push_back(resultId);
    code.push_back(storageClass);
  }
};

// Test: Empty SPIR-V throws exception
TEST_F(SpirvReflectionTest, EmptySpirv) {
  std::vector<uint32_t> emptyCode;
  EXPECT_THROW(reflectSpirvBindings(emptyCode),
               std::runtime_error);
}

// Test: Invalid magic number throws exception
TEST_F(SpirvReflectionTest, InvalidMagicNumber) {
  std::vector<uint32_t> invalidCode = {0xDEADBEEF, 0, 0, 10, 0};
  EXPECT_THROW(reflectSpirvBindings(invalidCode),
               std::runtime_error);
}

// Test: Too small SPIR-V throws exception
TEST_F(SpirvReflectionTest, TooSmallSpirv) {
  std::vector<uint32_t> tooSmall = {spirv::MagicNumber, 0, 0};
  EXPECT_THROW(reflectSpirvBindings(tooSmall),
               std::runtime_error);
}

// Test: Single storage buffer binding
TEST_F(SpirvReflectionTest, SingleStorageBuffer) {
  // IDs: 1=void, 2=int, 3=struct, 4=pointer, 5=variable
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 5, 0);
  addDescriptorSetDecoration(code, 5, 0);
  addBlockDecoration(code, 3);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2}); // struct { int }
  addTypePointer(code, 4, spirv::StorageClassStorageBuffer, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassStorageBuffer);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 0);
  EXPECT_EQ(reflection.bindings[0].set, 0);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::StorageBuffer);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadWrite);
}

// Test: Uniform buffer binding
TEST_F(SpirvReflectionTest, UniformBuffer) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 5, 1);
  addDescriptorSetDecoration(code, 5, 0);
  addBlockDecoration(code, 3);

  // Types
  addTypeVoid(code, 1);
  addTypeFloat(code, 2);
  addTypeStruct(code, 3, {2}); // struct { float }
  addTypePointer(code, 4, spirv::StorageClassUniform, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassUniform);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 1);
  EXPECT_EQ(reflection.bindings[0].set, 0);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::UniformBuffer);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadOnly);
}

// Test: Multiple bindings with different sets
TEST_F(SpirvReflectionTest, MultipleBindingsDifferentSets) {
  auto code = makeHeader(15);
  addCapability(code);
  addMemoryModel(code);

  // Decorations for first variable (set 0, binding 0)
  addBindingDecoration(code, 7, 0);
  addDescriptorSetDecoration(code, 7, 0);

  // Decorations for second variable (set 1, binding 2)
  addBindingDecoration(code, 8, 2);
  addDescriptorSetDecoration(code, 8, 1);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2});
  addTypeStruct(code, 4, {2});
  addTypePointer(code, 5, spirv::StorageClassStorageBuffer, 3);
  addTypePointer(code, 6, spirv::StorageClassStorageBuffer, 4);

  // Variables
  addVariable(code, 5, 7, spirv::StorageClassStorageBuffer);
  addVariable(code, 6, 8, spirv::StorageClassStorageBuffer);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 2);

  // Find binding at set 0
  auto it0 = std::find_if(reflection.bindings.begin(), reflection.bindings.end(),
                          [](const BindingInfo &b) { return b.set == 0; });
  ASSERT_NE(it0, reflection.bindings.end());
  EXPECT_EQ(it0->binding, 0);
  EXPECT_EQ(it0->type, BindingType::StorageBuffer);

  // Find binding at set 1
  auto it1 = std::find_if(reflection.bindings.begin(), reflection.bindings.end(),
                          [](const BindingInfo &b) { return b.set == 1; });
  ASSERT_NE(it1, reflection.bindings.end());
  EXPECT_EQ(it1->binding, 2);
  EXPECT_EQ(it1->type, BindingType::StorageBuffer);
}

// Test: Read-only storage buffer (NonWritable decoration)
TEST_F(SpirvReflectionTest, ReadOnlyStorageBuffer) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 5, 0);
  addDescriptorSetDecoration(code, 5, 0);
  addNonWritableDecoration(code, 5);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2});
  addTypePointer(code, 4, spirv::StorageClassStorageBuffer, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassStorageBuffer);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadOnly);
}

// Test: Write-only storage buffer (NonReadable decoration)
TEST_F(SpirvReflectionTest, WriteOnlyStorageBuffer) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 5, 0);
  addDescriptorSetDecoration(code, 5, 0);
  addNonReadableDecoration(code, 5);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2});
  addTypePointer(code, 4, spirv::StorageClassStorageBuffer, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassStorageBuffer);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::WriteOnly);
}

// Test: Push constant detection
TEST_F(SpirvReflectionTest, PushConstant) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // No binding decoration for push constants

  // Types
  addTypeVoid(code, 1);
  addTypeFloat(code, 2);
  addTypeStruct(code, 3, {2, 2}); // struct { float, float }
  addTypePointer(code, 4, spirv::StorageClassPushConstant, 3);

  // Variable (push constant)
  addVariable(code, 4, 5, spirv::StorageClassPushConstant);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::PushConstant);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadOnly);
}

// Test: Sampler binding
TEST_F(SpirvReflectionTest, SamplerBinding) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 4, 0);
  addDescriptorSetDecoration(code, 4, 0);

  // Types
  addTypeVoid(code, 1);
  addTypeSampler(code, 2);
  addTypePointer(code, 3, spirv::StorageClassUniformConstant, 2);

  // Variable
  addVariable(code, 3, 4, spirv::StorageClassUniformConstant);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 0);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::Sampler);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadOnly);
}

// Test: Sampled image binding
TEST_F(SpirvReflectionTest, SampledImageBinding) {
  auto code = makeHeader(12);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 6, 1);
  addDescriptorSetDecoration(code, 6, 0);

  // Types
  addTypeVoid(code, 1);
  addTypeFloat(code, 2);
  addTypeImage(code, 3, 2);       // image2D<float>
  addTypeSampledImage(code, 4, 3); // sampler2D
  addTypePointer(code, 5, spirv::StorageClassUniformConstant, 4);

  // Variable
  addVariable(code, 5, 6, spirv::StorageClassUniformConstant);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 1);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::SampledImage);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadOnly);
}

// Test: Storage image binding
TEST_F(SpirvReflectionTest, StorageImageBinding) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Decorations
  addBindingDecoration(code, 5, 0);
  addDescriptorSetDecoration(code, 5, 0);

  // Types
  addTypeVoid(code, 1);
  addTypeFloat(code, 2);
  addTypeImage(code, 3, 2); // storage image
  addTypePointer(code, 4, spirv::StorageClassUniformConstant, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassUniformConstant);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 0);
  EXPECT_EQ(reflection.bindings[0].type, BindingType::StorageImage);
  EXPECT_EQ(reflection.bindings[0].access, BindingAccess::ReadWrite);
}

// Test: Variables without binding decoration are ignored
TEST_F(SpirvReflectionTest, UnboundVariablesIgnored) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Only add binding for one variable
  addBindingDecoration(code, 6, 0);
  addDescriptorSetDecoration(code, 6, 0);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2});
  addTypePointer(code, 4, spirv::StorageClassStorageBuffer, 3);
  addTypePointer(code, 5, spirv::StorageClassFunction, 2); // Function-local

  // Variables - one bound, one not
  addVariable(code, 4, 6, spirv::StorageClassStorageBuffer);
  addVariable(code, 5, 7, spirv::StorageClassFunction); // No binding

  auto reflection = reflectSpirvBindings(code);

  // Should only have one binding (the storage buffer)
  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 0);
}

// Test: Default descriptor set is 0 when not specified
TEST_F(SpirvReflectionTest, DefaultDescriptorSet) {
  auto code = makeHeader(10);
  addCapability(code);
  addMemoryModel(code);

  // Only binding, no descriptor set decoration
  addBindingDecoration(code, 5, 3);

  // Types
  addTypeVoid(code, 1);
  addTypeInt(code, 2);
  addTypeStruct(code, 3, {2});
  addTypePointer(code, 4, spirv::StorageClassStorageBuffer, 3);

  // Variable
  addVariable(code, 4, 5, spirv::StorageClassStorageBuffer);

  auto reflection = reflectSpirvBindings(code);

  ASSERT_EQ(reflection.bindings.size(), 1);
  EXPECT_EQ(reflection.bindings[0].binding, 3);
  EXPECT_EQ(reflection.bindings[0].set, 0); // Default
}
