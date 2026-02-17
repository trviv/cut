#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <ComputeStructs.h>

using namespace cut;

// ThreadSize tests

TEST(ThreadSize, DefaultConstructorInitializesToZero) {
  ThreadSize tgs;
  EXPECT_EQ(tgs.x, 0);
  EXPECT_EQ(tgs.y, 0);
  EXPECT_EQ(tgs.z, 0);
}

TEST(ThreadSize, AggregateInitialization) {
  ThreadSize tgs{8, 8, 1};
  EXPECT_EQ(tgs.x, 8);
  EXPECT_EQ(tgs.y, 8);
  EXPECT_EQ(tgs.z, 1);
}

TEST(ThreadSize, PartialInitialization) {
  ThreadSize tgs{16};
  EXPECT_EQ(tgs.x, 16);
  EXPECT_EQ(tgs.y, 0);
  EXPECT_EQ(tgs.z, 0);
}

// DataReference tests

TEST(DataReference, ConstructFromPrimitive) {
  int value = 42;
  DataReference ref(value);
  EXPECT_EQ(ref.ptr, &value);
  EXPECT_EQ(ref.size, sizeof(int));
}

TEST(DataReference, ConstructFromStruct) {
  struct TestStruct {
    float x, y, z, w;
  };
  TestStruct data{1.0f, 2.0f, 3.0f, 4.0f};
  DataReference ref(data);
  EXPECT_EQ(ref.ptr, &data);
  EXPECT_EQ(ref.size, sizeof(TestStruct));
}

TEST(DataReference, ConstructFromPointerAndSize) {
  std::vector<uint32_t> data = {1, 2, 3, 4};
  DataReference ref(data.data(),
                    static_cast<uint32_t>(data.size() * sizeof(uint32_t)));
  EXPECT_EQ(ref.ptr, data.data());
  EXPECT_EQ(ref.size, 16);
}

TEST(DataReference, ConstructFromArray) {
  float array[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  DataReference ref(array);
  EXPECT_EQ(ref.ptr, array);
  EXPECT_EQ(ref.size, sizeof(array));
}

// compileShaderToSpirv tests

TEST(CompileShaderToSpirv, CompilesSimpleComputeShader) {
  const std::string source = R"(
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Data {
    float values[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    values[idx] = values[idx] * 2.0;
}
)";

  std::vector<uint32_t> spirv =
      compileShaderToSpirv(source, "test_shader.comp");

  // Verify SPIR-V magic number (0x07230203)
  ASSERT_FALSE(spirv.empty());
  EXPECT_EQ(spirv[0], 0x07230203u);
}

TEST(CompileShaderToSpirv, ThrowsOnInvalidShader) {
  const std::string invalidSource = R"(
#version 450
layout(local_size_x = 64) in;

void main() {
    // Syntax error - missing semicolon
    float x = 1.0
}
)";

  EXPECT_THROW(compileShaderToSpirv(invalidSource, "invalid.comp"),
               std::runtime_error);
}

TEST(CompileShaderToSpirv, ThrowsOnMissingMain) {
  const std::string noMainSource = R"(
#version 450
layout(local_size_x = 64) in;

void notMain() {
    float x = 1.0;
}
)";

  EXPECT_THROW(compileShaderToSpirv(noMainSource, "no_main.comp"),
               std::runtime_error);
}

TEST(CompileShaderToSpirv, CompilesShaderWithPushConstants) {
  const std::string source = R"(
#version 450
layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint offset;
    uint count;
    float scale;
};

layout(set = 0, binding = 0, std430) buffer Data {
    float values[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        values[offset + idx] *= scale;
    }
}
)";

  std::vector<uint32_t> spirv =
      compileShaderToSpirv(source, "push_constants.comp");

  ASSERT_FALSE(spirv.empty());
  EXPECT_EQ(spirv[0], 0x07230203u);
}

TEST(CompileShaderToSpirv, CompilesShaderWithMultipleBindings) {
  const std::string source = R"(
#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer Input1 { float a[]; };
layout(set = 0, binding = 1, std430) readonly buffer Input2 { float b[]; };
layout(set = 0, binding = 2, std430) readonly buffer Input3 { float c[]; };
layout(set = 0, binding = 3, std430) writeonly buffer Output { float out_data[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    out_data[idx] = a[idx] + b[idx] + c[idx];
}
)";

  std::vector<uint32_t> spirv =
      compileShaderToSpirv(source, "multi_binding.comp");

  ASSERT_FALSE(spirv.empty());
  EXPECT_EQ(spirv[0], 0x07230203u);
}

// ComputeData scalar/data routing tests

TEST(ComputeData, SmallDataStoredAsScalar) {
  float value = 3.14f;
  DataReference ref(&value, sizeof(float));
  ComputeData data(ref);

  EXPECT_TRUE(data.isScalar());
  EXPECT_FALSE(data.isData());
  EXPECT_FALSE(data.isHandle());
  EXPECT_FLOAT_EQ(data.getScalar<float>(), 3.14f);
}

TEST(ComputeData, UintScalar) {
  uint32_t value = 42;
  DataReference ref(&value, sizeof(uint32_t));
  ComputeData data(ref);

  EXPECT_TRUE(data.isScalar());
  EXPECT_FALSE(data.isData());
  EXPECT_EQ(data.getScalar<uint32_t>(), 42u);
}

TEST(ComputeData, LargeDataStoredAsData) {
  uint32_t values[3] = {10, 20, 30};
  DataReference ref(values, sizeof(values));
  ComputeData data(ref);

  EXPECT_FALSE(data.isScalar());
  EXPECT_TRUE(data.isData());
  EXPECT_FALSE(data.isHandle());

  const auto &bytes = data.getData();
  EXPECT_EQ(bytes.size(), 3 * sizeof(uint32_t));
  const uint32_t *result = reinterpret_cast<const uint32_t *>(bytes.data());
  EXPECT_EQ(result[0], 10u);
  EXPECT_EQ(result[1], 20u);
  EXPECT_EQ(result[2], 30u);
}

TEST(ComputeBinding, ScalarForwarding) {
  float value = 2.5f;
  ComputeBinding binding(3, DataReference(&value, sizeof(float)));

  EXPECT_EQ(binding.index(), 3u);
  EXPECT_TRUE(binding.isScalar());
  EXPECT_FALSE(binding.isData());
  EXPECT_FLOAT_EQ(binding.getScalar<float>(), 2.5f);
}

TEST(ComputeBinding, DataForwarding) {
  uint32_t values[2] = {100, 200};
  ComputeBinding binding(1, DataReference(values, sizeof(values)));

  EXPECT_EQ(binding.index(), 1u);
  EXPECT_FALSE(binding.isScalar());
  EXPECT_TRUE(binding.isData());

  const auto &bytes = binding.getData();
  EXPECT_EQ(bytes.size(), 2 * sizeof(uint32_t));
}

// compileShaderFileToSpirv tests

TEST(CompileShaderFileToSpirv, ThrowsOnNonexistentFile) {
  EXPECT_THROW(compileShaderFileToSpirv("/nonexistent/path/shader.comp"),
               std::runtime_error);
}

// Integration test: compile and reflect

TEST(ShaderCompileAndReflect, CompiledShaderCanBeReflected) {
  const std::string source = R"(
#version 450
layout(local_size_x = 64) in;

layout(push_constant) uniform PushConstants {
    uint count;
};

layout(set = 0, binding = 0, std430) readonly buffer Input { float in_data[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Output { float out_data[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < count) {
        out_data[idx] = in_data[idx] * 2.0;
    }
}
)";

  std::vector<uint32_t> spirv =
      compileShaderToSpirv(source, "reflect_test.comp");
  ASSERT_FALSE(spirv.empty());

  ShaderReflection reflection = reflectSpirvBindings(spirv);

  // Should have push constant
  EXPECT_GT(reflection.pushConstantSize, 0u);

  // Should have 3 bindings: push constant + 2 storage buffers
  EXPECT_EQ(reflection.bindings.size(), 3u);

  // Find the storage buffer bindings
  int readOnlyCount = 0;
  int writeOnlyCount = 0;
  for (const auto &binding : reflection.bindings) {
    if (binding.type == BindingType::StorageBuffer) {
      if (binding.access == BindingAccess::ReadOnly) {
        readOnlyCount++;
      } else if (binding.access == BindingAccess::WriteOnly) {
        writeOnlyCount++;
      }
    }
  }
  EXPECT_EQ(readOnlyCount, 1);
  EXPECT_EQ(writeOnlyCount, 1);
}
