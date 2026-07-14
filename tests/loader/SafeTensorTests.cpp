#include "safetensor_reader.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

namespace {

// Helper to convert float to BF16
static uint16_t f32ToBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(float));
  uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

// Helper to convert float to F16
static uint16_t f32ToF16(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(float));
  int s = (f >> 16) & 0x8000;
  int e = (f >> 23) & 0xFF;
  int m = f & 0x7FFFFF;

  if (e == 0) {
    // Subnormal or zero
    m >>= 13;
    if (m != 0) {
      // Subnormal, round to nearest even
      m += (m & 1);
    }
    return static_cast<uint16_t>(s | m);
  } else {
    // Normal
    if (e == 0xFF) {
      // Infinity or NaN
      return static_cast<uint16_t>(s | 0x7C00 | (m != 0 ? 0x200 : 0));
    }
    e -= 127;
    m >>= 13;
    if ((m & 1) && ((m & 2) || ((f & 0x7FFFFF) != 0))) {
      m += 1;
    }
    return static_cast<uint16_t>(s | ((e + 15) << 10) | m);
  }
}

static std::string tempFilePath() {
  return std::string(::testing::TempDir()) + "st_test.safetensors";
}

} // namespace

TEST(SafeTensorReaderTest, ReadsHeaderShapesAndData) {
  const std::string path = tempFilePath();

  // Create test data
  std::vector<float> a_data = {1.5f, -2.25f, 0.f, 3.f, 4.5f, -0.5f};
  std::vector<uint16_t> b_data;
  for (float v : {1.0f, -1.0f, 0.5f, 2.0f}) {
    b_data.push_back(f32ToBf16(v));
  }
  std::vector<uint16_t> c_data;
  for (float v : {0.25f, -0.75f, 8.f, -8.f}) {
    c_data.push_back(f32ToF16(v));
  }

  // Build JSON header
  std::string json_header =
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},"c":{"dtype":"F16","shape":[2,2],"data_offsets":[32,40]}})";
  size_t header_size = json_header.size();

  // Write file
  {
    std::ofstream out(path, std::ios::binary);
    // Write header length (8 bytes)
    uint64_t header_len = header_size;
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    // Write JSON header
    out.write(json_header.data(), json_header.size());
    // Write tensor data
    out.write(reinterpret_cast<const char*>(a_data.data()), a_data.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(b_data.data()), b_data.size() * sizeof(uint16_t));
    out.write(reinterpret_cast<const char*>(c_data.data()), c_data.size() * sizeof(uint16_t));
  }

  // Read file
  safetensor::SafeTensorReader reader(path);

  // Check metadata
  EXPECT_EQ(reader.metadata().get("format"), "pt");

  // Check tensors exist
  EXPECT_TRUE(reader.has_tensor("a"));
  EXPECT_TRUE(reader.has_tensor("b"));
  EXPECT_TRUE(reader.has_tensor("c"));
  EXPECT_EQ(reader.get_tensor_names().size(), 3);

  // Check tensor info
  const auto& a_info = reader.get_tensor_info("a");
  EXPECT_EQ(a_info.dtype, safetensor::DType::F32);
  EXPECT_EQ(a_info.shape, (std::vector<size_t>{2, 3}));
  EXPECT_EQ(a_info.data_size, 24u);

  const auto& b_info = reader.get_tensor_info("b");
  EXPECT_EQ(b_info.dtype, safetensor::DType::BF16);

  const auto& c_info = reader.get_tensor_info("c");
  EXPECT_EQ(c_info.dtype, safetensor::DType::F16);

  // Check data
  auto a_read = reader.read_tensor_f32("a");
  EXPECT_EQ(a_read.size(), a_data.size());
  for (size_t i = 0; i < a_data.size(); ++i) {
    EXPECT_FLOAT_EQ(a_read[i], a_data[i]);
  }

  const float b_expected[] = {1.0f, -1.0f, 0.5f, 2.0f};
  auto b_read = reader.read_tensor_f32("b");
  EXPECT_EQ(b_read.size(), b_data.size());
  for (size_t i = 0; i < b_data.size(); ++i) {
    EXPECT_FLOAT_EQ(b_read[i], b_expected[i]);
  }

  const float c_expected[] = {0.25f, -0.75f, 8.f, -8.f};
  auto c_read = reader.read_tensor_f32("c");
  EXPECT_EQ(c_read.size(), c_data.size());
  for (size_t i = 0; i < c_data.size(); ++i) {
    EXPECT_FLOAT_EQ(c_read[i], c_expected[i]);
  }

  // Check raw read
  auto b_raw = reader.read_tensor_raw("b");
  EXPECT_EQ(b_raw.size(), 8);

  // Cleanup
  std::remove(path.c_str());
}

TEST(SafeTensorReaderTest, MissingTensorThrows) {
  const std::string path = tempFilePath();

  // Create minimal test file
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  std::string json_header =
      R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})";
  size_t header_size = json_header.size();

  {
    std::ofstream out(path, std::ios::binary);
    uint64_t header_len = header_size;
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(json_header.data(), json_header.size());
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  }

  safetensor::SafeTensorReader reader(path);

  EXPECT_THROW(reader.read_tensor_f32("nope"), safetensor::SafeTensorReaderError);
  EXPECT_THROW(safetensor::SafeTensorReader("/nonexistent/path.safetensors"), safetensor::SafeTensorReaderError);

  std::remove(path.c_str());
}
