#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gguf {

// GGUF Magic number: "GGUF" in ASCII (little-endian)
constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t GGUF_VERSION = 3;
constexpr uint32_t DEFAULT_ALIGNMENT = 32;

// GGML tensor data types (quantization formats)
enum class GGMLType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  // Q4_2 = 4,  // Removed
  // Q4_3 = 5,  // Removed
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  // 31-33 reserved
  TQ1_0 = 34,
  TQ2_0 = 35,
  COUNT
};

// GGUF metadata value types
enum class GGUFValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12
};

// Forward declaration for recursive variant
struct GGUFArray;

// Variant type for metadata values
using GGUFValue = std::variant<uint8_t,
                               int8_t,
                               uint16_t,
                               int16_t,
                               uint32_t,
                               int32_t,
                               float,
                               bool,
                               std::string,
                               std::vector<uint8_t>,
                               std::vector<int8_t>,
                               std::vector<uint16_t>,
                               std::vector<int16_t>,
                               std::vector<uint32_t>,
                               std::vector<int32_t>,
                               std::vector<float>,
                               std::vector<bool>,
                               std::vector<std::string>,
                               uint64_t,
                               int64_t,
                               double,
                               std::vector<uint64_t>,
                               std::vector<int64_t>,
                               std::vector<double>>;

// Tensor type information
struct GGMLTypeInfo {
  size_t block_size; // Number of elements per block
  size_t type_size;  // Bytes per block
  const char *name;
};

// Get type info for a GGML type
inline const GGMLTypeInfo &get_type_info(GGMLType type) {
  static const GGMLTypeInfo type_info[] = {
      {1, 4, "F32"},        // F32
      {1, 2, "F16"},        // F16
      {32, 18, "Q4_0"},     // Q4_0
      {32, 20, "Q4_1"},     // Q4_1
      {0, 0, "Q4_2"},       // Q4_2 (removed)
      {0, 0, "Q4_3"},       // Q4_3 (removed)
      {32, 22, "Q5_0"},     // Q5_0
      {32, 24, "Q5_1"},     // Q5_1
      {32, 34, "Q8_0"},     // Q8_0
      {32, 36, "Q8_1"},     // Q8_1
      {256, 84, "Q2_K"},    // Q2_K
      {256, 110, "Q3_K"},   // Q3_K
      {256, 144, "Q4_K"},   // Q4_K
      {256, 176, "Q5_K"},   // Q5_K
      {256, 210, "Q6_K"},   // Q6_K
      {256, 292, "Q8_K"},   // Q8_K
      {256, 66, "IQ2_XXS"}, // IQ2_XXS
      {256, 74, "IQ2_XS"},  // IQ2_XS
      {256, 98, "IQ3_XXS"}, // IQ3_XXS
      {256, 50, "IQ1_S"},   // IQ1_S
      {32, 18, "IQ4_NL"},   // IQ4_NL
      {256, 110, "IQ3_S"},  // IQ3_S
      {256, 82, "IQ2_S"},   // IQ2_S
      {256, 136, "IQ4_XS"}, // IQ4_XS
      {1, 1, "I8"},         // I8
      {1, 2, "I16"},        // I16
      {1, 4, "I32"},        // I32
      {1, 8, "I64"},        // I64
      {1, 8, "F64"},        // F64
      {256, 56, "IQ1_M"},   // IQ1_M
      {1, 2, "BF16"},       // BF16
  };

  auto idx = static_cast<size_t>(type);
  if (idx < sizeof(type_info) / sizeof(type_info[0])) {
    return type_info[idx];
  }
  static const GGMLTypeInfo unknown = {0, 0, "UNKNOWN"};
  return unknown;
}

// Get the name of a GGML type
inline const char *get_type_name(GGMLType type) {
  return get_type_info(type).name;
}

// Calculate the size in bytes for n elements of a given type
inline size_t get_tensor_size(GGMLType type, size_t n_elements) {
  const auto &info = get_type_info(type);
  if (info.block_size == 0)
    return 0;
  size_t n_blocks = (n_elements + info.block_size - 1) / info.block_size;
  return n_blocks * info.type_size;
}

// Tensor information
struct TensorInfo {
  std::string name;
  std::vector<uint64_t> dimensions;
  GGMLType type;
  uint64_t offset; // Offset relative to tensor data section

  size_t n_elements() const {
    size_t n = 1;
    for (auto dim : dimensions) {
      n *= dim;
    }
    return n;
  }

  size_t nbytes() const { return get_tensor_size(type, n_elements()); }
};

// Metadata container
class Metadata {
public:
  void set(const std::string &key, GGUFValue value) {
    data_[key] = std::move(value);
  }

  bool has(const std::string &key) const {
    return data_.find(key) != data_.end();
  }

  const GGUFValue &get(const std::string &key) const { return data_.at(key); }

  template <typename T>
  T get_as(const std::string &key) const {
    return std::get<T>(data_.at(key));
  }

  template <typename T>
  T get_as(const std::string &key, const T &default_value) const {
    auto it = data_.find(key);
    if (it == data_.end())
      return default_value;
    if (auto *val = std::get_if<T>(&it->second)) {
      return *val;
    }
    return default_value;
  }

  const std::unordered_map<std::string, GGUFValue> &items() const {
    return data_;
  }

  // Common metadata accessors
  std::string architecture() const {
    return get_as<std::string>("general.architecture", "");
  }

  std::string name() const { return get_as<std::string>("general.name", ""); }

  uint32_t alignment() const {
    return get_as<uint32_t>("general.alignment", DEFAULT_ALIGNMENT);
  }

private:
  std::unordered_map<std::string, GGUFValue> data_;
};

} // namespace gguf
