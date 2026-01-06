#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace safetensor {

// SafeTensor data types
enum class DType {
  BOOL,
  U8,
  I8,
  U16,
  I16,
  U32,
  I32,
  U64,
  I64,
  F16,
  BF16,
  F32,
  F64,
  UNKNOWN
};

// Convert string dtype to enum
inline DType dtype_from_string(const std::string &s) {
  if (s == "BOOL")
    return DType::BOOL;
  if (s == "U8")
    return DType::U8;
  if (s == "I8")
    return DType::I8;
  if (s == "U16")
    return DType::U16;
  if (s == "I16")
    return DType::I16;
  if (s == "U32")
    return DType::U32;
  if (s == "I32")
    return DType::I32;
  if (s == "U64")
    return DType::U64;
  if (s == "I64")
    return DType::I64;
  if (s == "F16")
    return DType::F16;
  if (s == "BF16")
    return DType::BF16;
  if (s == "F32")
    return DType::F32;
  if (s == "F64")
    return DType::F64;
  return DType::UNKNOWN;
}

// Get dtype name
inline const char *dtype_name(DType dtype) {
  switch (dtype) {
  case DType::BOOL:
    return "BOOL";
  case DType::U8:
    return "U8";
  case DType::I8:
    return "I8";
  case DType::U16:
    return "U16";
  case DType::I16:
    return "I16";
  case DType::U32:
    return "U32";
  case DType::I32:
    return "I32";
  case DType::U64:
    return "U64";
  case DType::I64:
    return "I64";
  case DType::F16:
    return "F16";
  case DType::BF16:
    return "BF16";
  case DType::F32:
    return "F32";
  case DType::F64:
    return "F64";
  default:
    return "UNKNOWN";
  }
}

// Get dtype size in bytes
inline size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::BOOL:
  case DType::U8:
  case DType::I8:
    return 1;
  case DType::U16:
  case DType::I16:
  case DType::F16:
  case DType::BF16:
    return 2;
  case DType::U32:
  case DType::I32:
  case DType::F32:
    return 4;
  case DType::U64:
  case DType::I64:
  case DType::F64:
    return 8;
  default:
    return 0;
  }
}

// Tensor information
struct TensorInfo {
  std::string name;
  DType dtype;
  std::vector<size_t> shape;
  size_t data_offset; // Offset in data section
  size_t data_size;   // Size in bytes

  size_t n_elements() const {
    size_t n = 1;
    for (auto dim : shape) {
      n *= dim;
    }
    return n;
  }

  size_t nbytes() const { return n_elements() * dtype_size(dtype); }
};

// Metadata container
class Metadata {
public:
  void set(const std::string &key, const std::string &value) {
    data_[key] = value;
  }

  bool has(const std::string &key) const {
    return data_.find(key) != data_.end();
  }

  const std::string &get(const std::string &key) const { return data_.at(key); }

  const std::string &get(const std::string &key,
                         const std::string &default_value) const {
    auto it = data_.find(key);
    return it != data_.end() ? it->second : default_value;
  }

  const std::unordered_map<std::string, std::string> &items() const {
    return data_;
  }

private:
  std::unordered_map<std::string, std::string> data_;
};

} // namespace safetensor
