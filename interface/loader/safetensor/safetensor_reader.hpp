#pragma once

#include "safetensor_types.hpp"
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace safetensor {

class SafeTensorReaderError : public std::runtime_error {
public:
  explicit SafeTensorReaderError(const std::string &msg)
      : std::runtime_error(msg) {}
};

class SafeTensorReader {
public:
  explicit SafeTensorReader(const std::string &path);
  ~SafeTensorReader() = default;

  // Non-copyable
  SafeTensorReader(const SafeTensorReader &) = delete;
  SafeTensorReader &operator=(const SafeTensorReader &) = delete;

  // Movable
  SafeTensorReader(SafeTensorReader &&) = default;
  SafeTensorReader &operator=(SafeTensorReader &&) = default;

  // Accessors
  const Metadata &metadata() const { return metadata_; }
  const std::unordered_map<std::string, TensorInfo> &tensors() const {
    return tensors_;
  }

  bool has_tensor(const std::string &name) const {
    return tensors_.find(name) != tensors_.end();
  }

  const TensorInfo &get_tensor_info(const std::string &name) const {
    return tensors_.at(name);
  }

  // Read raw tensor data into provided buffer
  void read_tensor_data(const std::string &name,
                        void *buffer,
                        size_t buffer_size) const;

  // Read tensor data and return as vector
  std::vector<uint8_t> read_tensor_raw(const std::string &name) const;

  // Read tensor and convert to float32
  std::vector<float> read_tensor_f32(const std::string &name) const;

  // Get tensor names
  std::vector<std::string> get_tensor_names() const;

  // File info
  const std::string &path() const { return path_; }
  size_t header_size() const { return header_size_; }
  size_t data_offset() const { return data_offset_; }

private:
  void parse_header();
  void parse_json_header(const std::string &json);

  // JSON parsing helpers (simple implementation)
  static std::string parse_string(const std::string &json, size_t &pos);
  static void skip_whitespace(const std::string &json, size_t &pos);

  // Conversion helpers
  static void
  convert_f16_to_f32(const uint16_t *data, float *output, size_t n_elements);
  static void
  convert_bf16_to_f32(const uint16_t *data, float *output, size_t n_elements);

  std::string path_;
  mutable std::ifstream file_;
  size_t header_size_ = 0;
  size_t data_offset_ = 0;

  Metadata metadata_;
  std::unordered_map<std::string, TensorInfo> tensors_;
};

// Convenience function
inline std::unique_ptr<SafeTensorReader>
load_safetensor(const std::string &path) {
  return std::make_unique<SafeTensorReader>(path);
}

} // namespace safetensor
