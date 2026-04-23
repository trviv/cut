#pragma once

#include "gguf_types.hpp"
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace gguf {

class GGUFReaderError : public std::runtime_error {
public:
  explicit GGUFReaderError(const std::string &msg) : std::runtime_error(msg) {}
};

class GGUFReader {
public:
  explicit GGUFReader(const std::string &path);
  ~GGUFReader() = default;

  // Non-copyable
  GGUFReader(const GGUFReader &) = delete;
  GGUFReader &operator=(const GGUFReader &) = delete;

  // Movable
  GGUFReader(GGUFReader &&) = default;
  GGUFReader &operator=(GGUFReader &&) = default;

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

  // Read tensor and dequantize to float32
  std::vector<float> read_tensor_f32(const std::string &name) const;

  // Q8_0 separated data: int8 values and f16 scales stored separately
  struct Q8SeparatedData {
    std::vector<int8_t> values;   // raw int8 values [rows * cols]
    std::vector<uint16_t> scales; // raw f16 scale bits [rows * (cols/32)]
    uint32_t rows;
    uint32_t cols;
  };

  // Read Q8_0 tensor and separate into int8 values and f16 scales
  Q8SeparatedData read_tensor_q8_separated(const std::string &name) const;

  // Q4_0 separated data: packed nibbles and f16 scales stored separately
  struct Q4SeparatedData {
    std::vector<uint8_t> packedValues; // packed nibbles [rows * (cols/2)]
    std::vector<uint16_t> scales;      // raw f16 scale bits [rows * (cols/32)]
    uint32_t rows;
    uint32_t cols;
  };

  // Read Q4_0 tensor and separate into packed nibbles and f16 scales
  Q4SeparatedData read_tensor_q4_separated(const std::string &name) const;

  // Get tensor names
  std::vector<std::string> get_tensor_names() const;

  // File info
  const std::string &path() const { return path_; }
  uint32_t version() const { return version_; }
  bool is_big_endian() const { return is_big_endian_; }
  size_t tensor_data_offset() const { return tensor_data_offset_; }

private:
  void parse_header();

  // Reading primitives
  uint8_t read_uint8();
  int8_t read_int8();
  uint16_t read_uint16();
  int16_t read_int16();
  uint32_t read_uint32();
  int32_t read_int32();
  uint64_t read_uint64();
  int64_t read_int64();
  float read_float32();
  double read_float64();
  bool read_bool();
  std::string read_string();

  GGUFValue read_value(GGUFValueType type);
  GGUFValue read_array();
  void read_metadata_kv();
  TensorInfo read_tensor_info();

  // Dequantization helpers
  static void
  dequantize_q4_0(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q4_1(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q5_0(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q4_k(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q5_k(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q6_k(const uint8_t *data, float *output, size_t n_elements);
  static void
  dequantize_q8_0(const uint8_t *data, float *output, size_t n_elements);
  static void
  convert_f16_to_f32(const uint16_t *data, float *output, size_t n_elements);
  static void
  convert_bf16_to_f32(const uint16_t *data, float *output, size_t n_elements);

  std::string path_;
  mutable std::ifstream file_;
  bool is_big_endian_ = false;
  uint32_t version_ = 0;
  size_t tensor_data_offset_ = 0;

  Metadata metadata_;
  std::unordered_map<std::string, TensorInfo> tensors_;
};

// Convenience function
inline std::unique_ptr<GGUFReader> load_gguf(const std::string &path) {
  return std::make_unique<GGUFReader>(path);
}

} // namespace gguf
