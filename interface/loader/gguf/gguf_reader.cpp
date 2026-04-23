#include "gguf_reader.hpp"
#include <algorithm>
#include <cmath>

namespace gguf {

GGUFReader::GGUFReader(const std::string &path) : path_(path) {
  file_.open(path, std::ios::binary);
  if (!file_.is_open()) {
    throw GGUFReaderError("Failed to open file: " + path);
  }
  parse_header();
}

void GGUFReader::parse_header() {
  // Read magic number
  uint32_t magic;
  file_.read(reinterpret_cast<char *>(&magic), sizeof(magic));

  if (magic != GGUF_MAGIC) {
    // Try big-endian
    uint32_t magic_swapped = ((magic >> 24) & 0xFF) | ((magic >> 8) & 0xFF00) |
                             ((magic << 8) & 0xFF0000) |
                             ((magic << 24) & 0xFF000000);
    if (magic_swapped == GGUF_MAGIC) {
      is_big_endian_ = true;
    } else {
      throw GGUFReaderError("Invalid GGUF magic number");
    }
  }

  // Read version
  version_ = read_uint32();
  if (version_ > GGUF_VERSION) {
    throw GGUFReaderError("Unsupported GGUF version: " +
                          std::to_string(version_));
  }

  // Read counts
  uint64_t tensor_count = read_uint64();
  uint64_t metadata_kv_count = read_uint64();

  // Read metadata
  for (uint64_t i = 0; i < metadata_kv_count; ++i) {
    read_metadata_kv();
  }

  // Read tensor infos
  for (uint64_t i = 0; i < tensor_count; ++i) {
    auto info = read_tensor_info();
    tensors_[info.name] = std::move(info);
  }

  // Calculate tensor data offset (aligned)
  uint32_t alignment = metadata_.alignment();
  size_t current_pos = static_cast<size_t>(file_.tellg());
  tensor_data_offset_ = ((current_pos + alignment - 1) / alignment) * alignment;
}

// Byte swapping utilities
template <typename T>
T swap_bytes(T value) {
  static_assert(std::is_integral_v<T>, "swap_bytes requires integral type");
  T result = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    result |= ((value >> (i * 8)) & 0xFF) << ((sizeof(T) - 1 - i) * 8);
  }
  return result;
}

inline float swap_float(float value) {
  uint32_t tmp;
  std::memcpy(&tmp, &value, sizeof(tmp));
  tmp = swap_bytes(tmp);
  std::memcpy(&value, &tmp, sizeof(value));
  return value;
}

inline double swap_double(double value) {
  uint64_t tmp;
  std::memcpy(&tmp, &value, sizeof(tmp));
  tmp = swap_bytes(tmp);
  std::memcpy(&value, &tmp, sizeof(value));
  return value;
}

uint8_t GGUFReader::read_uint8() {
  uint8_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

int8_t GGUFReader::read_int8() {
  int8_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return value;
}

uint16_t GGUFReader::read_uint16() {
  uint16_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

int16_t GGUFReader::read_int16() {
  int16_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

uint32_t GGUFReader::read_uint32() {
  uint32_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

int32_t GGUFReader::read_int32() {
  int32_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

uint64_t GGUFReader::read_uint64() {
  uint64_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

int64_t GGUFReader::read_int64() {
  int64_t value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_bytes(value) : value;
}

float GGUFReader::read_float32() {
  float value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_float(value) : value;
}

double GGUFReader::read_float64() {
  double value;
  file_.read(reinterpret_cast<char *>(&value), sizeof(value));
  return is_big_endian_ ? swap_double(value) : value;
}

bool GGUFReader::read_bool() {
  return read_uint8() != 0;
}

std::string GGUFReader::read_string() {
  uint64_t length = read_uint64();
  std::string result(length, '\0');
  file_.read(result.data(), length);
  return result;
}

GGUFValue GGUFReader::read_value(GGUFValueType type) {
  switch (type) {
  case GGUFValueType::UINT8:
    return read_uint8();
  case GGUFValueType::INT8:
    return read_int8();
  case GGUFValueType::UINT16:
    return read_uint16();
  case GGUFValueType::INT16:
    return read_int16();
  case GGUFValueType::UINT32:
    return read_uint32();
  case GGUFValueType::INT32:
    return read_int32();
  case GGUFValueType::FLOAT32:
    return read_float32();
  case GGUFValueType::BOOL:
    return read_bool();
  case GGUFValueType::STRING:
    return read_string();
  case GGUFValueType::ARRAY:
    return read_array();
  case GGUFValueType::UINT64:
    return read_uint64();
  case GGUFValueType::INT64:
    return read_int64();
  case GGUFValueType::FLOAT64:
    return read_float64();
  default:
    throw GGUFReaderError("Unknown value type: " +
                          std::to_string(static_cast<uint32_t>(type)));
  }
}

GGUFValue GGUFReader::read_array() {
  auto element_type = static_cast<GGUFValueType>(read_uint32());
  uint64_t length = read_uint64();

  switch (element_type) {
  case GGUFValueType::UINT8: {
    std::vector<uint8_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_uint8();
    return arr;
  }
  case GGUFValueType::INT8: {
    std::vector<int8_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_int8();
    return arr;
  }
  case GGUFValueType::UINT16: {
    std::vector<uint16_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_uint16();
    return arr;
  }
  case GGUFValueType::INT16: {
    std::vector<int16_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_int16();
    return arr;
  }
  case GGUFValueType::UINT32: {
    std::vector<uint32_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_uint32();
    return arr;
  }
  case GGUFValueType::INT32: {
    std::vector<int32_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_int32();
    return arr;
  }
  case GGUFValueType::FLOAT32: {
    std::vector<float> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_float32();
    return arr;
  }
  case GGUFValueType::BOOL: {
    std::vector<bool> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_bool();
    return arr;
  }
  case GGUFValueType::STRING: {
    std::vector<std::string> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_string();
    return arr;
  }
  case GGUFValueType::UINT64: {
    std::vector<uint64_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_uint64();
    return arr;
  }
  case GGUFValueType::INT64: {
    std::vector<int64_t> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_int64();
    return arr;
  }
  case GGUFValueType::FLOAT64: {
    std::vector<double> arr(length);
    for (uint64_t i = 0; i < length; ++i)
      arr[i] = read_float64();
    return arr;
  }
  default:
    throw GGUFReaderError("Unsupported array element type");
  }
}

void GGUFReader::read_metadata_kv() {
  std::string key = read_string();
  auto value_type = static_cast<GGUFValueType>(read_uint32());
  GGUFValue value = read_value(value_type);
  metadata_.set(key, std::move(value));
}

TensorInfo GGUFReader::read_tensor_info() {
  TensorInfo info;
  info.name = read_string();
  uint32_t n_dims = read_uint32();
  info.dimensions.resize(n_dims);
  for (uint32_t i = 0; i < n_dims; ++i) {
    info.dimensions[i] = read_uint64();
  }
  info.type = static_cast<GGMLType>(read_uint32());
  info.offset = read_uint64();
  return info;
}

void GGUFReader::read_tensor_data(const std::string &name,
                                  void *buffer,
                                  size_t buffer_size) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw GGUFReaderError("Tensor not found: " + name);
  }

  const auto &info = it->second;
  size_t tensor_size = info.nbytes();

  if (buffer_size < tensor_size) {
    throw GGUFReaderError("Buffer too small for tensor data");
  }

  file_.seekg(tensor_data_offset_ + info.offset);
  file_.read(reinterpret_cast<char *>(buffer), tensor_size);
}

std::vector<uint8_t>
GGUFReader::read_tensor_raw(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw GGUFReaderError("Tensor not found: " + name);
  }

  const auto &info = it->second;
  std::vector<uint8_t> data(info.nbytes());
  read_tensor_data(name, data.data(), data.size());
  return data;
}

std::vector<float> GGUFReader::read_tensor_f32(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw GGUFReaderError("Tensor not found: " + name);
  }

  const auto &info = it->second;
  size_t n_elements = info.n_elements();
  std::vector<float> result(n_elements);

  auto raw_data = read_tensor_raw(name);

  switch (info.type) {
  case GGMLType::F32:
    std::memcpy(result.data(), raw_data.data(), n_elements * sizeof(float));
    break;

  case GGMLType::F16:
    convert_f16_to_f32(reinterpret_cast<const uint16_t *>(raw_data.data()),
                       result.data(), n_elements);
    break;

  case GGMLType::BF16:
    convert_bf16_to_f32(reinterpret_cast<const uint16_t *>(raw_data.data()),
                        result.data(), n_elements);
    break;

  case GGMLType::Q4_0:
    dequantize_q4_0(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q4_1:
    dequantize_q4_1(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q5_0:
    dequantize_q5_0(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q8_0:
    dequantize_q8_0(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q4_K:
    dequantize_q4_k(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q5_K:
    dequantize_q5_k(raw_data.data(), result.data(), n_elements);
    break;

  case GGMLType::Q6_K:
    dequantize_q6_k(raw_data.data(), result.data(), n_elements);
    break;

  default:
    throw GGUFReaderError("Dequantization not implemented for type: " +
                          std::string(get_type_name(info.type)));
  }

  return result;
}

GGUFReader::Q8SeparatedData
GGUFReader::read_tensor_q8_separated(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw GGUFReaderError("Tensor not found: " + name);
  }

  const auto &info = it->second;
  if (info.type != GGMLType::Q8_0) {
    throw GGUFReaderError("Tensor is not Q8_0: " + name);
  }
  if (info.dimensions.size() != 2) {
    throw GGUFReaderError("Q8 separation requires 2D tensor: " + name);
  }

  // GGML convention: dimensions[0] = innermost (cols), dimensions[1] = rows
  uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
  uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);

  if (cols % 32 != 0) {
    throw GGUFReaderError("Q8_0 innermost dimension must be multiple of 32: " +
                          name);
  }

  uint32_t blocks_per_row = cols / 32;
  size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;

  auto raw_data = read_tensor_raw(name);

  Q8SeparatedData result;
  result.rows = rows;
  result.cols = cols;
  result.values.resize(static_cast<size_t>(rows) * cols);
  result.scales.resize(total_blocks);

  // Q8_0 block layout: 2 bytes (f16 scale) + 32 bytes (int8 values) = 34 bytes
  constexpr size_t block_bytes = 34;
  const uint8_t *src = raw_data.data();

  for (size_t block = 0; block < total_blocks; ++block) {
    const uint8_t *block_ptr = src + block * block_bytes;

    // Extract f16 scale (first 2 bytes)
    uint16_t scale_bits;
    std::memcpy(&scale_bits, block_ptr, sizeof(scale_bits));
    result.scales[block] = scale_bits;

    // Extract 32 int8 values (next 32 bytes)
    size_t row = block / blocks_per_row;
    size_t block_in_row = block % blocks_per_row;
    size_t val_offset = row * cols + block_in_row * 32;
    std::memcpy(&result.values[val_offset], block_ptr + 2, 32);
  }

  return result;
}

GGUFReader::Q4SeparatedData
GGUFReader::read_tensor_q4_separated(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw GGUFReaderError("Tensor not found: " + name);
  }

  const auto &info = it->second;
  if (info.type != GGMLType::Q4_0) {
    throw GGUFReaderError("Tensor is not Q4_0: " + name);
  }
  if (info.dimensions.size() != 2) {
    throw GGUFReaderError("Q4 separation requires 2D tensor: " + name);
  }

  // GGML convention: dimensions[0] = innermost (cols), dimensions[1] = rows
  uint32_t cols = static_cast<uint32_t>(info.dimensions[0]);
  uint32_t rows = static_cast<uint32_t>(info.dimensions[1]);

  if (cols % 32 != 0) {
    throw GGUFReaderError("Q4_0 innermost dimension must be multiple of 32: " +
                          name);
  }

  uint32_t blocks_per_row = cols / 32;
  size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;

  auto raw_data = read_tensor_raw(name);

  Q4SeparatedData result;
  result.rows = rows;
  result.cols = cols;
  result.packedValues.resize(static_cast<size_t>(rows) * (cols / 2));
  result.scales.resize(total_blocks);

  // Q4_0 block layout: 2 bytes (f16 scale) + 16 bytes (32 nibbles) = 18 bytes
  constexpr size_t block_bytes = 18;
  const uint8_t *src = raw_data.data();

  for (size_t block = 0; block < total_blocks; ++block) {
    const uint8_t *block_ptr = src + block * block_bytes;

    // Extract f16 scale (first 2 bytes)
    uint16_t scale_bits;
    std::memcpy(&scale_bits, block_ptr, sizeof(scale_bits));
    result.scales[block] = scale_bits;

    // Extract 16 packed bytes (32 nibbles, next 16 bytes)
    size_t row = block / blocks_per_row;
    size_t block_in_row = block % blocks_per_row;
    size_t val_offset = row * (cols / 2) + block_in_row * 16;
    std::memcpy(&result.packedValues[val_offset], block_ptr + 2, 16);
  }

  return result;
}

std::vector<std::string> GGUFReader::get_tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

// Dequantization implementations

/// Converts a float16 value (as uint16_t) to float32, including subnormals.
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h & 0x7C00) >> 10;
  uint32_t mantissa = (h & 0x03FF) << 13;

  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero (positive or negative)
      float result;
      std::memcpy(&result, &sign, sizeof(result));
      return result;
    }
    // Subnormal float16: value = (-1)^sign × mantissa_raw × 2^(-24)
    float result =
        static_cast<float>(h & 0x03FF) * 5.960464477539063e-08f; // 2^(-24)
    return (h & 0x8000) ? -result : result;
  }
  if (exponent == 31) {
    return mantissa ? NAN : ((h & 0x8000) ? -INFINITY : INFINITY);
  }
  uint32_t f32_bits = sign | ((exponent + 112) << 23) | mantissa;
  float result;
  std::memcpy(&result, &f32_bits, sizeof(result));
  return result;
}

void GGUFReader::dequantize_q4_0(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  constexpr size_t block_size = 32;
  constexpr size_t half_block = block_size / 2; // 16
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t offset = 0;

  for (size_t i = 0; i < n_blocks; ++i) {
    // Read scale (float16)
    uint16_t scale_bits;
    std::memcpy(&scale_bits, data + offset, sizeof(scale_bits));
    offset += 2;

    float scale = f16_to_f32(scale_bits);
    size_t base = i * block_size;

    // GGML layout: lower nibbles → first half, upper nibbles → second half
    for (size_t j = 0; j < half_block; ++j) {
      uint8_t byte = data[offset + j];
      int x0 = (byte & 0x0F) - 8;
      int x1 = ((byte >> 4) & 0x0F) - 8;

      if (base + j < n_elements)
        output[base + j] = static_cast<float>(x0) * scale;
      if (base + j + half_block < n_elements)
        output[base + j + half_block] = static_cast<float>(x1) * scale;
    }
    offset += half_block;
  }
}

void GGUFReader::dequantize_q4_1(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  constexpr size_t block_size = 32;
  constexpr size_t half_block = block_size / 2; // 16
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t offset = 0;

  for (size_t i = 0; i < n_blocks; ++i) {
    // Read scale (float16)
    uint16_t scale_bits;
    std::memcpy(&scale_bits, data + offset, sizeof(scale_bits));
    offset += 2;

    // Read min (float16)
    uint16_t min_bits;
    std::memcpy(&min_bits, data + offset, sizeof(min_bits));
    offset += 2;

    float scale = f16_to_f32(scale_bits);
    float min = f16_to_f32(min_bits);
    size_t base = i * block_size;

    // GGML layout: lower nibbles → first half, upper nibbles → second half
    for (size_t j = 0; j < half_block; ++j) {
      uint8_t byte = data[offset + j];
      int x0 = byte & 0x0F;
      int x1 = (byte >> 4) & 0x0F;

      if (base + j < n_elements)
        output[base + j] = static_cast<float>(x0) * scale + min;
      if (base + j + half_block < n_elements)
        output[base + j + half_block] = static_cast<float>(x1) * scale + min;
    }
    offset += half_block;
  }
}

void GGUFReader::dequantize_q5_0(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  constexpr size_t block_size = 32;
  constexpr size_t half_block = block_size / 2; // 16
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t offset = 0;

  for (size_t i = 0; i < n_blocks; ++i) {
    // Read scale (float16). Use memcpy — the data pointer is not aligned.
    uint16_t scale_bits;
    std::memcpy(&scale_bits, data + offset, sizeof(scale_bits));
    offset += 2;

    // Read the 4 high-bit bytes as a single uint32_t (little-endian host assumed).
    uint32_t qh_u32;
    std::memcpy(&qh_u32, data + offset, sizeof(qh_u32));
    offset += 4;

    float scale = f16_to_f32(scale_bits);
    size_t base = i * block_size;

    // GGML Q5_0 packing (same nibble layout as Q4_0):
    //   low nibble  of qs[j] → element (base + j)          for j in [0, 16)
    //   high nibble of qs[j] → element (base + j + 16)     for j in [0, 16)
    // Additionally, bit j of qh_u32 is the 5th (high) bit for element (base + j),
    // bit (j + 16) of qh_u32 is the 5th bit for element (base + j + 16).
    for (size_t j = 0; j < half_block; ++j) {
      uint8_t byte = data[offset + j];
      int low_lo = byte & 0x0F;
      int low_hi = (byte >> 4) & 0x0F;

      int hbit_lo = (qh_u32 >> j) & 1;
      int hbit_hi = (qh_u32 >> (j + half_block)) & 1;

      int q5_lo = low_lo | (hbit_lo << 4);       // 5-bit, range [0, 31]
      int q5_hi = low_hi | (hbit_hi << 4);

      // Center to signed range [-16, 15] and scale.
      if (base + j < n_elements)
        output[base + j] = static_cast<float>(q5_lo - 16) * scale;
      if (base + j + half_block < n_elements)
        output[base + j + half_block] = static_cast<float>(q5_hi - 16) * scale;
    }
    offset += half_block;   // advance past the 16 bytes of qs
  }
}

/// Extract 6-bit sub-block scale and min from the 12-byte scales array.
/// Q4_K packs 8 sub-block scales and 8 sub-block mins into 12 bytes:
///   bytes 0-3:  lower 6 bits of scales for sub-blocks 0-3
///   bytes 4-7:  lower 6 bits of mins for sub-blocks 0-3
///   bytes 8-11: lower 4 bits of scales (lo nibble) and mins (hi nibble)
///               for sub-blocks 4-7, with upper 2 bits from bytes 0-7.
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t &sc, uint8_t &m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void GGUFReader::dequantize_q4_k(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  // Q4_K: 256 elements per super-block, 144 bytes per block.
  // Layout: [d:f16][dmin:f16][scales:12 bytes][qs:128 bytes]
  constexpr size_t block_size = 256;
  constexpr size_t block_bytes = 144;
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t out_idx = 0;
  for (size_t i = 0; i < n_blocks && out_idx < n_elements; ++i) {
    const uint8_t *block = data + i * block_bytes;

    // Super-block scale and minimum (f16)
    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, block, 2);
    std::memcpy(&dmin_bits, block + 2, 2);
    float d = f16_to_f32(d_bits);
    float dmin = f16_to_f32(dmin_bits);

    const uint8_t *scales = block + 4; // 12 bytes of packed sub-block info
    const uint8_t *qs = block + 16;    // 128 bytes of 4-bit quants

    // Process 4 pairs of sub-blocks (each pair = 64 elements, 32 bytes of qs)
    int is = 0;
    for (int j = 0; j < 256 && out_idx < n_elements; j += 64) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, sc1, m1);
      float d1 = d * sc1;
      float dm1 = dmin * m1;
      get_scale_min_k4(is + 1, scales, sc2, m2);
      float d2 = d * sc2;
      float dm2 = dmin * m2;

      // First 32 elements: lower nibble
      for (int l = 0; l < 32 && out_idx < n_elements; ++l) {
        output[out_idx++] = d1 * (qs[l] & 0xF) - dm1;
      }
      // Next 32 elements: upper nibble
      for (int l = 0; l < 32 && out_idx < n_elements; ++l) {
        output[out_idx++] = d2 * ((qs[l] >> 4) & 0xF) - dm2;
      }
      qs += 32;
      is += 2;
    }
  }
}

void GGUFReader::dequantize_q5_k(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  // Q5_K: 256 elements per super-block, 176 bytes per block.
  // Layout: [d:f16][dmin:f16][scales:12][qh:32][qs:128]
  constexpr size_t block_size = 256;
  constexpr size_t block_bytes = 176;
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t out_idx = 0;
  for (size_t i = 0; i < n_blocks && out_idx < n_elements; ++i) {
    const uint8_t *block = data + i * block_bytes;

    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, block, 2);
    std::memcpy(&dmin_bits, block + 2, 2);
    float d = f16_to_f32(d_bits);
    float dmin = f16_to_f32(dmin_bits);

    const uint8_t *scales = block + 4; // 12 bytes
    const uint8_t *qh = block + 16;    // 32 bytes (high bits)
    const uint8_t *ql = block + 48;    // 128 bytes (low 4 bits)

    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256 && out_idx < n_elements; j += 64) {
      uint8_t sc1, m1, sc2, m2;
      get_scale_min_k4(is, scales, sc1, m1);
      float d1 = d * sc1;
      float dm1 = dmin * m1;
      get_scale_min_k4(is + 1, scales, sc2, m2);
      float d2 = d * sc2;
      float dm2 = dmin * m2;

      for (int l = 0; l < 32 && out_idx < n_elements; ++l) {
        output[out_idx++] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - dm1;
      }
      for (int l = 0; l < 32 && out_idx < n_elements; ++l) {
        output[out_idx++] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - dm2;
      }
      ql += 32;
      u1 <<= 2;
      u2 <<= 2;
      is += 2;
    }
  }
}

void GGUFReader::dequantize_q6_k(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  // Q6_K: 256 elements per super-block, 210 bytes per block.
  // Layout: [ql:128][qh:64][scales:16][d:f16]
  constexpr size_t block_size = 256;
  constexpr size_t block_bytes = 210;
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t out_idx = 0;
  for (size_t i = 0; i < n_blocks && out_idx < n_elements; ++i) {
    const uint8_t *block = data + i * block_bytes;

    const uint8_t *ql = block;       // 128 bytes (lower 4 bits)
    const uint8_t *qh = block + 128; // 64 bytes (upper 2 bits)
    const int8_t *sc =
        reinterpret_cast<const int8_t *>(block + 192); // 16 bytes
    uint16_t d_bits;
    std::memcpy(&d_bits, block + 208, 2);
    float d = f16_to_f32(d_bits);

    // Process 2 groups of 128 elements each
    for (int n = 0; n < 256 && out_idx < n_elements; n += 128) {
      for (int l = 0; l < 32 && out_idx + 96 + l <= n_elements; ++l) {
        int is = n / 16 + l / 16;
        int8_t q1 =
            static_cast<int8_t>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) |
                                        (((qh[l] >> 2) & 3) << 4)) -
                    32;
        int8_t q3 =
            static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 =
            static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
            32;
        output[out_idx + l] = d * sc[is + 0] * q1;
        output[out_idx + l + 32] = d * sc[is + 2] * q2;
        output[out_idx + l + 64] = d * sc[is + 4] * q3;
        output[out_idx + l + 96] = d * sc[is + 6] * q4;
      }
      out_idx += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void GGUFReader::dequantize_q8_0(const uint8_t *data,
                                 float *output,
                                 size_t n_elements) {
  constexpr size_t block_size = 32;
  size_t n_blocks = (n_elements + block_size - 1) / block_size;

  size_t offset = 0;
  size_t out_idx = 0;

  for (size_t i = 0; i < n_blocks && out_idx < n_elements; ++i) {
    // Read scale (float16)
    uint16_t scale_bits;
    std::memcpy(&scale_bits, data + offset, sizeof(scale_bits));
    offset += 2;

    float scale = f16_to_f32(scale_bits);

    // Read and dequantize 32 int8 values
    for (size_t j = 0; j < 32 && out_idx < n_elements; ++j) {
      int8_t val = static_cast<int8_t>(data[offset + j]);
      output[out_idx++] = static_cast<float>(val) * scale;
    }
    offset += 32;
  }
}

void GGUFReader::convert_f16_to_f32(const uint16_t *data,
                                    float *output,
                                    size_t n_elements) {
  for (size_t i = 0; i < n_elements; ++i) {
    output[i] = f16_to_f32(data[i]);
  }
}

void GGUFReader::convert_bf16_to_f32(const uint16_t *data,
                                     float *output,
                                     size_t n_elements) {
  for (size_t i = 0; i < n_elements; ++i) {
    // BF16 is just the upper 16 bits of float32
    uint32_t f32_bits = static_cast<uint32_t>(data[i]) << 16;
    std::memcpy(&output[i], &f32_bits, sizeof(float));
  }
}

} // namespace gguf
