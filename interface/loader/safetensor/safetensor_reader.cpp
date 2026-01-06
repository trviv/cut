#include "safetensor_reader.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace safetensor {

SafeTensorReader::SafeTensorReader(const std::string &path) : path_(path) {
  file_.open(path, std::ios::binary);
  if (!file_.is_open()) {
    throw SafeTensorReaderError("Failed to open file: " + path);
  }
  parse_header();
}

void SafeTensorReader::parse_header() {
  // Read header size (8 bytes, little-endian uint64)
  uint64_t header_len;
  file_.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  if (!file_) {
    throw SafeTensorReaderError("Failed to read header size");
  }

  header_size_ = header_len;
  data_offset_ = 8 + header_len;

  // Read JSON header
  std::string json(header_len, '\0');
  file_.read(&json[0], header_len);
  if (!file_) {
    throw SafeTensorReaderError("Failed to read header JSON");
  }

  parse_json_header(json);
}

void SafeTensorReader::skip_whitespace(const std::string &json, size_t &pos) {
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                               json[pos] == '\n' || json[pos] == '\r')) {
    ++pos;
  }
}

std::string SafeTensorReader::parse_string(const std::string &json,
                                           size_t &pos) {
  if (json[pos] != '"') {
    throw SafeTensorReaderError("Expected string at position " +
                                std::to_string(pos));
  }
  ++pos;

  std::string result;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      ++pos;
      switch (json[pos]) {
      case 'n':
        result += '\n';
        break;
      case 't':
        result += '\t';
        break;
      case 'r':
        result += '\r';
        break;
      case '"':
        result += '"';
        break;
      case '\\':
        result += '\\';
        break;
      default:
        result += json[pos];
        break;
      }
    } else {
      result += json[pos];
    }
    ++pos;
  }

  if (pos < json.size() && json[pos] == '"') {
    ++pos;
  }

  return result;
}

void SafeTensorReader::parse_json_header(const std::string &json) {
  size_t pos = 0;
  skip_whitespace(json, pos);

  if (json[pos] != '{') {
    throw SafeTensorReaderError("Expected JSON object");
  }
  ++pos;

  while (pos < json.size()) {
    skip_whitespace(json, pos);

    if (json[pos] == '}') {
      break;
    }

    if (json[pos] == ',') {
      ++pos;
      continue;
    }

    // Parse key
    std::string key = parse_string(json, pos);
    skip_whitespace(json, pos);

    if (json[pos] != ':') {
      throw SafeTensorReaderError("Expected ':' after key");
    }
    ++pos;
    skip_whitespace(json, pos);

    // Check if this is __metadata__ or a tensor
    if (key == "__metadata__") {
      // Parse metadata object
      if (json[pos] != '{') {
        throw SafeTensorReaderError("Expected object for __metadata__");
      }
      ++pos;

      while (pos < json.size() && json[pos] != '}') {
        skip_whitespace(json, pos);
        if (json[pos] == ',' || json[pos] == '}') {
          if (json[pos] == ',')
            ++pos;
          continue;
        }

        std::string meta_key = parse_string(json, pos);
        skip_whitespace(json, pos);
        if (json[pos] == ':')
          ++pos;
        skip_whitespace(json, pos);
        std::string meta_value = parse_string(json, pos);
        metadata_.set(meta_key, meta_value);
        skip_whitespace(json, pos);
      }
      if (json[pos] == '}')
        ++pos;
    } else {
      // Parse tensor info object
      TensorInfo tensor;
      tensor.name = key;

      if (json[pos] != '{') {
        throw SafeTensorReaderError("Expected object for tensor " + key);
      }
      ++pos;

      while (pos < json.size() && json[pos] != '}') {
        skip_whitespace(json, pos);
        if (json[pos] == ',' || json[pos] == '}') {
          if (json[pos] == ',')
            ++pos;
          continue;
        }

        std::string field = parse_string(json, pos);
        skip_whitespace(json, pos);
        if (json[pos] == ':')
          ++pos;
        skip_whitespace(json, pos);

        if (field == "dtype") {
          std::string dtype_str = parse_string(json, pos);
          tensor.dtype = dtype_from_string(dtype_str);
        } else if (field == "shape") {
          // Parse array of integers
          if (json[pos] != '[') {
            throw SafeTensorReaderError("Expected array for shape");
          }
          ++pos;

          while (pos < json.size() && json[pos] != ']') {
            skip_whitespace(json, pos);
            if (json[pos] == ',' || json[pos] == ']') {
              if (json[pos] == ',')
                ++pos;
              continue;
            }

            // Parse integer
            size_t num_start = pos;
            while (pos < json.size() &&
                   (json[pos] >= '0' && json[pos] <= '9')) {
              ++pos;
            }
            if (pos > num_start) {
              tensor.shape.push_back(
                  std::stoull(json.substr(num_start, pos - num_start)));
            }
            skip_whitespace(json, pos);
          }
          if (json[pos] == ']')
            ++pos;
        } else if (field == "data_offsets") {
          // Parse [start, end] array
          if (json[pos] != '[') {
            throw SafeTensorReaderError("Expected array for data_offsets");
          }
          ++pos;

          std::vector<size_t> offsets;
          while (pos < json.size() && json[pos] != ']') {
            skip_whitespace(json, pos);
            if (json[pos] == ',' || json[pos] == ']') {
              if (json[pos] == ',')
                ++pos;
              continue;
            }

            size_t num_start = pos;
            while (pos < json.size() &&
                   (json[pos] >= '0' && json[pos] <= '9')) {
              ++pos;
            }
            if (pos > num_start) {
              offsets.push_back(
                  std::stoull(json.substr(num_start, pos - num_start)));
            }
            skip_whitespace(json, pos);
          }
          if (json[pos] == ']')
            ++pos;

          if (offsets.size() >= 2) {
            tensor.data_offset = offsets[0];
            tensor.data_size = offsets[1] - offsets[0];
          }
        }

        skip_whitespace(json, pos);
      }
      if (json[pos] == '}')
        ++pos;

      tensors_[tensor.name] = tensor;
    }

    skip_whitespace(json, pos);
  }
}

void SafeTensorReader::read_tensor_data(const std::string &name,
                                        void *buffer,
                                        size_t buffer_size) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw SafeTensorReaderError("Tensor not found: " + name);
  }

  const auto &tensor = it->second;
  if (buffer_size < tensor.data_size) {
    throw SafeTensorReaderError("Buffer too small for tensor " + name);
  }

  file_.seekg(data_offset_ + tensor.data_offset);
  file_.read(reinterpret_cast<char *>(buffer), tensor.data_size);

  if (!file_) {
    throw SafeTensorReaderError("Failed to read tensor data: " + name);
  }
}

std::vector<uint8_t>
SafeTensorReader::read_tensor_raw(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw SafeTensorReaderError("Tensor not found: " + name);
  }

  const auto &tensor = it->second;
  std::vector<uint8_t> data(tensor.data_size);
  read_tensor_data(name, data.data(), data.size());
  return data;
}

void SafeTensorReader::convert_f16_to_f32(const uint16_t *data,
                                          float *output,
                                          size_t n_elements) {
  for (size_t i = 0; i < n_elements; ++i) {
    uint16_t h = data[i];
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;
      } else {
        // Denormalized
        exp = 1;
        while ((mant & 0x400) == 0) {
          mant <<= 1;
          exp--;
        }
        mant &= 0x3FF;
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
      }
    } else if (exp == 31) {
      // Inf or NaN
      f = sign | 0x7F800000 | (mant << 13);
    } else {
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    std::memcpy(&output[i], &f, sizeof(float));
  }
}

void SafeTensorReader::convert_bf16_to_f32(const uint16_t *data,
                                           float *output,
                                           size_t n_elements) {
  for (size_t i = 0; i < n_elements; ++i) {
    uint32_t f = static_cast<uint32_t>(data[i]) << 16;
    std::memcpy(&output[i], &f, sizeof(float));
  }
}

std::vector<float>
SafeTensorReader::read_tensor_f32(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw SafeTensorReaderError("Tensor not found: " + name);
  }

  const auto &tensor = it->second;
  size_t n_elements = tensor.n_elements();
  std::vector<float> output(n_elements);

  std::vector<uint8_t> raw = read_tensor_raw(name);

  switch (tensor.dtype) {
  case DType::F32:
    std::memcpy(output.data(), raw.data(), raw.size());
    break;

  case DType::F16:
    convert_f16_to_f32(reinterpret_cast<const uint16_t *>(raw.data()),
                       output.data(), n_elements);
    break;

  case DType::BF16:
    convert_bf16_to_f32(reinterpret_cast<const uint16_t *>(raw.data()),
                        output.data(), n_elements);
    break;

  case DType::F64:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] =
          static_cast<float>(reinterpret_cast<const double *>(raw.data())[i]);
    }
    break;

  case DType::I8:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] =
          static_cast<float>(reinterpret_cast<const int8_t *>(raw.data())[i]);
    }
    break;

  case DType::U8:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] = static_cast<float>(raw[i]);
    }
    break;

  case DType::I16:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] =
          static_cast<float>(reinterpret_cast<const int16_t *>(raw.data())[i]);
    }
    break;

  case DType::I32:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] =
          static_cast<float>(reinterpret_cast<const int32_t *>(raw.data())[i]);
    }
    break;

  case DType::I64:
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] =
          static_cast<float>(reinterpret_cast<const int64_t *>(raw.data())[i]);
    }
    break;

  default:
    throw SafeTensorReaderError("Unsupported dtype for conversion: " +
                                std::string(dtype_name(tensor.dtype)));
  }

  return output;
}

std::vector<std::string> SafeTensorReader::get_tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

} // namespace safetensor
