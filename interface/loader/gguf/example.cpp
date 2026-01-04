/**
 * Example usage of the GGUF reader.
 *
 * Compile with:
 *   clang++ -std=c++17 -o gguf_example example.cpp gguf_reader.cpp
 *
 * Run with:
 *   ./gguf_example path/to/model.gguf
 */

#include "gguf_reader.hpp"
#include <iomanip>
#include <iostream>
#include <type_traits>

int main(int argc, char *argv[]) {
  // if (argc < 2) {
  //   std::cerr << "Usage: " << argv[0] << " <path_to_gguf_file>\n";
  //   std::cerr << "\nExample:\n";
  //   std::cerr << "  " << argv[0] << " model.gguf\n";
  //   return 1;
  // }

#ifdef MODELS_DIR
  std::string filePath = std::string(MODELS_DIR) + "/llm.gguf";
#else
  std::string filePath = "models/llm.gguf";
#endif

  try {
    std::cout << "Loading GGUF file: " << filePath.c_str() << "\n";
    std::cout << std::string(60, '-') << "\n";

    auto reader = gguf::load_gguf(filePath.c_str());

    // Print basic info
    std::cout << "Version: " << reader->version() << "\n";
    std::cout << "Architecture: " << reader->metadata().architecture() << "\n";
    std::cout << "Model Name: " << reader->metadata().name() << "\n";
    std::cout << "Alignment: " << reader->metadata().alignment() << "\n";
    std::cout << "Tensor Count: " << reader->tensors().size() << "\n";
    std::cout << "Tensor Data Offset: " << reader->tensor_data_offset() << "\n";
    std::cout << std::string(60, '-') << "\n";

    // Print metadata
    std::cout << "\nMetadata:\n";
    for (const auto &[key, value] : reader->metadata().items()) {
      std::cout << "  " << key << ": ";

      std::visit(
          [](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
              if (v.length() > 50) {
                std::cout << v.substr(0, 50) << "...";
              } else {
                std::cout << v;
              }
            } else if constexpr (std::is_same_v<T, bool>) {
              std::cout << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, int8_t>) {
              std::cout << static_cast<int>(v);
            } else if constexpr (std::is_same_v<T, uint8_t>) {
              std::cout << static_cast<unsigned int>(v);
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
              std::cout << "[" << v.size() << " strings]";
            } else if constexpr (std::is_arithmetic_v<T>) {
              std::cout << v;
            } else {
              // All vector types
              std::cout << "[" << v.size() << " items]";
            }
          },
          value);
      std::cout << "\n";
    }

    // Print tensor info
    std::cout << "\nTensors (first 10):\n";
    int count = 0;
    for (const auto &[name, info] : reader->tensors()) {
      if (count++ >= 10) {
        std::cout << "  ... and " << (reader->tensors().size() - 10)
                  << " more tensors\n";
        break;
      }

      std::cout << "  " << name << "\n";
      std::cout << "    shape: [";
      for (size_t i = 0; i < info.dimensions.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << info.dimensions[i];
      }
      std::cout << "], dtype: " << gguf::get_type_name(info.type);
      std::cout << ", size: " << info.nbytes() << " bytes\n";
    }

    // Example: read first tensor
    if (!reader->tensors().empty()) {
      const auto &[name, info] = *reader->tensors().begin();
      std::cout << "\nReading tensor: " << name << "\n";

      try {
        auto data = reader->read_tensor_raw(name);
        std::cout << "  Loaded " << data.size() << " bytes\n";
        std::cout << "  First 8 bytes: ";
        for (size_t i = 0; i < std::min(size_t(8), data.size()); ++i) {
          std::cout << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(data[i]) << " ";
        }
        std::cout << std::dec << "\n";
      } catch (const std::exception &e) {
        std::cout << "  Error reading tensor: " << e.what() << "\n";
      }
    }

  } catch (const gguf::GGUFReaderError &e) {
    std::cerr << "GGUF Error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
