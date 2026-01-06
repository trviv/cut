#include "safetensor_reader.hpp"
#include <iomanip>
#include <iostream>

#ifndef MODELS_DIR
#define MODELS_DIR "../../models"
#endif

int main(int argc, char *argv[]) {
  std::string model_path = MODELS_DIR "/model.safetensors";

  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Loading SafeTensor file: " << model_path << std::endl;

  try {
    auto reader = safetensor::load_safetensor(model_path);

    std::cout << "\n=== Header Info ===" << std::endl;
    std::cout << "Header size: " << reader->header_size() << " bytes"
              << std::endl;
    std::cout << "Data offset: " << reader->data_offset() << " bytes"
              << std::endl;

    // Print metadata
    const auto &metadata = reader->metadata();
    if (!metadata.items().empty()) {
      std::cout << "\n=== Metadata ===" << std::endl;
      for (const auto &[key, value] : metadata.items()) {
        std::cout << "  " << key << ": " << value << std::endl;
      }
    }

    // Print tensor info
    std::cout << "\n=== Tensors ===" << std::endl;
    auto tensor_names = reader->get_tensor_names();
    std::cout << "Total tensors: " << tensor_names.size() << std::endl;

    for (const auto &name : tensor_names) {
      const auto &info = reader->get_tensor_info(name);
      std::cout << "\n  " << name << ":" << std::endl;
      std::cout << "    dtype: " << safetensor::dtype_name(info.dtype)
                << std::endl;
      std::cout << "    shape: [";
      for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << info.shape[i];
      }
      std::cout << "]" << std::endl;
      std::cout << "    size: " << info.data_size << " bytes" << std::endl;
    }

    // Try reading first tensor
    if (!tensor_names.empty()) {
      const auto &first_tensor = tensor_names[0];
      std::cout << "\n=== Reading first tensor: " << first_tensor
                << " ===" << std::endl;

      auto data = reader->read_tensor_f32(first_tensor);
      std::cout << "Elements: " << data.size() << std::endl;
      std::cout << "First 10 values: ";
      for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "\nSuccess!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
