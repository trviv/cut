/**
 * GGUF Model Inference Example
 *
 * Loads a LLaMA-architecture model from a GGUF file and runs
 * autoregressive generation using CUT GPU operators.
 *
 * Build via CMake (from project root):
 *   mkdir build && cd build && cmake .. && make -j8
 *
 * Run:
 *   ./gguf_example [path/to/model.gguf]
 */

#include "Runtime.h"
#include "llama.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  std::string filePath;

  if (argc >= 2) {
    filePath = argv[1];
  } else {
#ifdef MODELS_DIR
    filePath = std::string(MODELS_DIR) + "/llm.gguf";
#else
    filePath = "models/llm.gguf";
#endif
  }

  try {
    // Initialize CUT runtime
    cut::Runtime runtime;
    runtime.init(cut::BackendType::Vulkan);

    // Load model
    gguf::LlamaModel model;
    model.load(filePath, runtime);

    const auto &cfg = model.config();
    std::cout << "\nModel ready:\n";
    std::cout << "  dim=" << cfg.dim << " layers=" << cfg.n_layers
              << " heads=" << cfg.n_heads << " vocab=" << cfg.vocab_size
              << "\n\n";

    // Prompt tokens (BOS token = 1 for LLaMA-style models)
    // Without a tokenizer, we use raw token IDs.
    // Token 1 is typically <s> (BOS) for LLaMA models.
    std::vector<int> prompt = {1};

    int max_new_tokens = 32;
    if (argc >= 3) {
      max_new_tokens = std::atoi(argv[2]);
    }

    std::cout << "Generating " << max_new_tokens << " tokens...\n";
    auto tokens = model.generate(prompt, max_new_tokens);

    std::cout << "\nGenerated token IDs: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << tokens[i];
    }
    std::cout << "]\n";

    std::string text = model.detokenize(tokens);
    if (!text.empty()) {
      std::cout << "\nDecoded text:\n" << text << "\n";
    }

    runtime.shutdown();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
