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
 *   ./gguf_example [model.gguf] [max_tokens] [prompt] [repeat_penalty]
 */

#include "Runtime.h"
#include "llama.h"

#include <exception>
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

    int max_new_tokens = 32;
    if (argc >= 3) {
      max_new_tokens = std::atoi(argv[2]);
    }

    // Tokenize prompt from CLI or use default
    std::string prompt_text = "Hello, how are you?";
    if (argc >= 4) {
      prompt_text = argv[3];
    }

    std::cout << "Prompt: \"" << prompt_text << "\"\n";
    std::vector<int> prompt;
    try {
      prompt = model.tokenize(prompt_text);
    } catch (std::exception &) {
      prompt = {1};
    }

    std::cout << "Prompt token IDs: [";
    for (size_t i = 0; i < prompt.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << prompt[i];
    }
    std::cout << "]\n";

    float repeat_penalty = 1.05f;
    if (argc >= 5) {
      repeat_penalty = static_cast<float>(std::atof(argv[4]));
    }

    std::cout << "Generating " << max_new_tokens
              << " tokens (repeat_penalty=" << repeat_penalty << ")...\n";
    auto tokens = model.generate(prompt, max_new_tokens, repeat_penalty);

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

    // Note: do NOT call runtime.shutdown() here — model still holds tensor
    // handles. Natural destruction order (model before runtime) handles
    // cleanup correctly.

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
