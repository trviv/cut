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

#include <chrono>
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

    // For instruct/chat models, wrap in ChatML template if special tokens
    // exist. Skip if --no-chat flag is passed as any argument.
    bool noChat = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--no-chat") {
        noChat = true;
      }
    }

    int im_start = model.tokenId("<|im_start|>");
    int im_end = model.tokenId("<|im_end|>");
    std::string tokenizer_input = prompt_text;
    if (!noChat && im_start >= 0 && im_end >= 0) {
      tokenizer_input = "<|im_start|>user\n" + prompt_text +
                        "<|im_end|>\n<|im_start|>assistant\n";
      model.addStopToken(im_end);
      std::cout << "Using ChatML template (im_start=" << im_start
                << " im_end=" << im_end << ").\n";
    }

    std::cout << "Prompt: \"" << prompt_text << "\"\n";
    std::vector<int> prompt = model.tokenize(tokenizer_input);

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
    // model.setProfilingEnabled(true);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tokens = model.generate(prompt, max_new_tokens, repeat_penalty);
    auto t1 = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int newTokens =
        static_cast<int>(tokens.size()) - static_cast<int>(prompt.size());
    if (newTokens > 0) {
      std::cout << "\nGeneration: " << totalMs << " ms total, "
                << (totalMs / newTokens) << " ms/token, "
                << (1000.0 * newTokens / totalMs) << " tok/s\n";
    }

    std::cout << "\nBuffers after generation: " << runtime.bufferCount()
              << "  GPU memory: "
              << (runtime.activeBufferMemoryBytes() / (1024.0 * 1024.0))
              << " MB\n";

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
