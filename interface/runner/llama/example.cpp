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
 *                  [--ctx-size N] [--no-chat]
 */

#include "Runtime.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  // Separate positional args from --flags
  bool noChat = false;
  uint32_t ctxSize = 0; // 0 = use default (512, matching llama.cpp)
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-chat") == 0) {
      noChat = true;
    } else if (std::strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
      ctxSize = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else {
      positional.push_back(argv[i]);
    }
  }

  std::string filePath;
  if (!positional.empty()) {
    filePath = positional[0];
  } else {
#ifdef MODELS_DIR
    filePath = std::string(MODELS_DIR) + "/llm.gguf";
#else
    filePath = "models/llm.gguf";
#endif
  }

  try {
    // Initialize CUT runtime. CUT_DEVICES="vulkan:1,vulkan:2" (backend[:index],
    // comma-separated) selects one or more devices; default is one Vulkan
    // device.
    cut::Runtime runtime;
    if (const char *devEnv = std::getenv("CUT_DEVICES")) {
      std::vector<cut::DeviceDesc> descs;
      std::string s(devEnv);
      size_t start = 0;
      while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos)
          comma = s.size();
        std::string entry = s.substr(start, comma - start);
        start = comma + 1;
        cut::DeviceDesc desc;
        size_t colon = entry.find(':');
        std::string backend =
            entry.substr(0, colon == std::string::npos ? entry.size() : colon);
        if (colon != std::string::npos) {
          desc.deviceIndex = std::atoi(entry.substr(colon + 1).c_str());
        }
        desc.backend = (backend == "cuda") ? cut::BackendType::CUDA
                                           : cut::BackendType::Vulkan;
        descs.push_back(desc);
      }
      runtime.init(descs);
      std::cout << "Initialized " << runtime.deviceCount() << " devices\n";
    } else {
      runtime.init(cut::BackendType::Vulkan);
    }

    // Automatic layer placement (multi-GPU split + host-RAM overflow).
    // CUT_DEVICE_BUDGET_MB caps per-device memory so small models can
    // exercise the split machinery; explicit CUT_DEVICE_SPLIT/CUT_HOST_LAYERS
    // still win.
    gguf::autoPlaceModel(runtime, filePath);

    // Load model
    gguf::LlamaModel model;
    model.load(filePath, runtime, ctxSize);

    const auto &cfg = model.config();
    std::cout << "\nModel ready:\n";
    std::cout << "  dim=" << cfg.dim << " layers=" << cfg.n_layers
              << " heads=" << cfg.n_heads << " vocab=" << cfg.vocab_size
              << " ctx=" << cfg.max_seq_len << "\n\n";

    int max_new_tokens = 32;
    if (positional.size() >= 2) {
      max_new_tokens = std::atoi(positional[1].c_str());
    }

    // Tokenize prompt from CLI or use default
    std::string prompt_text = "Hello, how are you?";
    if (positional.size() >= 3) {
      prompt_text = positional[2];
    }

    // For instruct/chat models, wrap in ChatML template if special tokens
    // exist. Skip if --no-chat flag is passed as any argument.
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
    if (positional.size() >= 4) {
      repeat_penalty = static_cast<float>(std::atof(positional[3].c_str()));
    }

    std::cout << "Generating " << max_new_tokens
              << " tokens (repeat_penalty=" << repeat_penalty << ")...\n";
    // model.setProfilingEnabled(true);
    auto result = model.generate(prompt, max_new_tokens, repeat_penalty);

    std::cout << "\nPrefill: " << result.prefillMs << " ms, "
              << result.promptTokens << " tokens, "
              << (1000.0 * result.promptTokens / result.prefillMs)
              << " tok/s\n";
    if (result.generatedTokens > 1) {
      // First token comes from prefill; generation phase produces the rest
      int decodeTokens = result.generatedTokens - 1;
      std::cout << "Decode:  " << result.generateMs << " ms, " << decodeTokens
                << " tokens, " << (1000.0 * decodeTokens / result.generateMs)
                << " tok/s\n";
    }
    double totalMs = result.prefillMs + result.generateMs;
    std::cout << "Total:   " << totalMs << " ms, " << result.generatedTokens
              << " tokens\n";

    std::cout << "\nBuffers after generation: " << runtime.bufferCount()
              << "  GPU memory: "
              << (runtime.activeBufferMemoryBytes() / (1024.0 * 1024.0))
              << " MB\n";

    std::cout << "\nGenerated token IDs: [";
    for (size_t i = 0; i < result.tokens.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << result.tokens[i];
    }
    std::cout << "]\n";

    std::string text = model.detokenize(result.tokens);
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
