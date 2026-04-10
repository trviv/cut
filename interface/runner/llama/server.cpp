/**
 * GGUF HTTP Server
 *
 * Loads a GGUF model and serves inference over HTTP.
 * Keeps the model in GPU memory for fast repeated inference.
 *
 * Usage:
 *   gguf_server <model.gguf> [--port 8080] [--host 0.0.0.0]
 *               [--max-tokens 128] [--no-chat]
 */

#include "Runtime.h"
#include "llama.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static httplib::Server *g_server = nullptr;

static void signalHandler(int) {
  if (g_server) {
    g_server->stop();
  }
}

struct ServerConfig {
  std::string modelPath;
  std::string host = "0.0.0.0";
  int port = 8080;
  int defaultMaxTokens = 128;
  float defaultRepeatPenalty = 1.05f;
  bool noChat = false;
};

static ServerConfig parseArgs(int argc, char *argv[]) {
  ServerConfig cfg;

  if (argc < 2) {
    std::cerr << "Usage: gguf_server <model.gguf> [options]\n"
              << "Options:\n"
              << "  --port N           HTTP port (default: 8080)\n"
              << "  --host ADDR        Bind address (default: 0.0.0.0)\n"
              << "  --max-tokens N     Default max tokens (default: 128)\n"
              << "  --repeat-penalty F Default repeat penalty (default: 1.05)\n"
              << "  --no-chat          Disable ChatML wrapping\n";
    std::exit(1);
  }

  cfg.modelPath = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--port" && i + 1 < argc) {
      cfg.port = std::atoi(argv[++i]);
    } else if (arg == "--host" && i + 1 < argc) {
      cfg.host = argv[++i];
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      cfg.defaultMaxTokens = std::atoi(argv[++i]);
    } else if (arg == "--repeat-penalty" && i + 1 < argc) {
      cfg.defaultRepeatPenalty = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--no-chat") {
      cfg.noChat = true;
    }
  }

  return cfg;
}

static json errorJson(const std::string &msg) {
  return json{{"error", msg}};
}

static void log(const std::string &msg) {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  char buf[20];
  std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
  std::cout << "[" << buf << "." << std::setfill('0') << std::setw(3)
            << ms.count() << "] " << msg << "\n";
}

int main(int argc, char *argv[]) {
  auto cfg = parseArgs(argc, argv);

  // Initialize runtime and load model
  std::cout << "Initializing runtime...\n";
  cut::Runtime runtime;
  runtime.init(cut::BackendType::Vulkan);

  std::cout << "Loading model: " << cfg.modelPath << "\n";
  gguf::LlamaModel model;
  model.load(cfg.modelPath, runtime);

  const auto &modelCfg = model.config();
  std::cout << "Model loaded: dim=" << modelCfg.dim
            << " layers=" << modelCfg.n_layers << " heads=" << modelCfg.n_heads
            << " vocab=" << modelCfg.vocab_size << "\n";

  // Auto-detect ChatML support
  int imStartId = model.tokenId("<|im_start|>");
  int imEndId = model.tokenId("<|im_end|>");
  bool chatAvailable = !cfg.noChat && imStartId >= 0 && imEndId >= 0;
  if (chatAvailable) {
    model.addStopToken(imEndId);
    std::cout << "ChatML detected (im_start=" << imStartId
              << " im_end=" << imEndId << ")\n";
  }

  std::mutex modelMutex;

  // Create server
  httplib::Server server;
  g_server = &server;
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // --- GET /v1/health ---
  server.Get(
      "/v1/health", [&](const httplib::Request &, httplib::Response &res) {
        log("GET /v1/health");
        json resp = {{"status", "ok"},
                     {"model_path", cfg.modelPath},
                     {"chat_mode", chatAvailable},
                     {"default_max_tokens", cfg.defaultMaxTokens},
                     {"gpu_memory_mb",
                      runtime.activeBufferMemoryBytes() / (1024.0 * 1024.0)},
                     {"buffer_count", runtime.bufferCount()},
                     {"model_config",
                      {{"dim", modelCfg.dim},
                       {"n_layers", modelCfg.n_layers},
                       {"n_heads", modelCfg.n_heads},
                       {"n_kv_heads", modelCfg.n_kv_heads},
                       {"vocab_size", modelCfg.vocab_size},
                       {"ffn_dim", modelCfg.ffn_dim},
                       {"max_seq_len", modelCfg.max_seq_len}}}};
        res.set_content(resp.dump(), "application/json");
      });

  // --- POST /v1/reset ---
  server.Post("/v1/reset",
              [&](const httplib::Request &, httplib::Response &res) {
                log("POST /v1/reset");
                std::lock_guard<std::mutex> lock(modelMutex);
                model.resetCache();
                log("  KV cache reset");
                json resp = {{"status", "ok"}, {"message", "KV cache reset"}};
                res.set_content(resp.dump(), "application/json");
              });

  // --- POST /v1/generate ---
  server.Post("/v1/generate", [&](const httplib::Request &req,
                                  httplib::Response &res) {
    try {
      auto body = json::parse(req.body);

      if (!body.contains("prompt") || !body["prompt"].is_string()) {
        res.status = 400;
        res.set_content(errorJson("Missing 'prompt' string").dump(),
                        "application/json");
        return;
      }

      std::string prompt = body["prompt"].get<std::string>();
      int maxTokens = body.value("max_tokens", cfg.defaultMaxTokens);
      float repeatPenalty =
          body.value("repeat_penalty", cfg.defaultRepeatPenalty);
      int repeatLastN = body.value("repeat_last_n", 64);
      bool useChatMode = body.value("chat_mode", chatAvailable);
      bool resetCache = body.value("reset_cache", true);

      std::string promptPreview =
          prompt.size() > 80 ? prompt.substr(0, 80) + "..." : prompt;
      log("POST /v1/generate | prompt=\"" + promptPreview +
          "\" max_tokens=" + std::to_string(maxTokens));

      std::lock_guard<std::mutex> lock(modelMutex);

      if (resetCache) {
        model.resetCache();
      }

      // Build tokenizer input
      std::string tokenizerInput = prompt;
      if (chatAvailable && useChatMode) {
        tokenizerInput = "<|im_start|>user\n" + prompt +
                         "<|im_end|>\n<|im_start|>assistant\n";
      }

      auto promptTokens = model.tokenize(tokenizerInput);
      size_t promptSize = promptTokens.size();
      log("  Tokenized: " + std::to_string(promptSize) + " prompt tokens");

      log("  Generating...");
      auto result =
          model.generate(promptTokens, maxTokens, repeatPenalty, repeatLastN);

      std::vector<int> genTokens(result.tokens.begin() +
                                     static_cast<int>(promptSize),
                                 result.tokens.end());
      std::string text = model.detokenize(genTokens);

      double totalMs = result.prefillMs + result.generateMs;
      int decodeTokens =
          result.generatedTokens > 1 ? result.generatedTokens - 1 : 0;
      double decodeTokPerSec =
          decodeTokens > 0 ? 1000.0 * decodeTokens / result.generateMs : 0.0;

      log("  Done: " + std::to_string(genTokens.size()) + " tokens in " +
          std::to_string(static_cast<int>(totalMs)) + "ms (decode " +
          std::to_string(static_cast<int>(decodeTokPerSec)) + " tok/s)");

      json resp = {{"text", text},
                   {"prompt_tokens", promptSize},
                   {"generated_tokens", genTokens.size()},
                   {"total_tokens", result.tokens.size()},
                   {"prefill_ms", result.prefillMs},
                   {"generate_ms", result.generateMs},
                   {"elapsed_ms", totalMs},
                   {"tokens_per_second", decodeTokPerSec}};
      res.set_content(resp.dump(), "application/json");

    } catch (const json::exception &e) {
      log("  Error: Invalid JSON: " + std::string(e.what()));
      res.status = 400;
      res.set_content(
          errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
          "application/json");
    } catch (const std::exception &e) {
      log("  Error: " + std::string(e.what()));
      res.status = 500;
      res.set_content(errorJson(e.what()).dump(), "application/json");
    }
  });

  // --- POST /v1/chat ---
  server.Post("/v1/chat", [&](const httplib::Request &req,
                              httplib::Response &res) {
    try {
      auto body = json::parse(req.body);

      if (!body.contains("message") || !body["message"].is_string()) {
        res.status = 400;
        res.set_content(errorJson("Missing 'message' string").dump(),
                        "application/json");
        return;
      }

      std::string message = body["message"].get<std::string>();
      int maxTokens = body.value("max_tokens", cfg.defaultMaxTokens);
      float repeatPenalty =
          body.value("repeat_penalty", cfg.defaultRepeatPenalty);
      int repeatLastN = body.value("repeat_last_n", 64);
      std::string systemPrompt = body.value(
          "system_prompt", std::string("You are a helpful coding assistant."));

      std::string msgPreview =
          message.size() > 80 ? message.substr(0, 80) + "..." : message;
      log("POST /v1/chat | message=\"" + msgPreview + "\"");

      // Read files and build context
      std::string fileContext;
      if (body.contains("files") && body["files"].is_array()) {
        for (const auto &filePath : body["files"]) {
          std::string path = filePath.get<std::string>();
          log("  Reading file: " + path);
          std::ifstream ifs(path);
          if (!ifs.is_open()) {
            log("  Error: cannot read file: " + path);
            res.status = 400;
            res.set_content(errorJson("Cannot read file: " + path).dump(),
                            "application/json");
            return;
          }
          std::ostringstream ss;
          ss << ifs.rdbuf();
          fileContext += "[File: " + path + "]\n```\n" + ss.str() + "\n```\n\n";
        }
      }

      // Build prompt
      std::string tokenizerInput;
      if (chatAvailable) {
        tokenizerInput = "<|im_start|>system\n" + systemPrompt +
                         "<|im_end|>\n<|im_start|>user\n";
        if (!fileContext.empty()) {
          tokenizerInput += fileContext;
        }
        tokenizerInput += message + "<|im_end|>\n<|im_start|>assistant\n";
      } else {
        tokenizerInput = systemPrompt + "\n\n";
        if (!fileContext.empty()) {
          tokenizerInput += fileContext;
        }
        tokenizerInput += message + "\n\n";
      }

      std::lock_guard<std::mutex> lock(modelMutex);
      model.resetCache();

      auto promptTokens = model.tokenize(tokenizerInput);
      size_t promptSize = promptTokens.size();
      log("  Tokenized: " + std::to_string(promptSize) + " prompt tokens");

      log("  Generating...");
      auto result =
          model.generate(promptTokens, maxTokens, repeatPenalty, repeatLastN);

      std::vector<int> genTokens(result.tokens.begin() +
                                     static_cast<int>(promptSize),
                                 result.tokens.end());
      std::string text = model.detokenize(genTokens);

      double totalMs = result.prefillMs + result.generateMs;
      int decodeTokens =
          result.generatedTokens > 1 ? result.generatedTokens - 1 : 0;
      double decodeTokPerSec =
          decodeTokens > 0 ? 1000.0 * decodeTokens / result.generateMs : 0.0;

      log("  Done: " + std::to_string(genTokens.size()) + " tokens in " +
          std::to_string(static_cast<int>(totalMs)) + "ms (decode " +
          std::to_string(static_cast<int>(decodeTokPerSec)) + " tok/s)");

      json resp = {{"text", text},
                   {"prompt_tokens", promptSize},
                   {"generated_tokens", genTokens.size()},
                   {"total_tokens", result.tokens.size()},
                   {"prefill_ms", result.prefillMs},
                   {"generate_ms", result.generateMs},
                   {"elapsed_ms", totalMs},
                   {"tokens_per_second", decodeTokPerSec}};
      res.set_content(resp.dump(), "application/json");

    } catch (const json::exception &e) {
      log("  Error: Invalid JSON: " + std::string(e.what()));
      res.status = 400;
      res.set_content(
          errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
          "application/json");
    } catch (const std::exception &e) {
      log("  Error: " + std::string(e.what()));
      res.status = 500;
      res.set_content(errorJson(e.what()).dump(), "application/json");
    }
  });

  // --- POST /v1/read ---
  server.Post("/v1/read", [&](const httplib::Request &req,
                              httplib::Response &res) {
    try {
      auto body = json::parse(req.body);

      if (!body.contains("path") || !body["path"].is_string()) {
        res.status = 400;
        res.set_content(errorJson("Missing 'path' string").dump(),
                        "application/json");
        return;
      }

      std::string path = body["path"].get<std::string>();
      log("POST /v1/read | path=" + path);
      std::ifstream ifs(path);
      if (!ifs.is_open()) {
        log("  Error: file not found");
        res.status = 404;
        res.set_content(errorJson("File not found: " + path).dump(),
                        "application/json");
        return;
      }

      std::ostringstream ss;
      ss << ifs.rdbuf();
      std::string content = ss.str();
      log("  Read " + std::to_string(content.size()) + " bytes");

      json resp = {
          {"path", path}, {"content", content}, {"size_bytes", content.size()}};
      res.set_content(resp.dump(), "application/json");

    } catch (const json::exception &e) {
      res.status = 400;
      res.set_content(
          errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
          "application/json");
    }
  });

  // --- POST /v1/write ---
  server.Post(
      "/v1/write", [&](const httplib::Request &req, httplib::Response &res) {
        try {
          auto body = json::parse(req.body);

          if (!body.contains("path") || !body["path"].is_string() ||
              !body.contains("content") || !body["content"].is_string()) {
            res.status = 400;
            res.set_content(
                errorJson("Missing 'path' and/or 'content' strings").dump(),
                "application/json");
            return;
          }

          std::string path = body["path"].get<std::string>();
          std::string content = body["content"].get<std::string>();
          log("POST /v1/write | path=" + path + " (" +
              std::to_string(content.size()) + " bytes)");

          std::ofstream ofs(path);
          if (!ofs.is_open()) {
            log("  Error: cannot open for writing");
            res.status = 500;
            res.set_content(
                errorJson("Cannot open file for writing: " + path).dump(),
                "application/json");
            return;
          }

          ofs << content;
          ofs.close();

          if (ofs.fail()) {
            log("  Error: write failed");
            res.status = 500;
            res.set_content(errorJson("Write failed: " + path).dump(),
                            "application/json");
            return;
          }

          log("  Written successfully");
          json resp = {{"path", path}, {"bytes_written", content.size()}};
          res.set_content(resp.dump(), "application/json");

        } catch (const json::exception &e) {
          res.status = 400;
          res.set_content(
              errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
              "application/json");
        }
      });

  // Start server
  double gpuMb = runtime.activeBufferMemoryBytes() / (1024.0 * 1024.0);
  std::cout << "\n=== GGUF Server ===\n"
            << "Model: " << cfg.modelPath << "\n"
            << "GPU memory: " << gpuMb << " MB (" << runtime.bufferCount()
            << " buffers)\n"
            << "Chat mode: " << (chatAvailable ? "enabled" : "disabled") << "\n"
            << "Default max tokens: " << cfg.defaultMaxTokens << "\n"
            << "Listening on http://" << cfg.host << ":" << cfg.port << "\n\n";

  if (!server.listen(cfg.host, cfg.port)) {
    std::cerr << "Failed to start server on " << cfg.host << ":" << cfg.port
              << "\n";
    return 1;
  }

  std::cout << "\nServer stopped.\n";
  return 0;
}
