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

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
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
  std::string devices;   // "vulkan:1,vulkan:2" etc.; "" = one default device
  uint32_t ctxSize = 0;  // 0 = model default
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
              << "  --no-chat          Disable ChatML wrapping\n"
              << "  --devices STR      Device list, e.g. \"vulkan:1,vulkan:2\""
                 " or \"cuda:0,vulkan:2\"\n"
              << "  --ctx-size N       Context size (default: model default)\n";
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
    } else if (arg == "--devices" && i + 1 < argc) {
      cfg.devices = argv[++i];
    } else if (arg == "--ctx-size" && i + 1 < argc) {
      cfg.ctxSize = static_cast<uint32_t>(std::atoi(argv[++i]));
    }
  }

  // Default device list from the environment when the flag is absent.
  if (cfg.devices.empty()) {
    if (const char *envDevices = std::getenv("CUT_DEVICES")) {
      cfg.devices = envDevices;
    }
  }

  return cfg;
}

/// Parses "backend[:index],..." into device descriptors ("" = one default
/// Vulkan device). backend is "vulkan" or "cuda"; index -1 = backend default.
static std::vector<cut::DeviceDesc> parseDeviceList(const std::string &s) {
  std::vector<cut::DeviceDesc> descs;
  if (s.empty()) {
    descs.push_back({cut::BackendType::Vulkan, -1});
    return descs;
  }
  size_t start = 0;
  while (start < s.size()) {
    size_t comma = s.find(',', start);
    if (comma == std::string::npos) {
      comma = s.size();
    }
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
  return descs;
}

/// Model id for the OpenAI API: file name without the .gguf extension.
static std::string modelIdFromPath(const std::string &p) {
  return std::filesystem::path(p).stem().string();
}

/// Truncates a string so it does not end in the middle of a UTF-8 sequence.
static std::string trimIncompleteUtf8(const std::string &s) {
  size_t end = s.size();
  // Find the last lead byte within the final 4 bytes.
  size_t i = end;
  while (i > 0 && end - i < 4) {
    --i;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if ((c & 0x80) == 0) {
      return s.substr(0, end); // ASCII — everything complete
    }
    if ((c & 0xC0) == 0xC0) {
      // Lead byte: expected sequence length from the high bits.
      size_t len = 2;
      if ((c & 0xF0) == 0xE0) {
        len = 3;
      } else if ((c & 0xF8) == 0xF0) {
        len = 4;
      }
      return s.substr(0, (i + len <= end) ? end : i);
    }
    // Continuation byte — keep scanning backwards.
  }
  return s.substr(0, end);
}

/// Parameters for one OpenAI-style generation request. Sampling is greedy
/// argmax; temperature/top_p are accepted but ignored.
struct GenParams {
  std::vector<int> promptTokens;
  int maxTokens = 128;
  std::vector<std::string> stopStrings;
  bool stream = false;
};

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

  // Initialize runtime (one or more devices) and place the model.
  std::cout << "Initializing runtime...\n";
  cut::Runtime runtime;
  runtime.init(parseDeviceList(cfg.devices));
  std::cout << "Devices: " << runtime.deviceCount() << "\n";

  gguf::autoPlaceModel(runtime, cfg.modelPath);

  std::cout << "Loading model: " << cfg.modelPath << "\n";
  gguf::LlamaModel model;
  model.load(cfg.modelPath, runtime, cfg.ctxSize);

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

  // =========================================================================
  // OpenAI-compatible API (works with Cline, OpenCode, Codex CLI, etc.)
  // Generation is greedy argmax; temperature/top_p are accepted but ignored.
  // =========================================================================

  const std::string modelId = modelIdFromPath(cfg.modelPath);
  std::atomic<uint64_t> requestCounter{0};

  // Core greedy generation loop. Caller must hold modelMutex. Calls
  // onDelta(textDelta) for each newly visible UTF-8-complete chunk.
  // Returns {finishReason, completionTokens, fullText}.
  auto runGeneration =
      [&model, imEndId](const GenParams &params,
                        const std::function<void(const std::string &)>
                            &onDelta) -> std::tuple<std::string, size_t,
                                                    std::string> {
    model.resetCache();
    std::vector<int> gen;
    std::string full;
    std::string finishReason = "length";
    size_t emitted = 0;

    // Suppress EOS/stop tokens for the first few tokens (counteracts FP32
    // drift elevating the EOS logit on longer prompts — same heuristic as
    // LlamaModel::generate()).
    const int minNewTokens =
        std::max(1, static_cast<int>(params.promptTokens.size()) / 4);
    bool suppressing = true;
    model.setStopTokensSuppressed(true);

    // CUT_PREFILL=per_token: correctness fallback for model geometries where
    // the batched prefill ops misbehave (routes through the decode path).
    static const bool perTokenPrefill = [] {
      const char *m = std::getenv("CUT_PREFILL");
      return m && std::string(m) == "per_token";
    }();
    int next;
    if (perTokenPrefill) {
      next = model.prefill(params.promptTokens);
    } else {
      next = model.prefillBatched(params.promptTokens);
    }
    size_t pos = params.promptTokens.size();

    while (true) {
      if (next == model.eosTokenId() || (imEndId >= 0 && next == imEndId)) {
        finishReason = "stop";
        break;
      }
      gen.push_back(next);
      if (suppressing && static_cast<int>(gen.size()) >= minNewTokens) {
        model.setStopTokensSuppressed(false);
        suppressing = false;
      }
      full = model.detokenize(gen);

      bool hitStopString = false;
      for (const auto &s : params.stopStrings) {
        const size_t at = full.find(s);
        if (at != std::string::npos) {
          full.resize(at);
          hitStopString = true;
          break;
        }
      }
      if (hitStopString) {
        finishReason = "stop";
        break;
      }

      const std::string visible = trimIncompleteUtf8(full);
      if (visible.size() > emitted && onDelta) {
        onDelta(visible.substr(emitted));
        emitted = visible.size();
      }

      if (static_cast<int>(gen.size()) >= params.maxTokens) {
        break; // finishReason stays "length"
      }
      next = model.decodeStep(next, static_cast<int>(pos));
      ++pos;
    }
    // Flush any bytes held back for UTF-8 completeness.
    if (onDelta && full.size() > emitted) {
      onDelta(full.substr(emitted));
    }
    if (suppressing) {
      model.setStopTokensSuppressed(false); // leave factors clean
    }
    return {finishReason, gen.size(), full};
  };

  // Parses shared OpenAI params (max_tokens, stop, stream) and clamps to the
  // context window. Returns false (and sets the error response) on failure.
  auto fillGenParams = [&model, &cfg](const json &body,
                                      std::vector<int> promptTokens,
                                      GenParams &params,
                                      httplib::Response &res) -> bool {
    params.promptTokens = std::move(promptTokens);
    params.maxTokens =
        body.value("max_tokens",
                   body.value("max_completion_tokens", cfg.defaultMaxTokens));
    params.stream = body.value("stream", false);
    if (body.contains("stop")) {
      if (body["stop"].is_string()) {
        params.stopStrings.push_back(body["stop"].get<std::string>());
      } else if (body["stop"].is_array()) {
        for (const auto &s : body["stop"]) {
          if (s.is_string()) {
            params.stopStrings.push_back(s.get<std::string>());
          }
        }
      }
    }
    const size_t maxSeqLen = model.config().max_seq_len;
    if (params.promptTokens.size() + 1 >= maxSeqLen) {
      res.status = 400;
      res.set_content(errorJson("Prompt too long for context window").dump(),
                      "application/json");
      return false;
    }
    const int room =
        static_cast<int>(maxSeqLen - params.promptTokens.size() - 1);
    params.maxTokens = std::min(params.maxTokens, room);
    return true;
  };

  // Shared streaming responder: runs generation inside the chunked-content
  // provider (which httplib invokes after this handler returns), holding the
  // model mutex for the whole stream via state.
  struct StreamState {
    std::unique_lock<std::mutex> lock;
    GenParams params;
    std::string id;
    time_t created = 0;
    bool chat = false;
  };
  auto streamResponse = [&runGeneration, &modelId](
                            httplib::Response &res,
                            std::shared_ptr<StreamState> state) {
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [state, &runGeneration, &modelId](size_t, httplib::DataSink &sink)
            -> bool {
          auto writeChunk = [&](const json &deltaOrText,
                                const json &finishReason) {
            json choice = {{"index", 0}, {"finish_reason", finishReason}};
            if (state->chat) {
              choice["delta"] = deltaOrText;
            } else {
              choice["text"] =
                  deltaOrText.is_string() ? deltaOrText : json("");
            }
            json chunk = {
                {"id", state->id},
                {"object",
                 state->chat ? "chat.completion.chunk" : "text_completion"},
                {"created", static_cast<int64_t>(state->created)},
                {"model", modelId},
                {"choices", json::array({choice})}};
            const std::string line = "data: " + chunk.dump() + "\n\n";
            sink.write(line.data(), line.size());
          };
          try {
            if (state->chat) {
              writeChunk(json{{"role", "assistant"}, {"content", ""}},
                         nullptr);
            }
            auto [finishReason, nTokens, text] = runGeneration(
                state->params, [&](const std::string &delta) {
                  writeChunk(state->chat ? json{{"content", delta}}
                                         : json(delta),
                             nullptr);
                });
            (void)nTokens;
            (void)text;
            writeChunk(state->chat ? json(json::object()) : json(""),
                       finishReason);
          } catch (const std::exception &e) {
            log(std::string("  Stream error: ") + e.what());
          }
          static const std::string doneLine = "data: [DONE]\n\n";
          sink.write(doneLine.data(), doneLine.size());
          sink.done();
          return true;
        });
  };

  // --- GET /v1/models ---
  server.Get("/v1/models",
             [&](const httplib::Request &, httplib::Response &res) {
               log("GET /v1/models");
               json resp = {{"object", "list"},
                            {"data", json::array({json{
                                         {"id", modelId},
                                         {"object", "model"},
                                         {"created",
                                          static_cast<int64_t>(
                                              std::time(nullptr))},
                                         {"owned_by", "cut"}}})}};
               res.set_content(resp.dump(), "application/json");
             });

  // --- POST /v1/chat/completions ---
  server.Post("/v1/chat/completions", [&](const httplib::Request &req,
                                          httplib::Response &res) {
    try {
      auto body = json::parse(req.body);
      if (!body.contains("messages") || !body["messages"].is_array() ||
          body["messages"].empty()) {
        res.status = 400;
        res.set_content(errorJson("Missing 'messages' array").dump(),
                        "application/json");
        return;
      }

      // Build the prompt from the message list (ChatML when available).
      std::string prompt;
      for (const auto &msg : body["messages"]) {
        const std::string role = msg.value("role", "user");
        std::string content;
        if (msg.contains("content") && msg["content"].is_string()) {
          content = msg["content"].get<std::string>();
        } else if (msg.contains("content") && msg["content"].is_array()) {
          for (const auto &part : msg["content"]) {
            if (part.value("type", "") == "text" && part.contains("text")) {
              content += part["text"].get<std::string>();
            }
          }
        }
        if (chatAvailable) {
          prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        } else {
          prompt += role + ": " + content + "\n";
        }
      }
      prompt += chatAvailable ? "<|im_start|>assistant\n" : "assistant: ";

      GenParams params;
      if (!fillGenParams(body, model.tokenize(prompt), params, res)) {
        return;
      }

      const time_t created = std::time(nullptr);
      const std::string requestId =
          "chatcmpl-" + std::to_string(created) + "-" +
          std::to_string(requestCounter.fetch_add(1));
      log("POST /v1/chat/completions | prompt_tokens=" +
          std::to_string(params.promptTokens.size()) +
          " max_tokens=" + std::to_string(params.maxTokens) +
          (params.stream ? " stream" : ""));

      if (params.stream) {
        auto state = std::make_shared<StreamState>();
        state->lock = std::unique_lock<std::mutex>(modelMutex);
        state->params = std::move(params);
        state->id = requestId;
        state->created = created;
        state->chat = true;
        streamResponse(res, std::move(state));
        return;
      }

      std::lock_guard<std::mutex> lock(modelMutex);
      auto [finishReason, nTokens, text] = runGeneration(params, nullptr);
      json resp = {
          {"id", requestId},
          {"object", "chat.completion"},
          {"created", static_cast<int64_t>(created)},
          {"model", modelId},
          {"choices",
           json::array({json{{"index", 0},
                             {"message", {{"role", "assistant"},
                                          {"content", text}}},
                             {"finish_reason", finishReason}}})},
          {"usage",
           {{"prompt_tokens", params.promptTokens.size()},
            {"completion_tokens", nTokens},
            {"total_tokens", params.promptTokens.size() + nTokens}}}};
      res.set_content(resp.dump(), "application/json");

    } catch (const json::exception &e) {
      res.status = 400;
      res.set_content(
          errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
          "application/json");
    } catch (const std::exception &e) {
      log(std::string("  Error: ") + e.what());
      res.status = 500;
      res.set_content(errorJson(e.what()).dump(), "application/json");
    }
  });

  // --- POST /v1/completions ---
  server.Post("/v1/completions", [&](const httplib::Request &req,
                                     httplib::Response &res) {
    try {
      auto body = json::parse(req.body);
      std::string prompt;
      if (body.contains("prompt") && body["prompt"].is_string()) {
        prompt = body["prompt"].get<std::string>();
      } else if (body.contains("prompt") && body["prompt"].is_array() &&
                 !body["prompt"].empty() && body["prompt"][0].is_string()) {
        prompt = body["prompt"][0].get<std::string>();
      } else {
        res.status = 400;
        res.set_content(errorJson("Missing 'prompt'").dump(),
                        "application/json");
        return;
      }

      GenParams params;
      if (!fillGenParams(body, model.tokenize(prompt), params, res)) {
        return;
      }

      const time_t created = std::time(nullptr);
      const std::string requestId =
          "cmpl-" + std::to_string(created) + "-" +
          std::to_string(requestCounter.fetch_add(1));
      log("POST /v1/completions | prompt_tokens=" +
          std::to_string(params.promptTokens.size()) +
          " max_tokens=" + std::to_string(params.maxTokens) +
          (params.stream ? " stream" : ""));

      if (params.stream) {
        auto state = std::make_shared<StreamState>();
        state->lock = std::unique_lock<std::mutex>(modelMutex);
        state->params = std::move(params);
        state->id = requestId;
        state->created = created;
        state->chat = false;
        streamResponse(res, std::move(state));
        return;
      }

      std::lock_guard<std::mutex> lock(modelMutex);
      auto [finishReason, nTokens, text] = runGeneration(params, nullptr);
      json resp = {
          {"id", requestId},
          {"object", "text_completion"},
          {"created", static_cast<int64_t>(created)},
          {"model", modelId},
          {"choices", json::array({json{{"index", 0},
                                        {"text", text},
                                        {"finish_reason", finishReason}}})},
          {"usage",
           {{"prompt_tokens", params.promptTokens.size()},
            {"completion_tokens", nTokens},
            {"total_tokens", params.promptTokens.size() + nTokens}}}};
      res.set_content(resp.dump(), "application/json");

    } catch (const json::exception &e) {
      res.status = 400;
      res.set_content(
          errorJson(std::string("Invalid JSON: ") + e.what()).dump(),
          "application/json");
    } catch (const std::exception &e) {
      log(std::string("  Error: ") + e.what());
      res.status = 500;
      res.set_content(errorJson(e.what()).dump(), "application/json");
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
