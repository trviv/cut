#include "ltx.h"
#include "Runtime.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
Binary file formats:
- Embedding file (from scripts/ltx/ltx_encode_prompt.py): uint32 magic 0x4C545845 ("LTXE"),
  uint32 nTokens, uint32 dim, then nTokens*dim float32 little-endian.
- Latent output file (for scripts/ltx/ltx_decode_latents.py): uint32 magic 0x4C54584C ("LTXL"),
  uint32 latentFrames, uint32 latentHeight, uint32 latentWidth, uint32 channels, then
  S*channels float32 (row-major [S, channels], token order frame-major then row then column).
*/

static bool readEmbeddingFile(const std::string &path, std::vector<float> &data,
                             uint32_t &nTokens, uint32_t &dim) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t magic;
  f.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != 0x4C545845U) {
    std::cerr << "Invalid embedding file magic (expected LTXE)\n";
    return false;
  }
  f.read(reinterpret_cast<char*>(&nTokens), 4);
  f.read(reinterpret_cast<char*>(&dim), 4);
  data.resize(nTokens * dim);
  f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return f.good();
}

static bool readLatentFile(const std::string &path, std::vector<float> &data,
                           uint32_t &lf, uint32_t &lh, uint32_t &lw,
                           uint32_t &lc) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  uint32_t magic;
  in.read(reinterpret_cast<char*>(&magic), 4);
  if (magic != 0x4C54584CU) {
    std::cerr << "Invalid latent file magic (expected LTXL)\n";
    return false;
  }
  in.read(reinterpret_cast<char*>(&lf), 4);
  in.read(reinterpret_cast<char*>(&lh), 4);
  in.read(reinterpret_cast<char*>(&lw), 4);
  in.read(reinterpret_cast<char*>(&lc), 4);
  data.resize(static_cast<size_t>(lf) * lh * lw * lc);
  in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
  return in.good();
}

static void writeLatentFile(const std::string &path, const std::vector<float> &data,
                           uint32_t latentFrames, uint32_t latentHeight,
                           uint32_t latentWidth, uint32_t channels) {
  std::ofstream f(path, std::ios::binary);
  uint32_t magic = 0x4C54584CU;
  f.write(reinterpret_cast<const char*>(&magic), 4);
  f.write(reinterpret_cast<const char*>(&latentFrames), 4);
  f.write(reinterpret_cast<const char*>(&latentHeight), 4);
  f.write(reinterpret_cast<const char*>(&latentWidth), 4);
  f.write(reinterpret_cast<const char*>(&channels), 4);
  f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char **argv) {
  std::string modelDir;
  std::string promptEmb, negativeEmb, initLatentsPath;
  std::string out = "latents.bin";
  uint32_t frames = 49, height = 512, width = 768, steps = 30, seed = 42;
  float guidance = 3.0f, fps = 25.0f;

  if (argc < 2) {
    std::cerr << "usage: ltx_example <modelDir> --prompt-emb pos.bin [--negative-emb neg.bin] [--frames N] [--height H] [--width W] [--steps S] [--guidance G] [--fps F] [--seed S] [--out PATH]\n";
    return 1;
  }
  modelDir = argv[1];

  for (int i = 2; i < argc; ) {
    std::string arg = argv[i];
    if (arg == "--prompt-emb" && i + 1 < argc) {
      promptEmb = argv[++i];
    } else if (arg == "--negative-emb" && i + 1 < argc) {
      negativeEmb = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (arg == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (arg == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::atoi(argv[++i]);
    } else if (arg == "--guidance" && i + 1 < argc) {
      guidance = std::atof(argv[++i]);
    } else if (arg == "--fps" && i + 1 < argc) {
      fps = std::atof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (arg == "--init-latents" && i + 1 < argc) {
      initLatentsPath = argv[++i];
    } else {
      ++i;
    }
  }

  if (promptEmb.empty()) {
    std::cerr << "usage: ltx_example <modelDir> --prompt-emb pos.bin [--negative-emb neg.bin] [--frames N] ...\n";
    return 1;
  }

  if (height % 32 != 0 || width % 32 != 0 || (frames - 1) % 8 != 0) {
    std::cerr << "frames must be 8k+1, height/width multiples of 32\n";
    return 1;
  }

  uint32_t lf = (frames - 1) / 8 + 1;
  uint32_t lh = height / 32;
  uint32_t lw = width / 32;

  try {
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
        std::string backend = entry.substr(0, colon == std::string::npos ? entry.size() : colon);
        if (colon != std::string::npos) {
          desc.deviceIndex = std::atoi(entry.substr(colon + 1).c_str());
        }
        desc.backend = (backend == "cuda") ? cut::BackendType::CUDA : cut::BackendType::Vulkan;
        descs.push_back(desc);
      }
      runtime.init(descs);
      std::cout << "Initialized " << runtime.deviceCount() << " devices\n";
    } else {
      runtime.init(cut::BackendType::Vulkan);
    }

    if (std::getenv("CUT_LTX_PROFILE")) {
      runtime.setProfilingEnabled(true);
    }

    std::vector<float> promptData, negData;
    uint32_t nTokensP = 0, dimP = 0, nTokensN = 0, dimN = 0;
    if (!readEmbeddingFile(promptEmb, promptData, nTokensP, dimP)) {
      std::cerr << "Failed to read prompt embedding\n";
      return 1;
    }
    if (dimP != 4096) {
      std::cerr << "embedding dim mismatch: expected 4096\n";
      return 1;
    }
    if (!negativeEmb.empty() && !readEmbeddingFile(negativeEmb, negData, nTokensN, dimN)) {
      std::cerr << "Failed to read negative embedding\n";
      return 1;
    }
    if (dimN != 4096 && !negData.empty()) {
      std::cerr << "embedding dim mismatch: expected 4096\n";
      return 1;
    }

    ltx::LtxModel model;
    model.load(modelDir, runtime);

    std::cout << "Generating: " << frames << "f " << height << "x" << width
              << ", tokens " << nTokensP << " (neg " << nTokensN << "), "
              << steps << " steps, guidance " << guidance << ", seed " << seed << "\n";

    std::vector<float> initData;
    if (!initLatentsPath.empty()) {
      uint32_t rf, rh, rw, rc;
      if (!readLatentFile(initLatentsPath, initData, rf, rh, rw, rc)) {
        std::cerr << "Failed to read init latents\n";
        return 1;
      }
      if (rf != lf || rh != lh || rw != lw ||
          rc != model.config().inChannels) {
        std::cerr << "init latents dims mismatch\n";
        return 1;
      }
    }

    auto latents = model.generate(promptData, nTokensP, negData, nTokensN,
                                  lf, lh, lw, fps, steps, guidance, seed,
                                  initData.empty() ? nullptr : &initData);

    writeLatentFile(out, latents, lf, lh, lw, model.config().inChannels);
    std::cout << "Wrote latents: " << out << " (" << latents.size() * sizeof(float) << " bytes)\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
