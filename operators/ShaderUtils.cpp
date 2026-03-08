#include "ShaderUtils.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/linker.hpp>

namespace cut {

namespace {

// SPIR-V opcodes
constexpr uint32_t kOpName = 5;
constexpr uint32_t kOpEntryPoint = 15;
constexpr uint32_t kOpExecutionMode = 16;
constexpr uint32_t kOpCapability = 17;
constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpMemberDecorate = 72;
constexpr uint32_t kOpFunction = 54;
constexpr uint32_t kOpFunctionParameter = 55;
constexpr uint32_t kOpFunctionEnd = 56;

// SPIR-V constants
constexpr uint32_t kCapabilityLinkage = 5;
constexpr uint32_t kDecorationLinkageAttributes = 41;
constexpr uint32_t kLinkageExport = 0;
constexpr uint32_t kLinkageImport = 1;

// Helpers to decode a SPIR-V instruction header word.
uint32_t spvOp(uint32_t w) {
  return w & 0xFFFF;
}
uint32_t spvWc(uint32_t w) {
  return w >> 16;
}

// Encode a string as null-terminated, 4-byte-padded SPIR-V literal words.
std::vector<uint32_t> packString(const std::string &s) {
  size_t n = (s.size() + 1 + 3) / 4;
  std::vector<uint32_t> v(n, 0);
  std::memcpy(v.data(), s.c_str(), s.size() + 1);
  return v;
}

// Decode a SPIR-V literal string starting at |p| spanning at most |maxWords|.
std::string unpackString(const uint32_t *p, size_t maxWords) {
  auto *c = reinterpret_cast<const char *>(p);
  return {c, strnlen(c, maxWords * 4)};
}

// Find the SPIR-V result-ID decorated with OpName |name|.
uint32_t findName(const std::vector<uint32_t> &s, const std::string &name) {
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    if (spvOp(s[i]) == kOpName && w >= 3 &&
        unpackString(&s[i + 2], w - 2) == name)
      return s[i + 1];
    i += w;
  }
  return 0;
}

// Find the function result-ID of the first OpEntryPoint.
uint32_t findEntryPoint(const std::vector<uint32_t> &s) {
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    if (spvOp(s[i]) == kOpEntryPoint && w >= 3)
      return s[i + 2];
    i += w;
  }
  return 0;
}

// Insert OpCapability |cap| after the existing capability block.
void addCapability(std::vector<uint32_t> &s, uint32_t cap) {
  size_t pos = 5;
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w)
      break;
    if (spvOp(s[i]) != kOpCapability)
      break;
    pos = i + w;
    i += w;
  }
  s.insert(s.begin() + pos, {(2u << 16) | kOpCapability, cap});
}

// Insert an OpDecorate LinkageAttributes instruction after existing
// decorations.
void addLinkage(std::vector<uint32_t> &s,
                uint32_t id,
                const std::string &name,
                uint32_t linkType) {
  auto nameW = packString(name);
  uint32_t total = 3 + static_cast<uint32_t>(nameW.size()) + 1;
  std::vector<uint32_t> d;
  d.push_back((total << 16) | kOpDecorate);
  d.push_back(id);
  d.push_back(kDecorationLinkageAttributes);
  d.insert(d.end(), nameW.begin(), nameW.end());
  d.push_back(linkType);

  // Find insertion point: after last OpDecorate/OpMemberDecorate, before
  // OpFunction.
  size_t pos = 5;
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    uint32_t o = spvOp(s[i]);
    if (o == kOpDecorate || o == kOpMemberDecorate)
      pos = i + w;
    if (o == kOpFunction)
      break;
    i += w;
  }
  s.insert(s.begin() + pos, d.begin(), d.end());
}

// Remove every instruction whose opcode appears in |opcodes|.
void removeOps(std::vector<uint32_t> &s, const std::vector<uint32_t> &opcodes) {
  size_t i = 5;
  while (i < s.size()) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    uint32_t o = spvOp(s[i]);
    bool rm = false;
    for (auto t : opcodes)
      if (o == t) {
        rm = true;
        break;
      }
    if (rm)
      s.erase(s.begin() + i, s.begin() + i + w);
    else
      i += w;
  }
}

// Strip the body of a function, leaving it as a declaration
// (OpFunction + OpFunctionParameter* + OpFunctionEnd, no OpLabel or body).
void stripBody(std::vector<uint32_t> &s, uint32_t funcId) {
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    if (spvOp(s[i]) == kOpFunction && w >= 3 && s[i + 2] == funcId) {
      i += w;
      // Skip OpFunctionParameter instructions.
      while (i < s.size() && spvOp(s[i]) == kOpFunctionParameter)
        i += spvWc(s[i]);
      // Erase body instructions up to (but not including) OpFunctionEnd.
      size_t bodyStart = i;
      while (i < s.size() && spvOp(s[i]) != kOpFunctionEnd)
        i += spvWc(s[i]);
      if (bodyStart < i)
        s.erase(s.begin() + bodyStart, s.begin() + i);
      return;
    }
    i += w;
  }
}

// Remove an entire function (OpFunction through OpFunctionEnd inclusive).
void removeFunc(std::vector<uint32_t> &s, uint32_t funcId) {
  for (size_t i = 5; i < s.size();) {
    uint32_t w = spvWc(s[i]);
    if (!w || i + w > s.size())
      break;
    if (spvOp(s[i]) == kOpFunction && w >= 3 && s[i + 2] == funcId) {
      size_t start = i;
      while (i < s.size()) {
        bool end = spvOp(s[i]) == kOpFunctionEnd;
        i += spvWc(s[i]);
        if (end)
          break;
      }
      s.erase(s.begin() + start, s.begin() + i);
      return;
    }
    i += w;
  }
}

} // anonymous namespace

std::vector<uint32_t>
compileCustomShader(const std::vector<uint32_t> &templateSpirv,
                    const std::string &templateFuncName,
                    const std::vector<uint32_t> &moduleSpirv,
                    const std::string &moduleFuncName) {
  auto tmpl = templateSpirv;
  auto mod = moduleSpirv;

  // 1. Patch template: strip the stub function body, mark as Import.
  uint32_t tmplFuncId = findName(tmpl, templateFuncName);
  if (!tmplFuncId)
    throw std::runtime_error("'" + templateFuncName +
                             "' not found in template SPIR-V");
  addCapability(tmpl, kCapabilityLinkage);
  stripBody(tmpl, tmplFuncId);
  addLinkage(tmpl, tmplFuncId, templateFuncName, kLinkageImport);

  // 2. Patch module: mark the replacement function as Export, remove entry
  //    point and its function so the module acts as a linkable library.
  uint32_t modFuncId = findName(mod, moduleFuncName);
  if (!modFuncId)
    throw std::runtime_error("'" + moduleFuncName +
                             "' not found in module SPIR-V");
  uint32_t modEntryId = findEntryPoint(mod);

  addCapability(mod, kCapabilityLinkage);
  addLinkage(mod, modFuncId, templateFuncName, kLinkageExport);
  removeOps(mod, {kOpEntryPoint, kOpExecutionMode});
  if (modEntryId && modEntryId != modFuncId)
    removeFunc(mod, modEntryId);

  // 3. Link both modules.
  // Use Vulkan 1.1 environment (matches VK_API_VERSION_1_1 in runtime).
  // The linker's SetUseHighestVersion merges SPIR-V versions from both
  // modules; Vulkan 1.1 supports SPIR-V up to 1.3 which DXC may emit.
  spvtools::Context ctx(SPV_ENV_VULKAN_1_1);
  std::vector<uint32_t> linked;
  spvtools::LinkerOptions lopts;
  lopts.SetAllowPtrTypeMismatch(true);
  lopts.SetUseHighestVersion(true);

  auto result = spvtools::Link(ctx, {tmpl, mod}, &linked, lopts);
  if (result != SPV_SUCCESS)
    throw std::runtime_error("SPIR-V linking failed (error " +
                             std::to_string(result) + ")");

  // Validate linked SPIR-V and log any issues
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
  tools.SetMessageConsumer([&](spv_message_level_t level, const char *,
                               const spv_position_t &, const char *msg) {
    if (level <= SPV_MSG_ERROR) {
      fprintf(stderr,
              "[ShaderUtils] SPIR-V validation error after linking "
              "'%s' <- '%s': %s\n",
              templateFuncName.c_str(), moduleFuncName.c_str(), msg);
    }
  });
  if (!tools.Validate(linked)) {
    fprintf(stderr,
            "[ShaderUtils] Linked SPIR-V is INVALID (%s <- %s). "
            "Template had func ID %u, module had func ID %u\n",
            templateFuncName.c_str(), moduleFuncName.c_str(), tmplFuncId,
            modFuncId);

    // Dump SPIR-V disassembly for debugging
    std::string disassembly;
    if (tools.Disassemble(linked, &disassembly)) {
      fprintf(stderr, "[ShaderUtils] Linked SPIR-V disassembly:\n%s\n",
              disassembly.c_str());
    }
  }

  return linked;
}

} // namespace cut
