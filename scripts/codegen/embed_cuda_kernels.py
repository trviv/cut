#!/usr/bin/env python3
"""Embed native and transpiled CUDA kernels into CompiledCudaKernels.cpp.

For each supported kernel in the transpiler manifest that has a matching
compiled .spv, computes the normalized SPIR-V hash (identical algorithm to
core/backends/cuda/CudaSpirvKey.cpp) and emits a registry entry mapping that
hash to the kernel's CUDA source. When a native manifest (produced by
generate_shader_variants.py --cuda-native-manifest) is provided, hand-authored
.cu kernels are embedded as well — one shared source constant per distinct .cu
file — plus every shared .cuh header. The shared prelude and operator-enum
header are embedded as string constants for NVRTC.

The hash MUST stay in lockstep with cudaNormalizedSpirvHash() in C++.
"""

import argparse
import json
import os
import struct
import sys

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

OP_DECORATE = 71
OP_SPEC_CONSTANT = 50
DECORATION_SPEC_ID = 1
HEADER_SIZE = 5


def normalized_spirv_hash(spv_bytes):
    """Mirror of cudaNormalizedSpirvHash: zero spec literals, FNV-1a/64 over LE bytes."""
    if len(spv_bytes) < HEADER_SIZE * 4 or len(spv_bytes) % 4 != 0:
        return None
    words = list(struct.unpack("<%dI" % (len(spv_bytes) // 4), spv_bytes))

    spec_ids = set()
    i = HEADER_SIZE
    n = len(words)
    while i < n:
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0 or i + wc > n:
            break
        if op == OP_DECORATE and wc >= 4 and words[i + 2] == DECORATION_SPEC_ID:
            spec_ids.add(words[i + 1])
        i += wc

    i = HEADER_SIZE
    while i < n:
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0 or i + wc > n:
            break
        if op == OP_SPEC_CONSTANT and wc >= 4 and words[i + 2] in spec_ids:
            words[i + 3] = 0
        i += wc

    h = FNV_OFFSET
    for w in words:
        for b in range(4):
            h ^= (w >> (8 * b)) & 0xFF
            h = (h * FNV_PRIME) & MASK64
    return h


_DELIM = "CUTKERNELSRC"


def _raw(s):
    return f'R"{_DELIM}(' + s + f'){_DELIM}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cu-dir", required=True, help="Transpiled .cu + manifest dir")
    ap.add_argument("--spv-dir", required=True, help="Compiled .spv dir")
    ap.add_argument("--prelude", required=True, help="cut_cuda_prelude.cuh path")
    ap.add_argument("--enums", required=True, help="ComputeOpsShared.h path")
    ap.add_argument("--output", required=True, help="CompiledCudaKernels.cpp path")
    ap.add_argument("--native-manifest", default=None,
                    help="Optional native CUDA kernel manifest JSON")
    args = ap.parse_args()

    with open(os.path.join(args.cu_dir, "cuda_kernels.json")) as f:
        manifest = json.load(f)

    if args.native_manifest:
        with open(args.native_manifest) as f:
            native_manifest = json.load(f)
    else:
        native_manifest = {"headers": [], "kernels": {}}

    # Transpiled entries: (hash, src, entry, func)
    transpiled_entries = []
    seen = {}
    n_skip = 0
    for func, meta in sorted(manifest.items()):
        if not meta.get("supported"):
            continue
        spv = os.path.join(args.spv_dir, func + ".spv")
        cu = os.path.join(args.cu_dir, meta["cu"])
        if not os.path.exists(spv) or not os.path.exists(cu):
            n_skip += 1
            continue
        with open(spv, "rb") as f:
            h = normalized_spirv_hash(f.read())
        if h is None:
            n_skip += 1
            continue
        if h in seen:
            # Distinct kernels must not collide; first wins, warn.
            print(f"WARNING: hash collision {func} vs {seen[h]}", file=sys.stderr)
            continue
        seen[h] = func
        with open(cu) as f:
            src = f.read()
        transpiled_entries.append((h, src, meta["entry"], func))

    # Native entries: (hash, stem, src_const, entry, defines). Sources are
    # deduped — one shared raw-string constant per distinct .cu file.
    native_entries = []
    native_seen = {}
    native_cu_index = {}  # cu path -> constant index
    for stem in sorted(native_manifest["kernels"]):
        meta = native_manifest["kernels"][stem]
        spv = os.path.join(args.spv_dir, stem + ".spv")
        if not os.path.exists(spv):
            print(f"WARNING: missing .spv for native kernel {stem}",
                  file=sys.stderr)
            n_skip += 1
            continue
        with open(spv, "rb") as f:
            h = normalized_spirv_hash(f.read())
        if h is None:
            n_skip += 1
            continue
        if h in native_seen:
            print(f"WARNING: native hash collision {stem} vs {native_seen[h]}",
                  file=sys.stderr)
            continue
        native_seen[h] = stem
        cu_path = meta["cu"]
        if cu_path not in native_cu_index:
            native_cu_index[cu_path] = len(native_cu_index)
        defines_str = " ".join(meta.get("defines", []))
        native_entries.append((h, stem, f"kNativeSrc{native_cu_index[cu_path]}",
                               meta["entry"], defines_str))

    native_srcs = []  # (const_name, contents) in index order
    for cu_path, idx in sorted(native_cu_index.items(), key=lambda kv: kv[1]):
        with open(cu_path) as f:
            native_srcs.append((f"kNativeSrc{idx}", f.read()))

    native_headers = []  # (name, contents)
    header_paths = {}  # basename -> path (duplicate names break nvrtcCreateProgram)
    for header in native_manifest["headers"]:
        if header["name"] in header_paths:
            print(f"ERROR: duplicate header name '{header['name']}' from "
                  f"{header_paths[header['name']]} and {header['path']}; "
                  f"NVRTC rejects duplicate header names, which disables "
                  f"EVERY CUDA kernel at runtime. Rename one of the files.",
                  file=sys.stderr)
            sys.exit(1)
        header_paths[header["name"]] = header["path"]
        with open(header["path"]) as f:
            native_headers.append((header["name"], f.read()))

    with open(args.prelude) as f:
        prelude = f.read()
    with open(args.enums) as f:
        enums = f.read()

    out = []
    out.append("// Auto-generated by embed_cuda_kernels.py — do not edit")
    out.append('#include <CudaKernelRegistry.h>')
    out.append("")
    out.append("#include <cstdlib>")
    out.append("#include <cstring>")
    out.append("")
    out.append("namespace cut {")
    out.append("")
    out.append(f"static const char *kCudaPrelude = {_raw(prelude)};")
    out.append("")
    out.append(f"static const char *kCudaEnums = {_raw(enums)};")
    out.append("")

    # Native kernel sources (one shared constant per distinct .cu file).
    for const_name, contents in native_srcs:
        out.append(f"static const char *{const_name} = {_raw(contents)};")
        out.append("")

    if native_entries:
        out.append("static const CudaKernelEntry kNativeEntriesArr[] = {")
        for h, stem, src_const, entry, defines in native_entries:
            out.append(f"  // {stem}")
            out.append(f'  {{ {h}ULL, "{stem}", {src_const}, "{entry}", '
                       f'"{defines}", true }},')
        out.append("};")
        out.append("static const CudaKernelEntry *kNativeEntries = "
                   "kNativeEntriesArr;")
        out.append("static const size_t kNativeEntryCount = "
                   "sizeof(kNativeEntriesArr) / sizeof(kNativeEntriesArr[0]);")
    else:
        out.append("static const CudaKernelEntry *kNativeEntries = nullptr;")
        out.append("static const size_t kNativeEntryCount = 0;")
    out.append("")

    if transpiled_entries:
        out.append("static const CudaKernelEntry kTranspiledEntriesArr[] = {")
        for h, src, entry, func in transpiled_entries:
            out.append(f"  // {func}")
            out.append(f'  {{ {h}ULL, "{func}", {_raw(src)}, "{entry}", '
                       f'"", false }},')
        out.append("};")
        out.append("static const CudaKernelEntry *kTranspiledEntries = "
                   "kTranspiledEntriesArr;")
        out.append("static const size_t kTranspiledEntryCount = "
                   "sizeof(kTranspiledEntriesArr) / "
                   "sizeof(kTranspiledEntriesArr[0]);")
    else:
        out.append("static const CudaKernelEntry *kTranspiledEntries = nullptr;")
        out.append("static const size_t kTranspiledEntryCount = 0;")
    out.append("")

    # Shared .cuh header table (passed to every NVRTC compile).
    for i, (name, contents) in enumerate(native_headers):
        out.append(f"static const char *kCudaHdr{i} = {_raw(contents)};")
        out.append("")
    if native_headers:
        out.append("static const CudaKernelHeader kCudaHeadersArr[] = {")
        for i, (name, contents) in enumerate(native_headers):
            out.append(f'  {{ "{name}", kCudaHdr{i} }},')
        out.append("};")
        out.append("static const CudaKernelHeader *kCudaHeaders = "
                   "kCudaHeadersArr;")
        out.append("static const size_t kCudaHeaderCount = "
                   "sizeof(kCudaHeadersArr) / sizeof(kCudaHeadersArr[0]);")
    else:
        out.append("static const CudaKernelHeader *kCudaHeaders = nullptr;")
        out.append("static const size_t kCudaHeaderCount = 0;")
    out.append("")

    out.append("static bool cudaForceTranspiled() {")
    out.append("  static const bool forced = [] {")
    out.append('    const char *env = std::getenv("CUT_CUDA_KERNELS");')
    out.append('    return env != nullptr && std::strcmp(env, "transpiled") == 0;')
    out.append("  }();")
    out.append("  return forced;")
    out.append("}")
    out.append("")
    out.append("const CudaKernelEntry *lookupCudaKernelByHash(uint64_t hash) {")
    out.append("  if (!cudaForceTranspiled()) {")
    out.append("    for (size_t i = 0; i < kNativeEntryCount; ++i) {")
    out.append("      if (kNativeEntries[i].hash == hash) return &kNativeEntries[i];")
    out.append("    }")
    out.append("  }")
    out.append("  for (size_t i = 0; i < kTranspiledEntryCount; ++i) {")
    out.append("    if (kTranspiledEntries[i].hash == hash) return &kTranspiledEntries[i];")
    out.append("  }")
    out.append("  return nullptr;")
    out.append("}")
    out.append("")
    out.append("const CudaKernelEntry *lookupCudaKernelByName(const char *name) {")
    out.append("  if (!cudaForceTranspiled()) {")
    out.append("    for (size_t i = 0; i < kNativeEntryCount; ++i) {")
    out.append("      if (std::strcmp(kNativeEntries[i].name, name) == 0)")
    out.append("        return &kNativeEntries[i];")
    out.append("    }")
    out.append("  }")
    out.append("  for (size_t i = 0; i < kTranspiledEntryCount; ++i) {")
    out.append("    if (std::strcmp(kTranspiledEntries[i].name, name) == 0)")
    out.append("      return &kTranspiledEntries[i];")
    out.append("  }")
    out.append("  return nullptr;")
    out.append("}")
    out.append("")
    out.append("size_t cudaKernelCount() {")
    out.append("  return kNativeEntryCount + kTranspiledEntryCount;")
    out.append("}")
    out.append("size_t cudaKernelHeaderCount() { return kCudaHeaderCount; }")
    out.append("const CudaKernelHeader *cudaKernelHeader(size_t index) {")
    out.append("  return index < kCudaHeaderCount ? &kCudaHeaders[index] : nullptr;")
    out.append("}")
    out.append("const char *cudaPreludeSource() { return kCudaPrelude; }")
    out.append("const char *cudaEnumsSource() { return kCudaEnums; }")
    out.append("")
    out.append("} // namespace cut")
    out.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(out))

    # Coverage summary over every compiled .spv stem. Stems whose normalized
    # hash collides with an embedded entry are covered by that entry (dtype
    # differences live in spec constants), so classify by hash, not by name.
    n_native = 0
    n_transpiled_only = 0
    n_unsupported = 0
    for fn in sorted(os.listdir(args.spv_dir)):
        if not fn.endswith(".spv"):
            continue
        with open(os.path.join(args.spv_dir, fn), "rb") as f:
            h = normalized_spirv_hash(f.read())
        if h in native_seen:
            n_native += 1
        elif h in seen:
            n_transpiled_only += 1
        else:
            n_unsupported += 1

    print(f"Embedded {len(native_entries) + len(transpiled_entries)} "
          f"CUDA kernel(s), skipped {n_skip}.")
    print(f"CUDA kernel coverage: {n_native} native, "
          f"{n_transpiled_only} transpiled-only, {n_unsupported} unsupported.")


if __name__ == "__main__":
    main()
