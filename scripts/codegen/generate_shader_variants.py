#!/usr/bin/env python3
"""Generic shader variant generator.

Scans subdirectories of --impl-dir for shaders.json files and produces:

1. Dtype-preprocessed .shader files in --output-dir (one per variant per dtype)
2. {Group}Variants.generated.h with dispatch table and lookup functions
   (for groups that have wg/eff_tile metadata)
3. {Group}Shaders.generated.h with forward declarations
   (for groups without dispatch metadata)
4. generated_shaders.cmake manifest for the build system

JSON format (unified for all shader groups):

Single-group format:
{
  "type": "variants",
  "cpp_prefix": "MatMul",           // prepended to variant name for full name
  "default_variant": "T16R4x4",     // optional, suffix form
  "variants": [...]
}

Multi-group format (multiple groups in one directory):
{
  "groups": [
    { "cpp_prefix": "MatMul", "default_variant": "T8R4x4", "variants": [...] },
    { "cpp_prefix": "MatMulQ4", "default_variant": "T16R4x4", "variants": [...] }
  ]
}

Variant entry:
    {
      "name": "Naive",              // suffix after cpp_prefix (can be "")
      "description": "...",
      "dtype_slots": {              // dtype configuration
        "combos": [                 // list of dtype combinations
          {"input": "Float32", "output": "Float32"},
          {"input": "Float16", "output": "Float32"}
        ]
      },
      "template": "MatMulNaive",    // optional: generate from template
      "defines": {},                // optional: template substitutions
      "wg": [16, 16],              // optional: workgroup size
      "eff_tile": [16, 16]         // optional: effective tile size
    }
"""

import argparse
import hashlib
import json
import os
import re
import sys


# =============================================================================
# Datatype definitions
# =============================================================================

DTYPE_DEFS = {
    "Float32": {"vec": "float4", "scalar": "float", "size": "4"},
    "Float16": {"vec": "half4",  "scalar": "half",  "size": "4"},
    "Int32":   {"vec": "int4",   "scalar": "int",   "size": "4"},
    "UInt32":  {"vec": "uint4",  "scalar": "uint",  "size": "4"},
    "Int8":    {"vec": "int4",   "scalar": "int",   "size": "1"},
}

# Mapping from dtype name to per-slot define lines.
# The slot name is uppercased and inserted: DTYPE_{SLOT}_IS_FLOAT, etc.
_SLOT_DEFINE_TEMPLATES = {
    "Float32": ["#define DTYPE_{SLOT}_IS_FLOAT 1"],
    "Float16": ["#define DTYPE_{SLOT}_IS_FLOAT 1",
                 "#define DTYPE_{SLOT}_IS_HALF 1"],
    "Int32":   ["#define DTYPE_{SLOT}_IS_INT 1"],
    "UInt32":  ["#define DTYPE_{SLOT}_IS_UINT 1"],
    "Int8":    ["#define DTYPE_{SLOT}_IS_INT8 1"],
}


def substitute_dtype_slot(content, slot_name, dtype_name):
    """Replace slot-specific dtype placeholders for a named slot.

    For slot 'input' and dtype 'Float32', replaces:
      %VEC_DTYPE_INPUT%    -> float4
      %SCALAR_DTYPE_INPUT% -> float
      %DTYPE_SIZE_INPUT%   -> 4
      %DTYPE_DEFINES_INPUT% -> #define DTYPE_INPUT_IS_FLOAT 1
    """
    d = DTYPE_DEFS[dtype_name]
    suffix = "_" + slot_name.upper()
    content = content.replace(f"%VEC_DTYPE{suffix}%", d["vec"])
    content = content.replace(f"%SCALAR_DTYPE{suffix}%", d["scalar"])
    content = content.replace(f"%DTYPE_SIZE{suffix}%", d["size"])
    slot_defines = "\n".join(
        t.replace("{SLOT}", slot_name.upper())
        for t in _SLOT_DEFINE_TEMPLATES[dtype_name]
    )
    content = content.replace(f"%DTYPE_DEFINES{suffix}%", slot_defines)
    return content


_SHADERH_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+\.shaderh)"\s*$')


def resolve_shaderh_includes(content, source_dir, impl_dir, include_stack=None):
    """Inline #include "*.shaderh" directives before dtype substitution.

    Searches source_dir first, then impl_dir. Recurses for nested includes.
    """
    if include_stack is None:
        include_stack = set()

    lines = content.split("\n")
    result = []
    for line in lines:
        m = _SHADERH_INCLUDE_RE.match(line)
        if not m:
            result.append(line)
            continue

        include_name = m.group(1)

        # Resolve: source_dir first, then impl_dir
        candidate = os.path.join(source_dir, include_name)
        if not os.path.exists(candidate):
            candidate = os.path.join(impl_dir, include_name)
        if not os.path.exists(candidate):
            print(f"ERROR: Shader header not found: {include_name} "
                  f"(searched {source_dir} and {impl_dir})", file=sys.stderr)
            sys.exit(1)

        real_path = os.path.realpath(candidate)
        if real_path in include_stack:
            print(f"ERROR: Circular include detected: {include_name}",
                  file=sys.stderr)
            sys.exit(1)

        with open(candidate, "r") as f:
            included_content = f.read()

        include_stack.add(real_path)
        included_content = resolve_shaderh_includes(
            included_content, os.path.dirname(real_path), impl_dir,
            include_stack)
        include_stack.discard(real_path)

        result.append(included_content)

    return "\n".join(result)


def write_if_changed(path, content):
    """Write file only if content differs from existing. Returns True if written."""
    if os.path.exists(path):
        with open(path, "r") as f:
            if f.read() == content:
                return False
    with open(path, "w") as f:
        f.write(content)
    return True


def full_name_for(cpp_prefix, variant_name):
    """Compute full shader name: cpp_prefix + name."""
    return cpp_prefix + variant_name


def _native_defines_for_combo(slots, combo, variant):
    """Build the NVRTC define list (KEY=VAL strings) for one dtype combo."""
    defines = []
    for slot in slots:
        d = DTYPE_DEFS[combo[slot]]
        s = slot.upper()
        defines.append(f"CUT_VEC_DTYPE_{s}={d['vec']}")
        defines.append(f"CUT_SCALAR_DTYPE_{s}={d['scalar']}")
        defines.append(f"CUT_DTYPE_SIZE_{s}={d['size']}")
        for tmpl in _SLOT_DEFINE_TEMPLATES[combo[slot]]:
            # "#define DTYPE_{SLOT}_IS_FLOAT 1" -> "CUT_DTYPE_INPUT_IS_FLOAT=1"
            token = tmpl.replace("#define ", "").replace("{SLOT}", s)
            name, value = token.split()
            defines.append(f"CUT_{name}={value}")
    for key, value in variant.get("defines", {}).items():
        defines.append(f"{key}={value}")
    return defines


# =============================================================================
# Shader file generation (template expansion + dtype preprocessing)
# =============================================================================

def _shader_extension(variant):
    """Return the source file extension based on compiler type."""
    return ".comp" if variant.get("compiler") == "glsl" else ".shader"


def _output_extension(variant):
    """Return the output file extension based on compiler type."""
    return _shader_extension(variant)


def _load_shader_content(variant, fname, group_dir):
    """Load base shader content for a variant (template or direct file)."""
    ext = _shader_extension(variant)
    if "template" in variant:
        template_path = os.path.join(group_dir,
                                     variant["template"] + ext)
        if not os.path.exists(template_path):
            print(f"ERROR: Shader template not found: {template_path}",
                  file=sys.stderr)
            sys.exit(1)
        with open(template_path, "r") as f:
            content = f.read()
        for key, value in variant.get("defines", {}).items():
            content = content.replace(f"%{key}%", str(value))
        return content

    shader_path = os.path.join(group_dir, fname + ext)
    if not os.path.exists(shader_path):
        print(f"ERROR: Shader source not found: {shader_path}",
              file=sys.stderr)
        sys.exit(1)
    with open(shader_path, "r") as f:
        return f.read()


def generate_shader_files(config, group_dir, output_dir, impl_dir,
                          native_kernels=None):
    """Generate dtype-preprocessed .shader files for all variants.

    Returns a list of (full_name, dtype_suffix, output_path, source_hash, slots)
    tuples.  slots is the ordered list of slot names and dtype_suffix is the
    dtypes in slot order joined by "_", e.g. "Float32_Float32".

    When native_kernels is a dict, variants whose source dir contains a native
    CUDA counterpart (<template-or-variant-source>.cu) get one manifest entry
    per dtype combo: stem -> {cu, entry, defines}.
    """
    cpp_prefix = config.get("cpp_prefix", "")
    generated = []

    for variant in config["variants"]:
        name = variant["name"]
        fname = full_name_for(cpp_prefix, name)

        base_content = _load_shader_content(variant, fname, group_dir)

        # Resolve .shaderh includes before dtype substitution (HLSL only)
        if variant.get("compiler") != "glsl":
            base_content = resolve_shaderh_includes(base_content, group_dir,
                                                    impl_dir)

        # Compute source hash (after include resolution, before dtype substitution)
        source_hash = hashlib.md5(base_content.encode()).hexdigest()

        # Native CUDA counterpart: resolved the same way as the shader source,
        # but with a .cu extension (no dtype preprocessing of its contents).
        cu_stem = variant["template"] if "template" in variant else fname
        cu_path = os.path.join(group_dir, cu_stem + ".cu")
        has_native_cu = os.path.exists(cu_path)

        ds = variant["dtype_slots"]
        combos = ds["combos"]
        # Derive slot names from the first combo's keys (preserves insertion order)
        slots = list(combos[0].keys())

        for combo in combos:
            for slot in slots:
                dtype = combo[slot]
                if dtype not in DTYPE_DEFS:
                    print(f"ERROR: Unknown dtype '{dtype}' for slot "
                          f"'{slot}' in variant '{fname}'",
                          file=sys.stderr)
                    sys.exit(1)

            preprocessed = base_content
            for slot in slots:
                preprocessed = substitute_dtype_slot(
                    preprocessed, slot, combo[slot])

            suffix = "_".join(combo[slot] for slot in slots)
            ext = _output_extension(variant)
            out_path = os.path.join(output_dir,
                                    f"{fname}_{suffix}{ext}")

            if write_if_changed(out_path, preprocessed):
                print(f"  Generated {fname}_{suffix}.shader")

            generated.append(
                (fname, suffix, out_path, source_hash, slots))

            if native_kernels is not None and has_native_cu:
                native_kernels[f"{fname}_{suffix}"] = {
                    "cu": os.path.abspath(cu_path),
                    "entry": "cut_main",
                    "defines": _native_defines_for_combo(slots, combo,
                                                         variant),
                }

    return generated


# =============================================================================
# Header generation: dispatch table (matmul-style with wg/eff_tile)
# =============================================================================

def has_dispatch_metadata(config):
    """Check if any variant has wg/eff_tile (matmul-style dispatch table)."""
    return any("wg" in v and "eff_tile" in v for v in config["variants"])


def _slot_params(slots):
    """Build C++ parameter list from slot names, e.g. 'DataType input, DataType output'."""
    return ", ".join(f"DataType {s}" for s in slots)


def _slot_defaults(slots):
    """Build C++ parameter list with defaults, e.g. 'DataType input = ..., DataType output = ...'."""
    return ", ".join(
        f"DataType {s} = DataType::Float32" for s in slots)


def generate_variant_header(config, group_dir, group_name):
    """Generate {Prefix}Variants.generated.h with dispatch table and functions."""
    variants = config["variants"]
    count = len(variants)
    cpp_prefix = config.get("cpp_prefix", "")
    cap = cpp_prefix if cpp_prefix else (group_name[0].upper() + group_name[1:])

    # Derive slot names from first variant
    slots = list(variants[0]["dtype_slots"]["combos"][0].keys())
    params = _slot_params(slots)
    params_with_defaults = _slot_defaults(slots)

    # Find default variant index
    default_name = config.get("default_variant", "")
    default_index = -1
    for i, v in enumerate(variants):
        if v["name"] == default_name:
            default_index = i
            break
    if default_index < 0:
        if default_name:
            print(f"WARNING: default_variant '{default_name}' not found, using 0",
                  file=sys.stderr)
        default_index = 0

    lines = []
    lines.append("// Auto-generated by generate_shader_variants.py — do not edit")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <ComputeCommon.h>")
    lines.append("#include <cstdint>")
    lines.append("#include <optional>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace cut {")
    lines.append("")

    # Variant info struct
    lines.append(f"struct {cap}VariantInfo {{")
    lines.append("    const char* name;")
    lines.append("    uint32_t wgX;")
    lines.append("    uint32_t wgY;")
    lines.append("    uint32_t effTileM;")
    lines.append("    uint32_t effTileN;")
    lines.append("    const char* description;")
    lines.append("};")
    lines.append("")

    # Constants
    lines.append(f"inline constexpr int k{cap}VariantCount = {count};")
    lines.append(f"inline constexpr int k{cap}DefaultVariant = {default_index};")
    lines.append("")

    # Dispatch table
    lines.append(
        f"inline constexpr {cap}VariantInfo "
        f"k{cap}Variants[k{cap}VariantCount] = {{"
    )
    for v in variants:
        wg = v["wg"]
        eff = v["eff_tile"]
        fname = full_name_for(cpp_prefix, v["name"])
        desc = v.get("description", fname)
        lines.append(
            f'    {{"{fname}", {wg[0]}, {wg[1]}, '
            f'{eff[0]}, {eff[1]}, "{desc}"}},'
        )
    lines.append("};")
    lines.append("")

    # Forward declarations for compiled shader functions
    lines.append("// Forward declarations (defined in CompiledShaders.cpp)")
    for v in variants:
        fname = full_name_for(cpp_prefix, v["name"])
        lines.append(
            f"std::optional<std::vector<uint32_t>> "
            f"compiled{fname}({params});"
        )
    lines.append("")

    # Function pointer type and lookup table
    dt_list = ", ".join("DataType" for _ in slots)
    lines.append(
        f"using Compiled{cap}Fn = "
        f"std::optional<std::vector<uint32_t>> (*)({dt_list});"
    )
    lines.append("")
    lines.append(
        f"inline const Compiled{cap}Fn "
        f"k{cap}CompiledFns[k{cap}VariantCount] = {{"
    )
    for v in variants:
        fname = full_name_for(cpp_prefix, v["name"])
        lines.append(f"    compiled{fname},")
    lines.append("};")
    lines.append("")

    # Inline helper functions
    call_args = ", ".join(slots)
    lines.append(f"/// Returns compiled SPIR-V for a {group_name} variant by index.")
    lines.append("inline std::optional<std::vector<uint32_t>>")
    lines.append(
        f"getCompiled{cap}(int variantIndex, "
        f"{params_with_defaults}) {{"
    )
    lines.append(
        f"    if (variantIndex < 0 || variantIndex >= k{cap}VariantCount)"
    )
    lines.append("        return std::nullopt;")
    lines.append(f"    return k{cap}CompiledFns[variantIndex]({call_args});")
    lines.append("}")
    lines.append("")

    lines.append(f"/// Returns the number of {group_name} variants.")
    lines.append(f"inline int get{cap}VariantCount() {{")
    lines.append(f"    return k{cap}VariantCount;")
    lines.append("}")
    lines.append("")

    lines.append(f"/// Returns the name of a {group_name} variant by index.")
    lines.append(f"inline const char* get{cap}VariantName(int variantIndex) {{")
    lines.append(
        f"    if (variantIndex < 0 || variantIndex >= k{cap}VariantCount)"
    )
    lines.append('        return "Unknown";')
    lines.append(f"    return k{cap}Variants[variantIndex].name;")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace cut")
    lines.append("")

    header_name = f"{cap}Variants.generated.h"
    header_path = os.path.join(group_dir, header_name)
    content = "\n".join(lines)

    if write_if_changed(header_path, content):
        print(f"  Generated {header_name}")
    else:
        print(f"  {header_name} unchanged")


# =============================================================================
# Header generation: simple forward declarations
# =============================================================================

def generate_simple_header(config, group_dir, group_name):
    """Generate {Group}Shaders.generated.h with forward declarations."""
    cpp_prefix = config.get("cpp_prefix", "")
    cap = group_name[0].upper() + group_name[1:]

    lines = []
    lines.append("// Auto-generated by generate_shader_variants.py — do not edit")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <ComputeCommon.h>")
    lines.append("#include <optional>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace cut {")
    lines.append("")

    for v in config["variants"]:
        fname = full_name_for(cpp_prefix, v["name"])
        slots = list(v["dtype_slots"]["combos"][0].keys())
        params = _slot_params(slots)
        lines.append(
            f"std::optional<std::vector<uint32_t>> "
            f"compiled{fname}({params});"
        )

    lines.append("")
    lines.append("} // namespace cut")
    lines.append("")

    header_name = f"{cap}Shaders.generated.h"
    header_path = os.path.join(group_dir, header_name)
    content = "\n".join(lines)

    if write_if_changed(header_path, content):
        print(f"  Generated {header_name}")
    else:
        print(f"  {header_name} unchanged")


# =============================================================================
# CMake manifest generation
# =============================================================================

def generate_cmake_manifest(all_generated, output_dir):
    """Write generated_shaders.cmake with file list and per-function dtype info."""
    # Collect per-function dtype lists, source hashes, and slot info
    func_dtypes = {}
    func_hashes = {}
    func_slots = {}
    func_order = []
    all_files = []

    for full_name, dtype_suffix, out_path, source_hash, slots in all_generated:
        all_files.append(out_path)
        if full_name not in func_dtypes:
            func_dtypes[full_name] = []
            func_hashes[full_name] = source_hash
            func_slots[full_name] = slots
            func_order.append(full_name)
        func_dtypes[full_name].append(dtype_suffix)

    lines = []
    lines.append("# Auto-generated by generate_shader_variants.py — do not edit")
    lines.append("")

    # List of all preprocessed shader files
    lines.append("set(GENERATED_SHADER_FILES")
    for path in all_files:
        lines.append(f"    {path}")
    lines.append(")")
    lines.append("")

    # Per-function dtype info for CompiledShaders.cpp generation
    # Format: "FunctionName|D1_D2,D1_D2,...|source_hash|slots:s1,s2"
    lines.append("set(SHADER_FUNCTION_DTYPES")
    for full_name in func_order:
        dtypes_str = ",".join(func_dtypes[full_name])
        src_hash = func_hashes[full_name]
        slots_str = ",".join(func_slots[full_name])
        lines.append(
            f'    "{full_name}|{dtypes_str}|{src_hash}'
            f'|slots:{slots_str}"')
    lines.append(")")
    lines.append("")

    manifest_path = os.path.join(output_dir, "generated_shaders.cmake")
    content = "\n".join(lines)
    write_if_changed(manifest_path, content)
    print(f"Generated manifest: generated_shaders.cmake")


# =============================================================================
# Native CUDA manifest generation
# =============================================================================

def generate_native_manifest(impl_dir, native_kernels, manifest_path):
    """Write the native CUDA manifest JSON (kernels + shared .cuh headers)."""
    headers = []
    for entry in sorted(os.listdir(impl_dir)):
        group_dir = os.path.join(impl_dir, entry)
        if not os.path.isdir(group_dir):
            continue
        for fn in sorted(os.listdir(group_dir)):
            if fn.endswith(".cuh"):
                headers.append(
                    {"name": fn,
                     "path": os.path.abspath(os.path.join(group_dir, fn))})
    manifest = {"headers": headers,
                "kernels": {k: native_kernels[k]
                            for k in sorted(native_kernels)}}
    content = json.dumps(manifest, indent=2) + "\n"
    write_if_changed(manifest_path, content)
    print(f"Native CUDA manifest: {len(native_kernels)} kernel variant(s), "
          f"{len(headers)} header(s)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate shader variants and forward-declaration headers"
    )
    parser.add_argument(
        "--impl-dir", required=True,
        help="Root directory containing shader group subdirectories"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for dtype-preprocessed shader files"
    )
    parser.add_argument(
        "--cuda-native-manifest", default=None,
        help="Optional path to write the native CUDA kernel manifest JSON"
    )
    args = parser.parse_args()

    impl_dir = args.impl_dir
    output_dir = args.output_dir

    if not os.path.isdir(impl_dir):
        print(f"ERROR: --impl-dir does not exist: {impl_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect all generated shader files across groups
    all_generated = []

    # Native CUDA manifest entries (only collected when requested)
    native_kernels = {} if args.cuda_native_manifest else None

    # Scan subdirectories for shaders.json
    processed = 0
    for entry in sorted(os.listdir(impl_dir)):
        group_dir = os.path.join(impl_dir, entry)
        config_path = os.path.join(group_dir, "shaders.json")

        if not os.path.isdir(group_dir) or not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            raw = json.load(f)

        # Support multi-group format: {"groups": [{...}, {...}]}
        if "groups" in raw:
            configs = raw["groups"]
        else:
            configs = [raw]

        for config in configs:
            cpp_prefix = config.get("cpp_prefix", "")
            group_name = cpp_prefix.lower() if cpp_prefix else entry

            print(f"Processing group: {group_name}")

            # Generate dtype-preprocessed shader files
            generated = generate_shader_files(config, group_dir, output_dir,
                                                     impl_dir, native_kernels)
            all_generated.extend(generated)

            # Generate appropriate header
            if has_dispatch_metadata(config):
                generate_variant_header(config, group_dir, group_name)
            else:
                generate_simple_header(config, group_dir, group_name)

            processed += 1

    # Write CMake manifest
    generate_cmake_manifest(all_generated, output_dir)

    # Write native CUDA manifest if requested
    if args.cuda_native_manifest:
        generate_native_manifest(impl_dir, native_kernels,
                                 args.cuda_native_manifest)

    print(f"Done. Processed {processed} shader group(s), "
          f"generated {len(all_generated)} shader file(s).")


if __name__ == "__main__":
    main()
