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
{
  "type": "variants",
  "cpp_prefix": "MatMul",           // prepended to variant name for full name
  "default_variant": "T16R4x4",     // optional, suffix form
  "variants": [
    {
      "name": "Naive",              // suffix after cpp_prefix (can be "")
      "description": "...",
      "dtypes": ["Float32", ...],
      "template": "MatMulNaive",    // optional: generate from template
      "defines": {},                // optional: template substitutions
      "wg": [16, 16],              // optional: workgroup size
      "eff_tile": [16, 16]         // optional: effective tile size
    }
  ]
}
"""

import argparse
import hashlib
import json
import os
import re
import sys


# =============================================================================
# Datatype definitions (moved from shader_loader.cmake)
# =============================================================================

DTYPE_DEFS = {
    "Float32": {
        "vec":     "float4",
        "scalar":  "float",
        "size":    "4",
        "defines": "#define DTYPE_IS_FLOAT 1",
    },
    "Float16": {
        "vec":     "half4",
        "scalar":  "half",
        "size":    "4",
        "defines": "#define DTYPE_IS_FLOAT 1\n#define DTYPE_IS_HALF 1",
    },
    "Int32": {
        "vec":     "int4",
        "scalar":  "int",
        "size":    "4",
        "defines": "#define DTYPE_IS_INT 1",
    },
    "UInt32": {
        "vec":     "uint4",
        "scalar":  "uint",
        "size":    "4",
        "defines": "#define DTYPE_IS_UINT 1",
    },
}


def substitute_dtype(content, dtype_name):
    """Replace dtype placeholders with concrete types for a given dtype."""
    d = DTYPE_DEFS[dtype_name]
    content = content.replace("%VEC_DTYPE%", d["vec"])
    content = content.replace("%SCALAR_DTYPE%", d["scalar"])
    content = content.replace("%DTYPE_SIZE%", d["size"])
    content = content.replace("%DTYPE_DEFINES%", d["defines"])
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


# =============================================================================
# Shader file generation (template expansion + dtype preprocessing)
# =============================================================================

def generate_shader_files(config, group_dir, output_dir, impl_dir):
    """Generate dtype-preprocessed .shader files for all variants.

    Returns a list of (full_name, dtype_name, output_path) tuples.
    """
    cpp_prefix = config.get("cpp_prefix", "")
    generated = []

    for variant in config["variants"]:
        name = variant["name"]
        fname = full_name_for(cpp_prefix, name)

        # Get base shader content
        if "template" in variant:
            # Load parameterised shader and substitute define placeholders
            template_path = os.path.join(group_dir,
                                         variant["template"] + ".shader")
            if not os.path.exists(template_path):
                print(f"ERROR: Shader template not found: {template_path}",
                      file=sys.stderr)
                sys.exit(1)
            with open(template_path, "r") as f:
                base_content = f.read()
            for key, value in variant.get("defines", {}).items():
                base_content = base_content.replace(f"%{key}%", str(value))
        else:
            # Read existing .shader source file
            shader_path = os.path.join(group_dir, fname + ".shader")
            if not os.path.exists(shader_path):
                print(f"ERROR: Shader source not found: {shader_path}",
                      file=sys.stderr)
                sys.exit(1)
            with open(shader_path, "r") as f:
                base_content = f.read()

        # Resolve .shaderh includes before dtype substitution
        base_content = resolve_shaderh_includes(base_content, group_dir,
                                                impl_dir)

        # Compute source hash (after include resolution, before dtype substitution)
        source_hash = hashlib.md5(base_content.encode()).hexdigest()

        # Generate dtype-specific shader files
        for dtype in variant["dtypes"]:
            if dtype not in DTYPE_DEFS:
                print(f"ERROR: Unknown dtype '{dtype}' in variant '{fname}'",
                      file=sys.stderr)
                sys.exit(1)

            preprocessed = substitute_dtype(base_content, dtype)
            out_path = os.path.join(output_dir, f"{fname}_{dtype}.shader")

            if write_if_changed(out_path, preprocessed):
                print(f"  Generated {fname}_{dtype}.shader")

            generated.append((fname, dtype, out_path, source_hash))

    return generated


# =============================================================================
# Header generation: dispatch table (matmul-style with wg/eff_tile)
# =============================================================================

def has_dispatch_metadata(config):
    """Check if any variant has wg/eff_tile (matmul-style dispatch table)."""
    return any("wg" in v and "eff_tile" in v for v in config["variants"])


def generate_variant_header(config, group_dir, group_name):
    """Generate {Prefix}Variants.generated.h with dispatch table and functions."""
    variants = config["variants"]
    count = len(variants)
    cpp_prefix = config.get("cpp_prefix", "")
    cap = cpp_prefix if cpp_prefix else (group_name[0].upper() + group_name[1:])

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
            f"compiled{fname}(DataType datatype);"
        )
    lines.append("")

    # Function pointer type and lookup table
    lines.append(
        f"using Compiled{cap}Fn = "
        "std::optional<std::vector<uint32_t>> (*)(DataType);"
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
    lines.append(f"/// Returns compiled SPIR-V for a {group_name} variant by index.")
    lines.append("inline std::optional<std::vector<uint32_t>>")
    lines.append(
        f"getCompiled{cap}(int variantIndex, "
        "DataType datatype = DataType::Float32) {"
    )
    lines.append(
        f"    if (variantIndex < 0 || variantIndex >= k{cap}VariantCount)"
    )
    lines.append("        return std::nullopt;")
    lines.append(f"    return k{cap}CompiledFns[variantIndex](datatype);")
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
        lines.append(
            f"std::optional<std::vector<uint32_t>> "
            f"compiled{fname}(DataType datatype = DataType::Float32);"
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
    # Collect per-function dtype lists and source hashes (preserving order)
    func_dtypes = {}
    func_hashes = {}
    func_order = []
    all_files = []

    for full_name, dtype, out_path, source_hash in all_generated:
        all_files.append(out_path)
        if full_name not in func_dtypes:
            func_dtypes[full_name] = []
            func_hashes[full_name] = source_hash
            func_order.append(full_name)
        func_dtypes[full_name].append(dtype)

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
    # Format: "FunctionName|Dtype1,Dtype2,...|source_hash"
    lines.append("set(SHADER_FUNCTION_DTYPES")
    for full_name in func_order:
        dtypes_str = ",".join(func_dtypes[full_name])
        src_hash = func_hashes[full_name]
        lines.append(f'    "{full_name}|{dtypes_str}|{src_hash}"')
    lines.append(")")
    lines.append("")

    manifest_path = os.path.join(output_dir, "generated_shaders.cmake")
    content = "\n".join(lines)
    write_if_changed(manifest_path, content)
    print(f"Generated manifest: generated_shaders.cmake")


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
    args = parser.parse_args()

    impl_dir = args.impl_dir
    output_dir = args.output_dir

    if not os.path.isdir(impl_dir):
        print(f"ERROR: --impl-dir does not exist: {impl_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Collect all generated shader files across groups
    all_generated = []

    # Scan subdirectories for shaders.json
    processed = 0
    for entry in sorted(os.listdir(impl_dir)):
        group_dir = os.path.join(impl_dir, entry)
        config_path = os.path.join(group_dir, "shaders.json")

        if not os.path.isdir(group_dir) or not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        group_name = entry

        print(f"Processing group: {group_name}")

        # Generate dtype-preprocessed shader files
        generated = generate_shader_files(config, group_dir, output_dir,
                                                 impl_dir)
        all_generated.extend(generated)

        # Generate appropriate header
        if has_dispatch_metadata(config):
            generate_variant_header(config, group_dir, group_name)
        else:
            generate_simple_header(config, group_dir, group_name)

        processed += 1

    # Write CMake manifest
    generate_cmake_manifest(all_generated, output_dir)

    print(f"Done. Processed {processed} shader group(s), "
          f"generated {len(all_generated)} shader file(s).")


if __name__ == "__main__":
    main()
