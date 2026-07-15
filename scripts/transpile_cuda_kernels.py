#!/usr/bin/env python3
"""Transpile dtype-preprocessed HLSL compute shaders to CUDA C++ kernels.

Counterpart to the HLSL->SPIR-V path: reads the same dtype-substituted
``{Func}_{dtypes}.shader`` files produced by generate_shader_variants.py and
emits ``{Func}_{dtypes}.cu`` CUDA kernels plus a JSON manifest of metadata
(entry name, specialization constants, buffer params).

The transpile is deliberately textual and targets the regular element-wise
kernel pattern (unary / binary / ternary / cast / creation): module-scope
specialization constants, a push-constant struct, StructuredBuffer bindings,
and a ``[numthreads] main(... SV_DispatchThreadID)`` entry. Kernels that use
shared memory, group intrinsics, or other unsupported constructs are skipped
(reported in the manifest as ``supported: false``) so the CUDA backend simply
falls back to no-kernel for those until they are ported.

The HLSL bodies compile against operators/runtime/cuda/cut_cuda_prelude.cuh,
which provides vector-type operators and HLSL intrinsic shims. Specialization
constants become NVRTC ``-D CUT_SPEC_<id>=<value>`` macros so each (op, dtype)
specialization compiles to its own module at runtime.
"""

import argparse
import json
import os
import re
import sys

ENTRY_NAME = "cut_main"

VEC_CAST_SUFFIX = {
    "float4": "f4",
    "int4": "i4",
    "uint4": "u4",
    "half4": "h4",
}

# Constructs we do not (yet) transpile. Presence of any => skip the kernel.
# Shared memory, barriers, group/thread IDs, and atomics ARE handled below; the
# remaining blockers are byte-address/typed buffers and constant buffers (rare),
# plus the GLSL cooperative-matrix matmul variants (a separate .comp path).
UNSUPPORTED_MARKERS = [
    "RWByteAddressBuffer",
    "ByteAddressBuffer",
    "cbuffer",
    "RWTexture",
    "Texture2D",
]

# CUDA built-ins keyed by the HLSL system-value semantic.
_SV_SETUP = {
    "SV_DispatchThreadID": (
        "{n}.x = blockIdx.x * blockDim.x + threadIdx.x; "
        "{n}.y = blockIdx.y * blockDim.y + threadIdx.y; "
        "{n}.z = blockIdx.z * blockDim.z + threadIdx.z;"),
    "SV_GroupThreadID": (
        "{n}.x = threadIdx.x; {n}.y = threadIdx.y; {n}.z = threadIdx.z;"),
    "SV_GroupID": (
        "{n}.x = blockIdx.x; {n}.y = blockIdx.y; {n}.z = blockIdx.z;"),
}

_SPEC_RE = re.compile(
    r"\[\[vk::constant_id\((\d+)\)\]\]\s*const\s+(\w+)\s+(\w+)\s*=\s*([^;]+);")
_PUSH_RE = re.compile(r"\[\[vk::push_constant\]\]\s*(\w+)\s+(\w+)\s*;")
_BUF_RE = re.compile(
    r"\[\[vk::binding\((\d+)\s*,\s*\d+\)\]\]\s*(RW)?StructuredBuffer<\s*([^>]+?)\s*>\s*(\w+)\s*;")
# numthreads dims may be macros (e.g. WG_SIZE); we don't need their values —
# the launch block size comes from runtime SPIR-V reflection, not the transpile.
_MAIN_RE = re.compile(
    r"\[numthreads\([^)]*\)\]\s*void\s+main\s*\(([^)]*)\)\s*\{")
_SVPARAM_RE = re.compile(r"uint3\s+(\w+)\s*:\s*(SV_\w+)")
# groupshared declaration (module scope) — moved into the kernel for CUDA.
_GROUPSHARED_RE = re.compile(r"(?m)^[ \t]*groupshared\s+([^;]+;)")
# A top-level helper function definition we must tag __device__. The return
# type is any single word (float4, uint2, void, ...); only column-0 lines match,
# so indented `return foo(...)` statements are not mistaken for definitions.
_FUNC_RE = re.compile(r"^(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{")


def _convert_vector_casts(text):
    """Replace HLSL vector casts with cut_cast_<suffix>(...).

    Handles both the parenthesized broadcast form "(float4)(expr)" and the
    bare-postfix form "(float4)ident[idx]" used for cross-dtype input loads.
    """
    # Parenthesized: "(T)(" -> "cut_cast_T("  (the existing close paren balances).
    text = re.sub(r"\((float4|int4|uint4|half4)\)\s*\(",
                  lambda m: f"cut_cast_{VEC_CAST_SUFFIX[m.group(1)]}(", text)
    # Bare postfix: "(T)name", "(T)name[idx]", or "(T)0" (numeric literal).
    text = re.sub(
        r"\((float4|int4|uint4|half4)\)\s*([A-Za-z_0-9][\w.]*(?:\[[^\]]*\])?)",
        lambda m: f"cut_cast_{VEC_CAST_SUFFIX[m.group(1)]}({m.group(2)})", text)
    return text


_DEVFUNC_RE = re.compile(r"__device__\s+[\w:<>]+\s+(\w+)\s*\([^)]*\)\s*\{")


def _match_brace(text, open_idx):
    """Return index just past the '}' matching the '{' at open_idx."""
    depth = 0
    i = open_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _device_funcs(text):
    """List (start, body_open, end, name) for each top-level __device__ helper."""
    funcs = []
    for m in _DEVFUNC_RE.finditer(text):
        end = _match_brace(text, m.end() - 1)
        funcs.append((m.start(), m.end() - 1, end, m.group(1)))
    return funcs


def _helpers_use_resources(text, resource_names):
    """True if any __device__ helper body references a module-scope resource
    (buffer / push constant / shared array), which CUDA scoping forbids unless
    we move them into a struct as members."""
    if not resource_names:
        return False
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in resource_names) +
                     r")\b")
    for start, body_open, end, _name in _device_funcs(text):
        if pat.search(text[body_open:end]):
            return True
    return False


def _struct_wrap(text, buffers, push, shared_vars, shared_decls, sig):
    """Wrap helpers + main into a struct so helpers can reach module-scope
    resources (buffers / push constant / shared memory) as members.

    Layout produced:
        <preamble: spec consts, PushConstants, #defines>   (stays global)
        struct CutKernel {
          <buffer/pc/shared-pointer members>
          <helpers (already __device__)>
          __device__ void cut_run() { <SV setup> <main body> }
        };
        extern "C" __global__ void cut_main(<sig>) {
          <__shared__ decls>
          CutKernel k; k.<member> = <param/shared>; ...
          k.cut_run();
        }
    """
    funcs = _device_funcs(text)
    mm = _MAIN_RE.search(text)
    if not funcs or mm is None:
        return None
    body_start = min(f[0] for f in funcs)        # first helper
    if body_start > mm.start():
        return None                              # helpers must precede main
    # The struct wrapper must open at preprocessor nesting level 0. Helpers are
    # often guarded by a module-scope "#if ..." whose "#endif" follows them; if
    # the struct opened between the "#if" and the helper, a false condition
    # would compile out the opening brace but not the trailing "};". Start the
    # wrapper right after the PushConstants struct (always level 0, before any
    # such guard) so the guard and its #endif are balanced inside the wrapper.
    if push:
        sm = re.search(r"\bstruct\s+" + re.escape(push[0]) + r"\s*\{", text)
        if sm:
            pc_end = _match_brace(text, text.index("{", sm.end() - 1))
            semi = text.find(";", pc_end)
            pc_body_start = (semi + 1 if semi >= 0 else pc_end)
            if pc_body_start <= mm.start():
                body_start = min(body_start, pc_body_start)
    main_end = _match_brace(text, text.index("{", mm.end() - 1))

    preamble = text[:body_start]
    helpers = text[body_start:mm.start()]

    # main body (between its braces), minus the moved __shared__ decls.
    main_open = text.index("{", mm.end() - 1)
    main_body = text[main_open + 1:main_end - 1]

    # SV index setup for cut_run.
    sv_lines = []
    for name, sem in _SVPARAM_RE.findall(mm.group(1)):
        sv_lines.append(f"        uint3 {name};")
        sv_lines.append("        " + _SV_SETUP[sem].format(n=name))

    # Struct members + initialization in the wrapper.
    members, inits = [], []
    for _idx, is_rw, elem, name in buffers:
        qual = "" if is_rw else "const "
        members.append(f"  {qual}{elem}* {name};")
        inits.append(f"  k.{name} = {name};")
    if push:
        members.append(f"  {push[0]} {push[1]};")
        inits.append(f"  k.{push[1]} = {push[1]};")
    for ty, name, inner in shared_vars:
        if inner:  # multi-dim: __shared__ T x[A][B] decays to T (*)[B]
            members.append(f"  {ty} (*{name}){inner};")
        else:
            members.append(f"  {ty}* {name};")
        inits.append(f"  k.{name} = {name};")

    out = []
    out.append(preamble.rstrip())
    out.append("\nstruct CutKernel {")
    out.extend(members)
    out.append(helpers.strip("\n"))
    out.append("  __device__ void cut_run() {")
    out.extend(sv_lines)
    out.append(main_body)
    out.append("  }")
    out.append("};")
    out.append(f'extern "C" __global__ void {ENTRY_NAME}({sig}) {{')
    out.extend(shared_decls)
    out.append("  CutKernel k;")
    out.extend(inits)
    out.append("  k.cut_run();")
    out.append("}")
    return "\n".join(out)


def _append_scalar_binaryop_wrapper(text):
    """HLSL lets a float4 binaryOp(float4,float4) be called with scalars
    (implicit splat) and the result assigned to a scalar (implicit .x
    truncation); C++ does neither, so scalar kernel variants that share the
    vec4 opFunc fail NVRTC. Emit an explicit scalar wrapper next to the vec4
    definition."""
    for vec, scal, cast in (("float4", "float", "cut_cast_f4"),):
        m = re.search(r"__device__\s+%s\s+binaryOp\s*\(\s*%s\s+\w+\s*,\s*%s\s+\w+\s*\)\s*\{"
                      % (vec, vec, vec), text)
        if m is None:
            continue
        end = _match_brace(text, text.index("{", m.start()))
        wrapper = ("\n__device__ __forceinline__ %s binaryOp(%s a, %s b) {\n"
                   "    return binaryOp(%s(a), %s(b)).x;\n"
                   "}\n" % (scal, scal, scal, cast, cast))
        text = text[:end] + wrapper + text[end:]
    return text


def transpile(source, func_name):
    """Transpile one preprocessed HLSL shader to CUDA. Returns (cu, meta) or None."""
    specs = []          # list of (id, name)
    buffers = []        # list of (index, is_rw, elem_type, name)
    push = None         # (type, var)

    # Drop the shared-enum include from the body; we re-add it in the header.
    source = re.sub(r'^\s*#include\s+"ComputeOpsShared\.h"\s*$', "",
                    source, flags=re.MULTILINE)

    # Strip comments. They are non-functional, and HLSL keywords mentioned in
    # comments (StructuredBuffer, groupshared, out/inout, ...) would otherwise
    # trip the textual transforms and the unsupported-construct post-check.
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)

    # HLSL sized scalar types -> CUDA (before buffer/element-type extraction).
    source = re.sub(r"\bfloat16_t\b", "half", source)
    source = re.sub(r"\bint16_t\b", "short", source)
    source = re.sub(r"\buint16_t\b", "unsigned short", source)
    # HLSL 2-/3-component vector constructor syntax -> CUDA make_* (the builtin
    # vector types have no constructor). "uint2(a,b)" -> "make_uint2(a,b)";
    # declarations like "uint2 name" are untouched (a space follows the type).
    source = re.sub(r"\b(float2|float3|int2|int3|uint2|uint3)\(",
                    r"make_\1(", source)
    # 4-component vector constructor syntax -> cut_mk_* (the custom vec types
    # have no 4-arg constructor; cut_mk_* build them field-by-field).
    _vec4ctor = {"float4": "cut_mk_f4", "int4": "cut_mk_i4", "uint4": "cut_mk_u4"}
    source = re.sub(r"\b(float4|int4|uint4)\(",
                    lambda m: _vec4ctor[m.group(1)] + "(", source)
    # HLSL out/inout parameter qualifiers -> CUDA references. Restricted to
    # parameter position (after '(' or ',') so the words "out"/"inout" inside
    # comments or expressions are never rewritten.
    source = re.sub(r"([(,]\s*)(?:out|inout)\s+(\w+)\s+(\w+)",
                    r"\1\2& \3", source)

    # Collect specialization constants and rewrite them to macro-backed consts.
    def spec_repl(m):
        spec_id, ty, name, default = m.group(1), m.group(2), m.group(3), m.group(4)
        specs.append((int(spec_id), name))
        return (f"#ifndef CUT_SPEC_{spec_id}\n"
                f"#define CUT_SPEC_{spec_id} ({default.strip()})\n"
                f"#endif\n"
                f"static const {ty} {name} = CUT_SPEC_{spec_id};")
    text = _SPEC_RE.sub(spec_repl, source)

    # Push-constant declaration -> remember it; drop the line.
    pm = _PUSH_RE.search(text)
    if pm:
        push = (pm.group(1), pm.group(2))
        text = _PUSH_RE.sub("", text)

    # Storage-buffer bindings -> kernel pointer params; drop the lines.
    for bm in _BUF_RE.finditer(text):
        buffers.append((int(bm.group(1)), bool(bm.group(2)),
                        bm.group(3).strip(), bm.group(4)))
    text = _BUF_RE.sub("", text)
    # A shader may declare the same binding in multiple #ifdef dtype branches
    # (e.g. an int8 path vs a float path). Keep the last declaration per binding
    # index so the param list matches the active (non-special) branch.
    by_index = {}
    for b in buffers:
        by_index[b[0]] = b
    buffers = sorted(by_index.values(), key=lambda b: b[0])

    if not buffers or _MAIN_RE.search(text) is None:
        return None  # not a compute-entry pattern we handle

    # Validate every main() signature uses only semantics we can map; collect
    # nothing here (the rewrite happens below) but bail early on the unknown.
    for mm in _MAIN_RE.finditer(text):
        paramstr = mm.group(1)
        svs = _SVPARAM_RE.findall(paramstr)
        if paramstr.count(":") != len(svs):
            return None  # a non-uint3 / unrecognized parameter
        if any(sem not in _SV_SETUP for _n, sem in svs):
            return None  # unknown system-value semantic

    # Strip HLSL loop/flow attributes (CUDA parses "[unroll]" as a lambda).
    text = re.sub(r"\[(?:unroll|loop|branch|flatten|fastopt)(?:\([^)]*\))?\]",
                  "", text)

    # Workgroup barriers -> CUDA block sync / fence.
    for hlsl in ("GroupMemoryBarrierWithGroupSync", "AllMemoryBarrierWithGroupSync",
                 "DeviceMemoryBarrierWithGroupSync"):
        text = text.replace(hlsl + "()", "__syncthreads()")
    for hlsl in ("GroupMemoryBarrier", "AllMemoryBarrier", "DeviceMemoryBarrier"):
        text = text.replace(hlsl + "()", "__threadfence_block()")

    # Atomics: HLSL InterlockedAdd(dest, val[, orig]) -> CUDA atomicAdd.
    text = re.sub(r"InterlockedAdd\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
                  r"\3 = atomicAdd(&(\1), \2)", text)
    text = re.sub(r"InterlockedAdd\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
                  r"atomicAdd(&(\1), \2)", text)

    # Hoist module-scope groupshared decls; CUDA __shared__ must be in-kernel.
    shared_decls = []   # full "__shared__ T name[..];" strings
    shared_vars = []    # (type, name) for struct-member pointers

    def _shared_collect(m):
        decl = m.group(1)  # e.g. "float tileA[16][17]; // comment"
        shared_decls.append("    __shared__ " + decl)
        nm = re.match(r"\s*([\w]+)\s+(\w+)\s*((?:\[[^\]]*\])+)", decl)
        if nm:
            dims = re.findall(r"\[[^\]]*\]", nm.group(3))
            inner = "".join(dims[1:])  # drop outermost dim (array->pointer decay)
            shared_vars.append((nm.group(1), nm.group(2), inner))
        return ""
    text = _GROUPSHARED_RE.sub(_shared_collect, text)

    # Vector broadcast casts.
    text = _convert_vector_casts(text)

    # Tag top-level helper functions as __device__ (skip main; handled below).
    out_lines = []
    for line in text.split("\n"):
        fm = _FUNC_RE.match(line)
        if fm and fm.group(2) != "main":
            line = "__device__ " + line
        out_lines.append(line)
    text = "\n".join(out_lines)

    # Build the kernel signature from collected buffers + push constant.
    params = []
    for _idx, is_rw, elem, name in buffers:
        qual = "" if is_rw else "const "
        ptr = f"{qual}{elem}* {name}"
        params.append(ptr)
    if push:
        params.append(f"{push[0]} {push[1]}")
    sig = ", ".join(params)

    shared_block = ("\n".join(shared_decls) + "\n") if shared_decls else ""

    # Resource names that helpers might (illegally, in CUDA) reference at module
    # scope. If any helper does, wrap helpers + main in a struct so they become
    # members; otherwise use the simple flat rewrite.
    resource_names = ([n for (_i, _r, _e, n) in buffers]
                      + ([push[1]] if push else [])
                      + [n for (_t, n, _d) in shared_vars])
    # Macros that expand to a resource (e.g. "#define OP_VAR pc.op") count too,
    # since a helper using the macro effectively references the resource.
    for mac, body in re.findall(r"#define\s+(\w+)\s+([^\n]*)", text):
        if any(re.search(r"\b" + re.escape(r) + r"\b", body)
               for r in resource_names):
            resource_names.append(mac)

    if _helpers_use_resources(text, resource_names):
        text = _struct_wrap(text, buffers, push, shared_vars, shared_decls, sig)
        if text is None:
            return None
    else:
        def main_repl(m):
            lines = [f'extern "C" __global__ void {ENTRY_NAME}({sig}) {{']
            for name, sem in _SVPARAM_RE.findall(m.group(1)):
                lines.append(f"    uint3 {name};")
                lines.append("    " + _SV_SETUP[sem].format(n=name))
            return "\n".join(lines) + "\n" + shared_block
        text = _MAIN_RE.sub(main_repl, text)

    # Append scalar wrappers for vec4 binaryOp functions.
    text = _append_scalar_binaryop_wrapper(text)

    # Nothing HLSL-specific should remain after rewriting.
    for marker in ("[[vk::", "StructuredBuffer", "numthreads", "groupshared",
                   "SV_Dispatch", "SV_Group"):
        if marker in text:
            return None

    header = ("// Auto-generated by transpile_cuda_kernels.py — do not edit\n"
              "#include \"ComputeOpsShared.h\"\n\n")
    cu = header + text + "\n"

    meta = {
        "func": func_name,
        "entry": ENTRY_NAME,
        "specs": [{"id": sid, "name": nm} for sid, nm in sorted(specs)],
        "buffers": [{"index": i, "rw": rw, "elem": e, "name": n}
                    for (i, rw, e, n) in buffers],
        "push": ({"type": push[0], "name": push[1]} if push else None),
        "supported": True,
    }
    return cu, meta


def _is_unsupported(source):
    for marker in UNSUPPORTED_MARKERS:
        if marker == "numthreads":
            continue  # element-wise kernels legitimately use [numthreads]
        if marker in source:
            return marker
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Directory of dtype-preprocessed .shader files")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for .cu kernels + manifest")
    ap.add_argument("--only", default=None,
                    help="Optional substring filter on shader file name")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {}
    n_ok = n_skip = 0

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".shader"):
            continue
        if args.only and args.only not in fname:
            continue
        stem = fname[:-len(".shader")]
        with open(os.path.join(args.input_dir, fname)) as f:
            source = f.read()

        reason = _is_unsupported(source)
        result = None if reason else transpile(source, stem)

        if result is None:
            manifest[stem] = {"supported": False,
                              "reason": reason or "unsupported pattern"}
            n_skip += 1
            continue

        cu, meta = result
        out_path = os.path.join(args.output_dir, stem + ".cu")
        with open(out_path, "w") as f:
            f.write(cu)
        meta["cu"] = stem + ".cu"
        manifest[stem] = meta
        n_ok += 1

    with open(os.path.join(args.output_dir, "cuda_kernels.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Transpiled {n_ok} kernel(s), skipped {n_skip}.")


if __name__ == "__main__":
    main()
