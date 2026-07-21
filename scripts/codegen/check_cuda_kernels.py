#!/usr/bin/env python3
"""Offline NVRTC compile gate for CUT CUDA kernels.

Compiles every supported transpiled kernel (from a transpiler manifest dir)
and every native kernel (from a native manifest) with the same header set and
options the runtime uses (core/backends/cuda/CudaContainers.cpp), without
touching GPU memory. Kills the "NVRTC failure => silent dispatch skip => zero
outputs" failure mode at build time instead of runtime.

Run with the project venv python (requires cuda.bindings.nvrtc):

    .venv/bin/python scripts/codegen/check_cuda_kernels.py \
        --cu-dir build-cuda-rel/generated_cuda \
        --native-manifest build-cuda-rel/generated_cuda/native_manifest.json \
        --prelude operators/runtime/cuda/cut_cuda_prelude.cuh \
        --enums operators/runtime/ComputeOpsShared.h \
        --impl-dir operators/impl

No CUT_SPEC defines are passed: kernels carry #ifndef defaults.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

from cuda.bindings import nvrtc


def compile_one(job, header_names, header_sources, base_opts):
    """Compile one kernel; returns (stem, ok, log)."""
    stem = job["stem"]
    try:
        with open(job["cu"], "r", encoding="utf-8") as f:
            cu_src = f.read()
    except OSError as e:
        return (stem, False, f"Failed to read {job['cu']}: {e}")

    # Mirror the runtime: prepend the prelude include to the kernel source.
    src = '#include "cut_cuda_prelude.cuh"\n' + cu_src

    opts = list(base_opts) + [f"-D{d}" for d in job.get("defines", [])]
    opts_bytes = [o.encode("utf-8") for o in opts]

    err, prog = nvrtc.nvrtcCreateProgram(
        src.encode("utf-8"), b"cut_kernel.cu", len(header_names),
        [s.encode("utf-8") for s in header_sources],
        [n.encode("utf-8") for n in header_names])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return (stem, False, f"nvrtcCreateProgram failed: {err}")

    err, = nvrtc.nvrtcCompileProgram(prog, len(opts_bytes), opts_bytes)
    ok = err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    log = ""
    if not ok:
        lerr, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if lerr == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 1:
            log_buf = bytearray(log_size)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            log = log_buf.decode("utf-8", errors="replace").strip("\x00 \n")
    nvrtc.nvrtcDestroyProgram(prog)
    return (stem, ok, log)


def main():
    parser = argparse.ArgumentParser(
        description="NVRTC compile-check every CUT CUDA kernel (no GPU memory)")
    parser.add_argument("--cu-dir", required=True,
                        help="Transpiled kernel dir containing cuda_kernels.json")
    parser.add_argument("--native-manifest", default=None,
                        help="Optional native CUDA kernel manifest JSON")
    parser.add_argument("--prelude", required=True,
                        help="cut_cuda_prelude.cuh path")
    parser.add_argument("--enums", required=True,
                        help="ComputeOpsShared.h path")
    parser.add_argument("--impl-dir", default=None,
                        help="operators/impl root; *.cuh in subdirs become headers")
    parser.add_argument("--arch", default="compute_86",
                        help="NVRTC --gpu-architecture value")
    parser.add_argument("--include-dir", default=None,
                        help="CUDA include dir for cuda_fp16.h etc. "
                             "(mirrors the runtime's CUT_CUDA_INCLUDE_DIR; "
                             "auto-detected when omitted)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Parallel compile jobs")
    parser.add_argument("--only", default=None,
                        help="Substring filter on kernel stem names")
    args = parser.parse_args()

    transpiled_path = Path(args.cu_dir) / "cuda_kernels.json"
    transpiled = {}
    if transpiled_path.exists():
        with open(transpiled_path, "r", encoding="utf-8") as f:
            transpiled = json.load(f)

    native = {"headers": [], "kernels": {}}
    if args.native_manifest and Path(args.native_manifest).exists():
        with open(args.native_manifest, "r", encoding="utf-8") as f:
            native = json.load(f)

    jobs = []
    for stem in sorted(transpiled):
        info = transpiled[stem]
        if not info.get("supported", False):
            continue
        if args.only and args.only not in stem:
            continue
        jobs.append({"stem": stem,
                     "cu": str(Path(args.cu_dir) / info["cu"]),
                     "defines": []})
    for stem in sorted(native.get("kernels", {})):
        info = native["kernels"][stem]
        if args.only and args.only not in stem:
            continue
        jobs.append({"stem": stem + " (native)",
                     "cu": info["cu"],
                     "defines": info.get("defines", [])})

    if not jobs:
        print("No kernels to compile (empty job list)")
        sys.exit(0 if args.only else 1)

    # Header set, mirroring the runtime: prelude, enums, then every shared
    # .cuh (impl-dir scan + native manifest, deduped by include name).
    header_names = ["cut_cuda_prelude.cuh", "ComputeOpsShared.h"]
    header_sources = []
    with open(args.prelude, "r", encoding="utf-8") as f:
        header_sources.append(f.read())
    with open(args.enums, "r", encoding="utf-8") as f:
        header_sources.append(f.read())

    extra_headers = []  # (name, path)
    if args.impl_dir:
        for cuh in sorted(Path(args.impl_dir).rglob("*.cuh")):
            extra_headers.append((cuh.name, str(cuh)))
    for hdr in native.get("headers", []):
        extra_headers.append((hdr["name"], hdr["path"]))
    for name, path in extra_headers:
        if name in header_names:
            continue
        with open(path, "r", encoding="utf-8") as f:
            header_sources.append(f.read())
        header_names.append(name)

    include_dir = args.include_dir
    if include_dir is None:
        # Auto-detect: venv-bundled CUDA headers, then a system install.
        candidates = [
            Path(nvrtc.__file__).parents[2] / "nvidia" / "cu13" / "include",
            Path(nvrtc.__file__).parents[2] / "nvidia" / "cu12" / "include",
            Path("/usr/local/cuda/include"),
        ]
        for cand in candidates:
            if (cand / "cuda_fp16.h").exists():
                include_dir = str(cand)
                break
    if include_dir is None:
        print("ERROR: could not find cuda_fp16.h; pass --include-dir",
              file=sys.stderr)
        sys.exit(1)

    base_opts = [f"--gpu-architecture={args.arch}", "--std=c++14",
                 f"-I{include_dir}"]

    n_ok = 0
    n_fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(compile_one, job, header_names, header_sources,
                      base_opts): job["stem"]
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            stem = futures[future]
            try:
                stem, ok, log = future.result()
            except Exception as e:  # worker crash
                print(f"FAIL {stem}")
                print(f"  Worker error: {e}")
                n_fail += 1
                continue
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"FAIL {stem}")
                for line in log.splitlines():
                    print(f"  {line}")

    print(f"PASS: {n_ok}/{len(jobs)} kernels compiled ({n_fail} failed).")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
