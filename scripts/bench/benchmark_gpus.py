#!/usr/bin/env python3
"""Benchmark CUT vs llama.cpp on every real GPU in the system.

Usage:
    python scripts/bench/benchmark_gpus.py models/SmolLM2-135M-Instruct-f16.gguf
    python scripts/bench/benchmark_gpus.py --include-igpu --max-tokens 64 models/...gguf

Note: On a CUDA/3090 box, free VRAM first (e.g. `ollama stop devstral-small-2:24b`).
"""

import argparse
import subprocess
import time
import os
import json
import shutil
import glob
from typing import List, Dict, Optional, Any
import statistics

PROMPT = "Hello, how are you?"
CHAT_PROMPT = f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"

def normalize_name(name: str) -> str:
    """Normalize GPU name for cross-framework matching."""
    name = name.strip().lower()
    # Collapse internal whitespace runs to single spaces
    name = ' '.join(name.split())
    # Take substring before first '('
    if '(' in name:
        name = name[:name.index('(')].strip()
    return name

def classify_vendor(name: str) -> str:
    """Classify GPU vendor from name."""
    name = name.lower()
    if any(kw in name for kw in ["nvidia", "geforce", "rtx", "gtx", "quadro", "tesla"]):
        return "nvidia"
    if any(kw in name for kw in ["amd", "radeon", "radv", "ryzen"]):
        return "amd"
    return "other"

def is_software(name: str) -> bool:
    """Check if device is a software rasterizer."""
    name = name.lower()
    return any(kw in name for kw in ["llvmpipe", "lavapipe", "swiftshader"])

def detect_cpu_vulkan_device_names() -> set:
    """Names of Vulkan devices whose VkPhysicalDeviceType is CPU (software
    implementations like llvmpipe), from `vulkaninfo --summary`. Returns
    normalized names; empty set if vulkaninfo is unavailable."""
    try:
        result = subprocess.run(["vulkaninfo", "--summary"],
                                capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return set()
    cpu_names = set()
    last_type = None
    # In `vulkaninfo --summary` each device's deviceType line precedes its
    # deviceName line, so remember the last seen type.
    for line in (result.stdout + result.stderr).split("\n"):
        line = line.strip()
        if line.startswith("deviceType"):
            last_type = line.split("=", 1)[1].strip() if "=" in line else None
        elif line.startswith("deviceName") and "=" in line:
            name = line.split("=", 1)[1].strip()
            if last_type == "PHYSICAL_DEVICE_TYPE_CPU":
                # Store both the full and the clean (pre-parenthesis) form so
                # the filter matches however the name was captured.
                cpu_names.add(normalize_name(name))
                cpu_names.add(normalize_name(name.split("(")[0].strip()))
            last_type = None
    return cpu_names

def detect_cut_vulkan_devices(cut_binary: str) -> List[Dict[str, Any]]:
    """Detect Vulkan devices available to CUT by probing indices 0..7."""
    if not os.path.exists(cut_binary):
        return []

    seen_names = set()
    devices = []
    for idx in range(8):
        env = os.environ.copy()
        env["CUT_DEVICES"] = f"vulkan:{idx}"
        try:
            result = subprocess.run(
                [cut_binary, "/dev/null"],
                capture_output=True,
                text=True,
                timeout=15,
                env=env
            )
            output = result.stdout + result.stderr
            found = False
            for line in output.split("\n"):
                if "Vulkan device:" in line:
                    name = line.split("Vulkan device:")[1].strip()
                    # Stop if we see a device we've already found
                    norm_name = normalize_name(name)
                    if norm_name in seen_names:
                        return devices
                    seen_names.add(norm_name)
                    devices.append({"idx": idx, "name": name})
                    found = True
                    break
            if not found:
                break
        except (subprocess.TimeoutExpired, Exception):
            break
    return devices

def detect_llama_vulkan_devices(llama_vk_bench: str) -> List[Dict[str, Any]]:
    """Detect Vulkan devices available to llama.cpp bench."""
    if not os.path.exists(llama_vk_bench):
        return []

    try:
        result = subprocess.run(
            [llama_vk_bench, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=15
        )
        output = result.stdout + result.stderr
        devices = []

        # First pass: collect shape-A entries (ggml_vulkan: N = ...)
        shape_a_devices = []
        for line in output.split("\n"):
            if "ggml_vulkan:" in line and "=" in line and "|" in line:
                # Shape: "ggml_vulkan: 1 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | ..."
                try:
                    idx_part = line.split("ggml_vulkan:")[1].strip()
                    idx = int(idx_part.split("=")[0].strip())
                    name = idx_part.split("=")[1].split("|")[0].strip()
                    uma_flag = "uma: 1" in line
                    shape_a_devices.append({"idx": idx, "name": name, "igpu": uma_flag})
                except (ValueError, IndexError):
                    continue

        if shape_a_devices:
            return shape_a_devices

        # Fallback: collect shape-B entries (VulkanN: ...)
        for line in output.split("\n"):
            if "Vulkan" in line and ":" in line and "(" in line:
                # Shape: "  Vulkan1: NVIDIA GeForce RTX 3090 (24576 MiB, ...)"
                try:
                    idx = int(line.split("Vulkan")[1].split(":")[0].strip())
                    name = line.split(":")[1].split("(")[0].strip()
                    devices.append({"idx": idx, "name": name, "igpu": None})
                except (ValueError, IndexError):
                    continue
        return devices
    except (subprocess.TimeoutExpired, Exception):
        return []

def detect_cuda_devices() -> List[Dict[str, Any]]:
    """Detect CUDA devices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10
        )
        devices = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line.startswith("GPU ") or ":" not in line:
                continue
            rest = line[4:]  # Drop "GPU "
            try:
                idx = int(rest.split(":", 1)[0].strip())
                name = rest.split(":", 1)[1].strip()
                name = name.split("(")[0].strip()
                devices.append({"idx": idx, "name": name})
            except (ValueError, IndexError):
                continue
        return devices
    except (subprocess.TimeoutExpired, Exception, FileNotFoundError):
        return []

def enumerate_gpus(cut_binary: str, llama_vk_bench: str) -> List[Dict[str, Any]]:
    """Merge GPU lists from all sources and return sorted list."""
    # Get devices from each source
    cut_vk_devices = detect_cut_vulkan_devices(cut_binary)
    llama_vk_devices = detect_llama_vulkan_devices(llama_vk_bench)
    cuda_devices = detect_cuda_devices()

    # Build map by normalized name
    gpu_map = {}

    for dev in cut_vk_devices:
        norm_name = normalize_name(dev["name"])
        clean_name = dev["name"].split("(")[0].strip()
        if norm_name in gpu_map:
            gpu_map[norm_name]["cut_vk"] = dev["idx"]
        else:
            gpu_map[norm_name] = {
                "name": clean_name,
                "vendor": classify_vendor(dev["name"]),
                "igpu": False,
                "cut_vk": dev["idx"],
                "llama_vk": None,
                "cuda": None
            }

    for dev in llama_vk_devices:
        norm_name = normalize_name(dev["name"])
        clean_name = dev["name"].split("(")[0].strip()
        if norm_name in gpu_map:
            gpu_map[norm_name]["llama_vk"] = dev["idx"]
            if dev.get("igpu") is True:
                gpu_map[norm_name]["igpu"] = True
        else:
            gpu_map[norm_name] = {
                "name": clean_name,
                "vendor": classify_vendor(dev["name"]),
                "igpu": dev.get("igpu", False),
                "cut_vk": None,
                "llama_vk": dev["idx"],
                "cuda": None
            }

    for dev in cuda_devices:
        norm_name = normalize_name(dev["name"])
        clean_name = dev["name"].split("(")[0].strip()
        if norm_name in gpu_map:
            gpu_map[norm_name]["cuda"] = dev["idx"]
        else:
            gpu_map[norm_name] = {
                "name": clean_name,
                "vendor": classify_vendor(dev["name"]),
                "igpu": False,
                "cut_vk": None,
                "llama_vk": None,
                "cuda": dev["idx"]
            }

    # Convert to list and apply igpu heuristic fallback
    gpus = list(gpu_map.values())
    for gpu in gpus:
        if gpu["igpu"] is False and gpu["vendor"] == "amd":
            name = gpu["name"].lower()
            if any(kw in name for kw in ["ryzen", "raphael", "rembrandt", "phoenix", "processor"]):
                gpu["igpu"] = True

    # Keep only real GPUs (discrete or integrated): drop software
    # rasterizers by name AND any device whose Vulkan deviceType is CPU
    # (catches CPU-emulated implementations regardless of their name).
    cpu_vk_names = detect_cpu_vulkan_device_names()
    gpus = [gpu for gpu in gpus
            if not is_software(gpu["name"])
            and normalize_name(gpu["name"]) not in cpu_vk_names]

    # Sort: discrete first, then nvidia -> amd -> other, then by name
    def sort_key(gpu):
        discrete = 1 if gpu["igpu"] else 0
        vendor_order = {"nvidia": 0, "amd": 1, "other": 2}
        return (discrete, vendor_order.get(gpu["vendor"], 2), gpu["name"])

    return sorted(gpus, key=sort_key)

def run_cut(cut_binary: str, model: str, backend: str, dev_idx: int, max_tokens: int, runs: int = 5, warmup: int = 1, synthetic: bool = True) -> Optional[Dict[str, Any]]:
    """Run CUT with specified backend and device index."""
    if not os.path.exists(cut_binary):
        return None

    env = os.environ.copy()
    env["CUT_DEVICES"] = f"{backend}:{dev_idx}"
    if synthetic:
        # Match llama-bench's tg test: decode measures pure forward-pass
        # throughput (skip per-token penalty upload, argmax readback, printing).
        env["CUT_BENCH_SYNTHETIC"] = "1"

    cmd = [cut_binary, model, str(max_tokens)]
    if runs > 0:
        cmd += ["--bench-runs", str(runs), "--bench-warmup", str(warmup), "--no-stop"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        output = result.stdout + result.stderr
        info = {}

        for line in output.split("\n"):
            if "Prefill:" in line:
                parts = line.split(",")
                info["prefill_ms"] = float(parts[0].split(":")[1].strip().split()[0])
                info["prefill_tokens"] = int(parts[1].strip().split()[0])
                info["prefill_tps"] = float(parts[2].strip().split()[0])
            elif "Decode:" in line:
                parts = line.split(",")
                info["decode_ms"] = float(parts[0].split(":")[1].strip().split()[0])
                info["decode_tokens"] = int(parts[1].strip().split()[0])
                info["decode_tps"] = float(parts[2].strip().split()[0])
            elif "Total:" in line and "ms" in line:
                parts = line.split(",")
                info["total_ms"] = float(parts[0].split(":")[1].strip().split()[0])
                info["total_tokens"] = int(parts[1].strip().split()[0])
            elif "Decoded text:" in line:
                idx = output.index("Decoded text:")
                text = output[idx + len("Decoded text:"):].strip()
                if "assistant\n" in text:
                    text = text.split("assistant\n", 1)[-1]
                clean_lines = []
                for l in text.split("\n"):
                    if l.startswith("WARNING:"):
                        break
                    clean_lines.append(l)
                info["text"] = "\n".join(clean_lines).strip()

        return info if info else None
    except (subprocess.TimeoutExpired, Exception):
        return None

def _parse_llama_bench(output: str) -> Optional[Dict[str, Any]]:
    """Parse llama-bench table output into an info dict."""
    info = {"text": "(benchmark mode — no text output)"}

    for line in output.split("\n"):
        if "|" in line and "pp" in line:
            parts = [p.strip() for p in line.split("|")]
            for p in parts:
                if p.startswith("pp"):
                    info["prefill_tokens"] = int(p[2:])
                if "±" in p:
                    info["prefill_tps"] = float(p.split("±")[0].strip())
            if "prefill_tps" in info and "prefill_tokens" in info:
                info["prefill_ms"] = info["prefill_tokens"] / info["prefill_tps"] * 1000
        elif "|" in line and "tg" in line:
            parts = [p.strip() for p in line.split("|")]
            for p in parts:
                if p.startswith("tg"):
                    info["decode_tokens"] = int(p[2:])
                if "±" in p:
                    info["decode_tps"] = float(p.split("±")[0].strip())
            if "decode_tps" in info and "decode_tokens" in info:
                info["decode_ms"] = info["decode_tokens"] / info["decode_tps"] * 1000

    if "prefill_ms" in info and "decode_ms" in info:
        info["total_ms"] = info["prefill_ms"] + info["decode_ms"]
        info["total_tokens"] = info.get("prefill_tokens", 0) + info.get("decode_tokens", 0)
    return info if "decode_ms" in info else None

def _parse_llama_bench_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse llama-bench JSON output into an info dict."""
    info = {"text": "(benchmark mode — no text output)"}
    try:
        entries = json.loads(stdout)
        if not isinstance(entries, list):
            return None

        prefill_entry = None
        decode_entry = None

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "n_prompt" not in entry or "n_gen" not in entry or "samples_ts" not in entry:
                continue
            samples = entry["samples_ts"]
            if not isinstance(samples, list) or len(samples) == 0:
                continue

            if entry["n_gen"] == 0 and entry["n_prompt"] > 0:
                prefill_entry = entry
            elif entry["n_gen"] > 0 and entry["n_prompt"] == 0:
                decode_entry = entry

        if prefill_entry is not None:
            prefill_tps = statistics.median(prefill_entry["samples_ts"])
            if prefill_tps > 0:
                info["prefill_tokens"] = prefill_entry["n_prompt"]
                info["prefill_tps"] = prefill_tps
                info["prefill_ms"] = info["prefill_tokens"] / prefill_tps * 1000

        if decode_entry is not None:
            decode_tps = statistics.median(decode_entry["samples_ts"])
            if decode_tps > 0:
                info["decode_tokens"] = decode_entry["n_gen"]
                info["decode_tps"] = decode_tps
                info["decode_ms"] = info["decode_tokens"] / decode_tps * 1000

        if "prefill_ms" in info and "decode_ms" in info:
            info["total_ms"] = info["prefill_ms"] + info["decode_ms"]
            info["total_tokens"] = info.get("prefill_tokens", 0) + info.get("decode_tokens", 0)

        return info if "decode_tps" in info else None
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        return None

def run_llama(llama_bench_exe: str, model: str, backend: str, dev_idx: int, max_tokens: int, runs: int = 5) -> Optional[Dict[str, Any]]:
    """Run llama.cpp bench with specified backend and device index."""
    if not os.path.exists(llama_bench_exe):
        return None

    cmd = [llama_bench_exe, "-m", model, "-p", "15", "-n", str(max_tokens), "-r", str(runs), "-o", "json"]

    env = os.environ.copy()
    if backend == "vulkan":
        env["GGML_VK_DEVICE"] = str(dev_idx)
        cmd.extend(["-ngl", "99"])
    elif backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(dev_idx)
        cmd.extend(["-ngl", "99"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        info = _parse_llama_bench_json(result.stdout)
        if info is None:
            info = _parse_llama_bench(result.stdout + result.stderr)
        return info
    except (subprocess.TimeoutExpired, Exception):
        return None

def print_result(info: Dict[str, Any], has_split: bool = False):
    """Print benchmark result in consistent format."""
    if has_split and "prefill_ms" in info:
        print(f"  Prefill:  {info['prefill_ms']:.1f} ms "
              f"({info.get('prefill_tokens', '?')} tokens, "
              f"{info.get('prefill_tps', 0):.1f} tok/s)")
        print(f"  Decode:   {info['decode_ms']:.1f} ms "
              f"({info.get('decode_tokens', '?')} tokens, "
              f"{info.get('decode_tps', 0):.1f} tok/s)")
        total = info.get("total_ms", info.get("prefill_ms", 0) + info.get("decode_ms", 0))
        total_tok = info.get("total_tokens",
                             info.get("prefill_tokens", 0) + info.get("decode_tokens", 0))
        print(f"  Total:    {total:.1f} ms ({total_tok} tokens)")
    else:
        print(f"  Total:    {info['total_ms']:.1f} ms "
              f"({info.get('total_tokens', '?')} tokens, "
              f"{info.get('overall_tps', 0):.1f} tok/s)")
    if "load_ms" in info:
        print(f"  Load:     {info['load_ms']:.0f} ms")
    if "text" in info:
        text = info["text"][:120]
        print(f"  Output:   \"{text}\"")

def print_summary_table(results: Dict[str, Dict[str, Dict[str, Any]]], gpu_name: str):
    """Print summary table for a single GPU."""
    if not results.get(gpu_name):
        return

    rows = []
    for label, info in results[gpu_name].items():
        prefill_ms = info.get("prefill_ms", 0)
        prefill_tps = info.get("prefill_tps", 0)
        decode_ms = info.get("decode_ms", 0)
        decode_tps = info.get("decode_tps", 0)
        decode_tokens = info.get("decode_tokens", 0)
        if decode_tps == 0 and "total_ms" in info:
            decode_tps = info.get("overall_tps", 0)
        total_ms = info.get("total_ms", prefill_ms + decode_ms)
        rows.append((label, prefill_ms, prefill_tps, decode_tokens, decode_ms, decode_tps, total_ms))

    if len(rows) < 2:
        return

    rows.sort(key=lambda r: r[6])
    baseline_ms = rows[-1][6]

    nw = max(len(r[0]) for r in rows)
    nw = max(nw, len("Runner"))
    print(f"  {'Runner':<{nw}} {'Prefill tok/s':>13} {'Decode tok/s':>13} {'Total ms':>10} {'vs slowest':>12}")
    print(f"  {'-'*nw} {'-'*13} {'-'*13} {'-'*10} {'-'*12}")
    for name, prefill_ms, prefill_tps, decode_tokens, decode_ms, decode_tps, total_ms in rows:
        speedup = baseline_ms / total_ms if total_ms > 0 else 0
        pfill_tps_str = f"{prefill_tps:.1f}" if prefill_tps > 0 else "-"
        dec_tps_str = f"{decode_tps:.1f}" if decode_tps > 0 else "-"
        print(f"  {name:<{nw}} {pfill_tps_str:>13} {dec_tps_str:>13} {total_ms:>10.1f} {speedup:>11.1f}x")

def short_gpu_name(name: str) -> str:
    """Return a shortened display name for table columns."""
    name = name.strip()
    prefixes = [
        "NVIDIA GeForce ",
        "AMD Radeon ",
        "AMD ",
        "Intel Arc ",
        "Intel "
    ]
    lower = name.lower()
    for prefix in prefixes:
        if lower.startswith(prefix.lower()):
            name = name[len(prefix):]
            break
    name = name.strip()
    if len(name) > 22:
        name = name[:21] + "…"
    return name

def print_overall_table(results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> None:
    """Print consolidated CUT vs llama.cpp performance table."""
    rows = []
    for model_name, model_results in results.items():
        for gpu_name, gpu_results in model_results.items():
            for backend in ["CUDA", "VULKAN"]:
                cut = gpu_results.get(f"CUT {backend}")
                llama = gpu_results.get(f"llama.cpp {backend}")
                if cut is None and llama is None:
                    continue
                cut_pp = cut["prefill_tps"] if cut and "prefill_tps" in cut else None
                llama_pp = llama["prefill_tps"] if llama and "prefill_tps" in llama else None
                pp_speedup = cut_pp / llama_pp if cut_pp is not None and llama_pp is not None and llama_pp > 0 else None
                cut_ppn = cut["prefill_tokens"] if cut and "prefill_tokens" in cut else None
                llama_ppn = llama["prefill_tokens"] if llama and "prefill_tokens" in llama else None
                cut_tg = cut["decode_tps"] if cut and "decode_tps" in cut else None
                llama_tg = llama["decode_tps"] if llama and "decode_tps" in llama else None
                tg_speedup = cut_tg / llama_tg if cut_tg is not None and llama_tg is not None and llama_tg > 0 else None
                rows.append((
                    model_name.replace(".gguf", ""),
                    short_gpu_name(gpu_name),
                    backend,
                    cut_pp,
                    llama_pp,
                    pp_speedup,
                    cut_tg,
                    llama_tg,
                    tg_speedup,
                    cut_ppn,
                    llama_ppn,
                ))
    if not rows:
        print("  (no paired results to tabulate)")
        return
    print("=" * 112)
    print("OVERALL: CUT vs llama.cpp  (tok/s, higher is better; tg = decode is the primary metric)")
    print("=" * 112)
    model_width = max(len(r[0]) for r in rows) if rows else 0
    gpu_width = max(len(r[1]) for r in rows) if rows else 0
    model_width = max(model_width, len("Model"))
    gpu_width = max(gpu_width, len("GPU"))
    print(f"  {'Model':<{model_width}} | {'GPU':<{gpu_width}} | {'Backend':<8} | {'CUT pp':>10} | {'llama pp':>10} | {'pp x':>8} | {'CUT tg':>10} | {'llama tg':>10} | {'tg x':>8}")
    print(f"  {('-'*model_width):<{model_width}} | {('-'*gpu_width):<{gpu_width}} | {'-'*8:<8} | {'-'*10:>10} | {'-'*10:>10} | {'-'*8:>8} | {'-'*10:>10} | {'-'*10:>10} | {'-'*8:>8}")
    for model, gpu, backend, cut_pp, llama_pp, pp_speedup, cut_tg, llama_tg, tg_speedup, cut_ppn, llama_ppn in rows:
        cut_pp_str = f"{cut_pp:.1f}" if cut_pp is not None else "-"
        llama_pp_str = f"{llama_pp:.1f}" if llama_pp is not None else "-"
        pp_speedup_str = f"{pp_speedup:.2f}x" if pp_speedup is not None else "-"
        cut_tg_str = f"{cut_tg:.1f}" if cut_tg is not None else "-"
        llama_tg_str = f"{llama_tg:.1f}" if llama_tg is not None else "-"
        tg_speedup_str = f"{tg_speedup:.2f}x" if tg_speedup is not None else "-"
        print(f"  {model:<{model_width}} | {gpu:<{gpu_width}} | {backend:<8} | {cut_pp_str:>10} | {llama_pp_str:>10} | {pp_speedup_str:>8} | {cut_tg_str:>10} | {llama_tg_str:>10} | {tg_speedup_str:>8}")

    if any(r[-2] is not None and r[-1] is not None and r[-2] != r[-1] for r in rows):
        print()
        print("  NOTE: pp x compares different prompt lengths - CUT prefills the tokenized")
        print("        chat prompt while llama-bench uses a fixed -p 15, so treat pp x as")
        print("        indicative only. tg x is the apples-to-apples comparison.")

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUT vs llama.cpp on GPUs")
    parser.add_argument("models", nargs="*", default=[],
                        help="GGUF model paths. If none given, auto-discovers models/*.gguf (skipping ones "
                             "larger than --max-model-gb).")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5,
                        help="Timed runs per measurement; the MEDIAN is reported. Default 5")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs before timing (CUT in-process; llama-bench warms up "
                             "internally). Default 1")
    parser.add_argument("--cut-binary", default=None,
                        help="Path to CUT gguf_example binary")
    parser.add_argument("--llama-dir", default="build/external_runners/llama.cpp",
                        help="Path to llama.cpp build directory")
    parser.add_argument("--include-igpu", action="store_true",
                        help="Also benchmark integrated GPUs (excluded by default; "
                             "only discrete GPUs are benchmarked unless this is set)")
    parser.add_argument("--cut-real-gen", action="store_true",
                        help="Measure CUT with real generation (sampling + argmax readback) "
                             "instead of the synthetic bench mode that matches llama-bench. "
                             "Note: on Vulkan this lowers CUT decode ~30%% due to per-token "
                             "sampling overhead llama-bench does not incur.")
    parser.add_argument("--cpu", action="store_true",
                        help="Also run llama.cpp CPU baseline")
    parser.add_argument("--json", type=str,
                        help="Dump raw results as JSON to this path")
    parser.add_argument("--max-model-gb", type=float, default=4.0,
                        help="When auto-discovering models, skip GGUF files larger than this many GB "
                             "(pass paths explicitly to force-include). Default 4.0")
    args = parser.parse_args()

    # Resolve CUT binary
    cut_binary = args.cut_binary
    if cut_binary is None:
        candidates = [
            "build-cuda-rel/interface/runner/llama/gguf_example",
            "build-release/interface/runner/llama/gguf_example",
            "build-cuda/interface/runner/llama/gguf_example",
            "build/interface/runner/llama/gguf_example"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                cut_binary = cand
                break
        if cut_binary is None:
            print("ERROR: CUT binary not found. Please build or specify with --cut-binary.")
            exit(1)

    # Resolve llama.cpp binaries
    llama_dir = args.llama_dir
    llama_vk_bench = os.path.join(llama_dir, "build_vk/bin/llama-bench")
    llama_cuda_bench = os.path.join(llama_dir, "build_cuda/bin/llama-bench")
    llama_cpu_bench = os.path.join(llama_dir, "build_cpu/bin/llama-bench")

    print("=" * 60)
    print("GPU Benchmark: CUT vs llama.cpp")
    print("=" * 60)

    # Model resolution
    import glob
    if args.models:
        model_paths = list(args.models)
    else:
        model_paths = []
        for m in sorted(glob.glob("models/*.gguf")):
            size_gb = os.path.getsize(m) / 1e9
            if size_gb > args.max_model_gb:
                print(f"  Skipping {m} ({size_gb:.1f} GB > {args.max_model_gb} GB; pass it explicitly to include)")
            else:
                model_paths.append(m)
    model_paths = [m for m in model_paths if os.path.exists(m)]
    if not model_paths:
        print("ERROR: no GGUF models found to benchmark. Pass model paths or put .gguf files in models/.")
        exit(1)

    print(f"Models:     {', '.join(os.path.basename(m) for m in model_paths)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"CUT binary: {cut_binary}")
    print(f"Methodology: {args.warmup} warmup + {args.runs} timed runs per measurement, median reported. "
          f"CUT uses in-process reps (gguf_example --bench-runs); llama-bench uses -r {args.runs} with "
          f"its own warmup and we take the median of per-rep samples. both decode the full {args.max_tokens} tokens (CUT runs with --no-stop; llama-bench -n), so tg is a fair comparison. Decode/tg tok/s is the primary metric. "
          f"By default CUT runs in synthetic bench mode (CUT_BENCH_SYNTHETIC) so its decode measures "
          f"pure forward-pass throughput like llama-bench (no sampling/readback); pass --cut-real-gen to "
          f"measure real end-to-end generation instead.")
    print()

    # Enumerate GPUs
    gpus = enumerate_gpus(cut_binary, llama_vk_bench)
    if not args.include_igpu:
        skipped = [gpu["name"] for gpu in gpus if gpu["igpu"]]
        gpus = [gpu for gpu in gpus if not gpu["igpu"]]
        if skipped:
            print(f"Skipping integrated GPUs (pass --include-igpu to benchmark them): "
                  f"{', '.join(skipped)}")

    if not gpus:
        print("No GPUs to benchmark (software rasterizers and, by default, "
              "integrated GPUs are excluded; see --include-igpu).")
        exit(0)

    results = {}  # results[model_name][gpu_name][label] = info

    # Run benchmarks for each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print("=" * 60)
        print(f"# {model_name}")
        print("=" * 60)
        results[model_name] = {}

        # Run benchmarks for each GPU
        for gpu in gpus:
            print("=" * 60)
            print(f"GPU: {gpu['name']}")
            print(f"  Vendor: {gpu['vendor']}, {'Integrated' if gpu['igpu'] else 'Discrete'}")
            print()

            gpu_results = {}
            backends = ["cuda", "vulkan"] if gpu["vendor"] == "nvidia" and gpu["cuda"] is not None else ["vulkan"]

            for backend in backends:
                # CUT run
                cut_dev_idx = gpu["cuda"] if backend == "cuda" else gpu["cut_vk"]
                if cut_dev_idx is not None:
                    label = f"CUT {backend.upper()}"
                    print(f"{label}:")
                    info = run_cut(cut_binary, model_path, backend, cut_dev_idx, args.max_tokens, args.runs, args.warmup, synthetic=not args.cut_real_gen)
                    if info:
                        print_result(info, has_split=True)
                        gpu_results[label] = info
                    else:
                        print("  FAILED")
                    print()

                # llama.cpp run
                if backend == "cuda":
                    exe = llama_cuda_bench
                    dev_idx = gpu["cuda"]
                    label = f"llama.cpp {backend.upper()}"
                    print(f"{label}:")
                    if not os.path.exists(exe):
                        print("  llama.cpp CUDA bench not built (needs CUDA toolkit / nvcc). Skipping.")
                        print()
                        continue
                else:
                    exe = llama_vk_bench
                    dev_idx = gpu["llama_vk"]
                    label = f"llama.cpp {backend.upper()}"
                    print(f"{label}:")
                    if not os.path.exists(exe):
                        print("  FAILED or not built")
                        print()
                        continue

                if dev_idx is not None:
                    info = run_llama(exe, model_path, backend, dev_idx, args.max_tokens, args.runs)
                    if info:
                        print_result(info, has_split=True)
                        gpu_results[label] = info
                    else:
                        print("  FAILED")
                print()

            # Print per-GPU summary
            if len(gpu_results) >= 2:
                print("  Per-GPU Summary:")
                print_summary_table({gpu["name"]: gpu_results}, gpu["name"])
                print()

            results[model_name][gpu["name"]] = gpu_results

        # CPU baseline
        if args.cpu and os.path.exists(llama_cpu_bench):
            print("=" * 60)
            print("CPU Baseline (llama.cpp)")
            print()
            info = run_llama(llama_cpu_bench, model_path, "cpu", 0, args.max_tokens, args.runs)
            if info:
                print_result(info)
                results[model_name]["CPU"] = {"llama.cpp CPU": info}
            else:
                print("  FAILED")
            print()

    # Overall summary
    print_overall_table(results)

    # NVIDIA CUDA note
    if any(gpu["vendor"] == "nvidia" for gpu in gpus) and not os.path.exists(llama_cuda_bench):
        print("=" * 60)
        print("NOTE: NVIDIA GPUs detected but llama.cpp CUDA bench not built.")
        print("  To enable CUDA benchmarks, run:")
        print("    ./scripts/setup/setup_benchmark_runners.sh")
        print("  This requires the CUDA toolkit (nvcc).")
        print("  Note: CUT's CUDA backend uses NVRTC and does not need nvcc.")
        print("=" * 60)
        print()

    # Write JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.json}")

if __name__ == "__main__":
    main()
