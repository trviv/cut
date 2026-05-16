#!/usr/bin/env python3
"""Compare CUT GGUF inference performance against other runners.

Usage:
    python scripts/benchmark_compare.py models/SmolLM2-135M-Instruct-f16.gguf
"""

import argparse
import subprocess
import time
import os

PROMPT = "Hello, how are you?"
CHAT_PROMPT = f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"


def run_cut(model_path: str, max_tokens: int = 32, device_index: int = None):
    """Run CUT gguf_example and parse timing."""
    exe = "build/interface/runner/llama/gguf_example"
    if not os.path.exists(exe):
        print("  Binary not found, skipping")
        return None

    env = os.environ.copy()
    if device_index is not None:
        env["CUT_VULKAN_DEVICE"] = str(device_index)

    result = subprocess.run([exe, model_path, str(max_tokens)],
                            capture_output=True, text=True,
                            timeout=120, env=env)
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
            # Remove any trailing WARNING lines
            clean_lines = []
            for l in text.split("\n"):
                if l.startswith("WARNING:"):
                    break
                clean_lines.append(l)
            info["text"] = "\n".join(clean_lines).strip()
    return info if info else None


def _parse_llama_bench(output: str):
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


def run_llamacpp_bench(model_path: str, max_tokens: int = 32,
                       vulkan: bool = False, vk_device: int = None):
    """Run native llama.cpp bench (CPU or Vulkan GPU)."""
    if vulkan:
        exe = "build/external_runners/llama.cpp/build_vk/bin/llama-bench"
    else:
        exe = "build/external_runners/llama.cpp/build_cpu/bin/llama-bench"
    if not os.path.exists(exe):
        kind = "Vulkan" if vulkan else "CPU"
        print(f"  llama-bench ({kind}) not found at {exe}")
        print(f"  Run: ./scripts/setup_benchmark_runners.sh")
        return None

    cmd = [exe, "-m", model_path, "-p", "15", "-n", str(max_tokens), "-r", "1",
           "--no-warmup"]
    if vulkan:
        cmd += ["-ngl", "99"]

    env = os.environ.copy()
    if vk_device is not None:
        env["GGML_VK_DEVICE"] = str(vk_device)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    return _parse_llama_bench(result.stdout + result.stderr)


def run_llamacpp_python(model_path: str, max_tokens: int = 32):
    """Run llama-cpp-python (CPU)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  llama-cpp-python not installed, skipping")
        return None

    print("  Loading model...")
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=0, verbose=False)

    print("  Generating...")
    t0 = time.perf_counter()
    output = llm(CHAT_PROMPT, max_tokens=max_tokens, temperature=0.0,
                 repeat_penalty=1.05, stop=["<|im_end|>"])
    gen_time = (time.perf_counter() - t0) * 1000

    text = output["choices"][0]["text"]
    usage = output["usage"]

    return {
        "total_ms": gen_time,
        "prompt_tokens": usage["prompt_tokens"],
        "decode_tokens": usage["completion_tokens"],
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "overall_tps": (usage["prompt_tokens"] + usage["completion_tokens"]) / (gen_time / 1000),
        "text": text.strip(),
    }


def run_transformers(model_name: str, max_tokens: int = 32):
    """Run HuggingFace transformers (PyTorch CPU)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers/torch not installed, skipping")
        return None

    print("  Loading model from HuggingFace...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, trust_remote_code=True
    ).to("cpu")
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000

    messages = [{"role": "user", "content": PROMPT}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    prompt_tokens = input_ids.shape[1]

    print("  Generating...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    gen_ms = (time.perf_counter() - t0) * 1000

    new_tokens = output_ids.shape[1] - prompt_tokens
    text = tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

    return {
        "load_ms": load_ms,
        "total_ms": gen_ms,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": new_tokens,
        "total_tokens": prompt_tokens + new_tokens,
        "overall_tps": (prompt_tokens + new_tokens) / (gen_ms / 1000),
        "text": text.strip(),
    }


def detect_vulkan_devices():
    """Detect Vulkan devices from llama-bench stderr."""
    exe = "build/external_runners/llama.cpp/build_vk/bin/llama-bench"
    if not os.path.exists(exe):
        return []
    result = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=10)
    # Run a quick dummy to get device listing from ggml_vulkan init
    result = subprocess.run(
        [exe, "-m", "/dev/null"], capture_output=True, text=True, timeout=10
    )
    output = result.stdout + result.stderr
    devices = []
    for line in output.split("\n"):
        # ggml_vulkan: 0 = AMD Ryzen ... | uma: 1 ...
        # ggml_vulkan: 1 = NVIDIA GeForce RTX 3090 ...
        if "ggml_vulkan:" in line and "=" in line and "|" in line:
            try:
                idx_part = line.split("ggml_vulkan:")[1].strip()
                idx = int(idx_part.split("=")[0].strip())
                name = idx_part.split("=")[1].split("(")[0].strip()
                is_igpu = "uma: 1" in line
                devices.append({"idx": idx, "name": name, "igpu": is_igpu})
            except (ValueError, IndexError):
                pass
    return devices


def detect_cut_vulkan_devices():
    """Detect available Vulkan devices by probing CUT with each device index.

    CUT prints 'Vulkan device: <name>' to stderr/stdout. We probe indices
    0..7 and parse the device name. Stops when it sees a duplicate (meaning
    the index wrapped around to fallback).
    """
    exe = "build/interface/runner/llama/gguf_example"
    if not os.path.exists(exe):
        return []

    seen_names = set()
    devices = []
    for idx in range(8):
        env = os.environ.copy()
        env["CUT_VULKAN_DEVICE"] = str(idx)
        try:
            # Run with a nonexistent model — Vulkan init happens before model
            # load, so we get the device name even though the run will fail.
            result = subprocess.run(
                [exe, "/dev/null"], capture_output=True, text=True,
                timeout=10, env=env
            )
            output = result.stdout + result.stderr
            found = False
            for line in output.split("\n"):
                if "Vulkan device:" in line:
                    name = line.split("Vulkan device:")[1].strip()
                    if "(" in name:
                        name = name[:name.index("(")].strip()
                    # Stop if we see a device we've already found — means the
                    # index exceeded the device count and fell back.
                    if name in seen_names:
                        return devices
                    seen_names.add(name)
                    # Detect integrated GPUs by checking for CPU brand names
                    # in the Vulkan device name (RADV exposes iGPUs this way)
                    igpu_keywords = ["Ryzen", "Core", "Celeron", "Pentium",
                                     "Atom", "RAPHAEL", "REMBRANDT",
                                     "PHOENIX", "llvmpipe", "SwiftShader"]
                    is_igpu = any(kw.lower() in name.lower()
                                  for kw in igpu_keywords)
                    devices.append({"idx": idx, "name": name, "igpu": is_igpu})
                    found = True
                    break
            if not found:
                break
        except (subprocess.TimeoutExpired, Exception):
            break
    return devices


def print_header(name):
    print("=" * 60)
    print(name)
    print("=" * 60)


def print_result(info, has_split=False):
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUT vs other inference runners")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--hf-model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="HuggingFace model ID for transformers benchmark")
    args = parser.parse_args()

    print(f"Model:      {args.model}")
    print(f"Prompt:     \"{PROMPT}\"")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = {}
    bench_idx = 1

    # 1+. CUT (Vulkan GPU) — run on each detected discrete device
    cut_devices = detect_cut_vulkan_devices()
    cut_discrete = [d for d in cut_devices if not d.get("igpu", False)]
    if cut_discrete:
        for dev in cut_discrete:
            label = f"CUT Vulkan ({dev['name']})"
            print_header(f"{bench_idx}. {label}")
            info = run_cut(args.model, args.max_tokens, device_index=dev["idx"])
            if info:
                print_result(info, has_split=True)
                results[label] = info
            else:
                print("  FAILED")
            print()
            bench_idx += 1
    else:
        # Fallback: run CUT without device selection
        print_header(f"{bench_idx}. CUT (Vulkan GPU)")
        info = run_cut(args.model, args.max_tokens)
        if info:
            print_result(info, has_split=True)
            results["CUT (Vulkan GPU)"] = info
        else:
            print("  FAILED")
        print()
        bench_idx += 1

    # llama.cpp Vulkan GPU (each discrete device)
    vk_devices = detect_vulkan_devices()
    for dev in vk_devices:
        if dev["igpu"]:
            continue  # skip integrated GPUs for the main comparison
        label = f"llama.cpp Vulkan ({dev['name']})"
        print_header(f"{bench_idx}. {label}")
        info = run_llamacpp_bench(args.model, args.max_tokens,
                                  vulkan=True, vk_device=dev["idx"])
        if info:
            print_result(info, has_split=True)
            results[label] = info
        else:
            print("  FAILED or not built")
        print()
        bench_idx += 1

    # If no discrete GPUs found, try default Vulkan device anyway
    if not any(not d["igpu"] for d in vk_devices):
        print_header(f"{bench_idx}. llama.cpp Vulkan GPU (default device)")
        info = run_llamacpp_bench(args.model, args.max_tokens, vulkan=True)
        if info:
            print_result(info, has_split=True)
            results["llama.cpp Vulkan GPU"] = info
        else:
            print("  FAILED or not built — run: ./scripts/setup_benchmark_runners.sh")
        print()
        bench_idx += 1

    # 3. llama.cpp CPU
    print_header(f"{bench_idx}. llama.cpp CPU (-march=native)")
    info = run_llamacpp_bench(args.model, args.max_tokens, vulkan=False)
    if info:
        print_result(info, has_split=True)
        results["llama.cpp CPU"] = info
    else:
        print("  FAILED or not built")
    print()
    bench_idx += 1

    # 4. llama-cpp-python (CPU)
    print_header(f"{bench_idx}. llama-cpp-python binding (CPU)")
    info = run_llamacpp_python(args.model, args.max_tokens)
    if info:
        print_result(info)
        results["llama-cpp-python (CPU)"] = info
    else:
        print("  FAILED")
    print()
    bench_idx += 1

    # 5. HuggingFace transformers (PyTorch CPU)
    print_header(f"{bench_idx}. HuggingFace transformers (PyTorch CPU)")
    info = run_transformers(args.hf_model, args.max_tokens)
    if info:
        print_result(info)
        results["transformers (PyTorch CPU)"] = info
    else:
        print("  FAILED")
    print()

    # --- Summary Table ---
    if len(results) > 1:
        print_header("SUMMARY")
        rows = []
        for name, info in results.items():
            prefill_ms = info.get("prefill_ms", 0)
            prefill_tps = info.get("prefill_tps", 0)
            decode_ms = info.get("decode_ms", 0)
            decode_tps = info.get("decode_tps", 0)
            decode_tokens = info.get("decode_tokens", 0)
            if decode_tps == 0 and "total_ms" in info:
                decode_tps = info.get("overall_tps", 0)
            total_ms = info.get("total_ms",
                                prefill_ms + decode_ms)
            rows.append((name, prefill_ms, prefill_tps, decode_tokens, decode_ms, decode_tps, total_ms))

        # Sort by total time
        rows.sort(key=lambda r: r[6])
        baseline_ms = rows[-1][6]  # slowest

        nw = max(len(r[0]) for r in rows)  # runner name column width
        nw = max(nw, len("Runner"))
        print(f"  {'Runner':<{nw}} {'Prefill ms':>10} {'Prefill tok/s':>13} {'Decode toks':>11} {'Decode ms':>10} {'Decode tok/s':>13} {'Total ms':>10} {'vs slowest':>12}")
        print(f"  {'-'*nw} {'-'*10} {'-'*13} {'-'*11} {'-'*10} {'-'*13} {'-'*10} {'-'*12}")
        for name, prefill_ms, prefill_tps, decode_tokens, decode_ms, decode_tps, total_ms in rows:
            speedup = baseline_ms / total_ms if total_ms > 0 else 0
            pfill_ms_str = f"{prefill_ms:.1f}" if prefill_ms > 0 else "-"
            pfill_tps_str = f"{prefill_tps:.1f}" if prefill_tps > 0 else "-"
            dec_toks_str = str(decode_tokens) if decode_tokens > 0 else "-"
            dec_ms_str = f"{decode_ms:.1f}" if decode_ms > 0 else "-"
            print(f"  {name:<{nw}} {pfill_ms_str:>10} {pfill_tps_str:>13} {dec_toks_str:>11} {dec_ms_str:>10} {decode_tps:>13.1f} {total_ms:>10.1f} {speedup:>11.1f}x")


if __name__ == "__main__":
    main()
