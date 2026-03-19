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


def run_cut(model_path: str):
    """Run CUT gguf_example and parse timing."""
    exe = "build/interface/loader/gguf/gguf_example"
    if not os.path.exists(exe):
        print("  Binary not found, skipping")
        return None

    result = subprocess.run([exe, model_path], capture_output=True, text=True, timeout=120)
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


def run_llamacpp_native(model_path: str, max_tokens: int = 32):
    """Run native llama.cpp bench (CPU, optimized C++)."""
    exe = "/tmp/llama_cpp_vulkan/build_cpu/bin/llama-bench"
    if not os.path.exists(exe):
        print("  llama-bench not found, skipping")
        return None

    result = subprocess.run(
        [exe, "-m", model_path, "-p", "15", "-n", str(max_tokens), "-r", "1"],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout + result.stderr

    info = {"text": "(benchmark mode — no text output)"}

    for line in output.split("\n"):
        # | llama 256M F16 | ... | pp15 | 1214.05 ± 0.00 |
        # | llama 256M F16 | ... | tg32 |   96.02 ± 0.00 |
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

    # 1. CUT (Vulkan GPU)
    print_header("1. CUT (Vulkan GPU)")
    info = run_cut(args.model)
    if info:
        print_result(info, has_split=True)
        results["CUT (Vulkan GPU)"] = info
    else:
        print("  FAILED")
    print()

    # 2. llama.cpp native CLI (CPU, optimized C++)
    print_header("2. llama.cpp native CLI (CPU, -march=native)")
    info = run_llamacpp_native(args.model, args.max_tokens)
    if info:
        print_result(info, has_split=True)
        results["llama.cpp CLI (CPU)"] = info
    else:
        print("  FAILED or not built")
    print()

    # 3. llama-cpp-python (CPU)
    print_header("3. llama-cpp-python binding (CPU)")
    info = run_llamacpp_python(args.model, args.max_tokens)
    if info:
        print_result(info)
        results["llama-cpp-python (CPU)"] = info
    else:
        print("  FAILED")
    print()

    # 4. HuggingFace transformers (PyTorch CPU)
    print_header("4. HuggingFace transformers (PyTorch CPU)")
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
        # Calculate decode tok/s for each
        rows = []
        for name, info in results.items():
            decode_tps = info.get("decode_tps", 0)
            if decode_tps == 0 and "total_ms" in info:
                decode_tps = info.get("overall_tps", 0)
            total_ms = info.get("total_ms",
                                info.get("prefill_ms", 0) + info.get("decode_ms", 0))
            rows.append((name, total_ms, decode_tps))

        # Sort by total time
        rows.sort(key=lambda r: r[1])
        baseline_ms = rows[-1][1]  # slowest

        print(f"  {'Runner':<35} {'Total ms':>10} {'Decode tok/s':>14} {'vs slowest':>12}")
        print(f"  {'-'*35} {'-'*10} {'-'*14} {'-'*12}")
        for name, total_ms, decode_tps in rows:
            speedup = baseline_ms / total_ms if total_ms > 0 else 0
            print(f"  {name:<35} {total_ms:>10.1f} {decode_tps:>14.1f} {speedup:>11.1f}x")


if __name__ == "__main__":
    main()
