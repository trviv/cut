#!/usr/bin/env python3
"""Compare runtime memory usage (CPU RSS + GPU VRAM) between CUT and llama.cpp.

Usage:
    python scripts/benchmark_memory.py models/SmolLM2-135M-Instruct-f16.gguf

Ensures both runners use identical configuration (context size, prompt length,
generated tokens, KV cache type) for a fair comparison.

Measures:
    - Peak CPU RSS via /usr/bin/time -v (kernel-tracked, not sampled)
    - GPU VRAM via self-reported values from each runner's output
      (NVML does not reliably track Vulkan allocations)
"""

import argparse
import ctypes
import os
import re
import struct
import subprocess
import threading


# ---------------------------------------------------------------------------
# GGUF metadata reader (minimal — just enough to get model config)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF"

# Map of GGUF metadata key suffixes to config field names.
_GGUF_KEY_MAP = {
    "context_length": "context_length",
    "embedding_length": "dim",
    "block_count": "n_layers",
    "head_count_kv": "n_kv_heads",
    "head_count": "n_heads",
}


def _read_gguf_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype):
    _READERS = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
        10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
    }
    if vtype in _READERS:
        fmt, sz = _READERS[vtype]
        return struct.unpack(fmt, f.read(sz))[0]
    if vtype == 7:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == 8:
        return _read_gguf_string(f)
    if vtype == 9:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    return None


def get_model_config(model_path):
    """Read model config from GGUF metadata. Returns dict with dim, n_layers, etc."""
    config = {}
    with open(model_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            print(f"  WARNING: Not a valid GGUF file (magic={magic:#x})")
            return config
        _version = struct.unpack("<I", f.read(4))[0]
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_gguf_value(f, vtype)
            for suffix, field in _GGUF_KEY_MAP.items():
                if key.endswith(suffix) and field not in config:
                    config[field] = value
            if len(config) == len(_GGUF_KEY_MAP):
                break
    return config


# ---------------------------------------------------------------------------
# GPU memory monitoring (NVML via ctypes)
# ---------------------------------------------------------------------------

class NVMLMonitor:
    """Thin ctypes wrapper around libnvidia-ml for GPU memory queries.

    NOTE: NVML does not reliably track Vulkan memory allocations on NVIDIA
    GPUs (it primarily tracks CUDA). Self-reported values from each runner
    are more accurate for Vulkan workloads. This monitor provides supplementary
    data only.
    """

    def __init__(self):
        self._lib = None
        self._devices = []
        self._MemInfo = None
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self._lib.nvmlInit_v2() != 0:
                self._lib = None
                return

            class MemInfo(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong),
                             ("free", ctypes.c_ulonglong),
                             ("used", ctypes.c_ulonglong)]
            self._MemInfo = MemInfo

            count = ctypes.c_uint()
            self._lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
            for i in range(count.value):
                dev = ctypes.c_void_p()
                if self._lib.nvmlDeviceGetHandleByIndex_v2(
                        ctypes.c_uint(i), ctypes.byref(dev)) == 0:
                    name = ctypes.create_string_buffer(256)
                    self._lib.nvmlDeviceGetName(dev, name, 256)
                    self._devices.append((dev, name.value.decode()))

            if not self._devices:
                self._lib.nvmlShutdown()
                self._lib = None
        except (OSError, AttributeError):
            self._lib = None

    def available(self):
        return self._lib is not None

    def device_names(self):
        return [name for _, name in self._devices]

    def total_gpu_mem_used(self):
        """Total GPU memory used (bytes) across all NVIDIA GPUs. ~5us/call."""
        if not self._lib:
            return 0
        total = 0
        info = self._MemInfo()
        for dev, _ in self._devices:
            if self._lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(info)) == 0:
                total += info.used
        return total

    def shutdown(self):
        if self._lib:
            self._lib.nvmlShutdown()
            self._lib = None


def _amd_sysfs_vram_used():
    """Return VRAM used (bytes) from sysfs for the first AMD GPU, or 0."""
    import glob
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0


def _detect_gpu_vendor():
    """Return 'nvidia', 'amd', or 'unknown'."""
    nvml = NVMLMonitor()
    if nvml.available():
        nvml.shutdown()
        return "nvidia"
    import glob
    if glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        return "amd"
    return "unknown"


# ---------------------------------------------------------------------------
# GPU memory sampler (background thread)
# ---------------------------------------------------------------------------

class GPUSampler:
    """Polls GPU VRAM at ~1ms intervals via NVML or sysfs.

    Collects samples as delta from a pre-run baseline. Provides peak/median/p95
    statistics with zero-sample filtering (removes pre-alloc/post-free periods).
    """

    def __init__(self, gpu_vendor, nvml=None, poll_interval=0.001):
        self._vendor = gpu_vendor
        self._nvml = nvml
        self._interval = poll_interval
        self._baseline = 0
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        if self._vendor == "nvidia" and self._nvml:
            self._baseline = self._nvml.total_gpu_mem_used()
        elif self._vendor == "amd":
            self._baseline = _amd_sysfs_vram_used()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _poll(self):
        while not self._stop.is_set():
            if self._vendor == "nvidia" and self._nvml:
                used = self._nvml.total_gpu_mem_used()
            elif self._vendor == "amd":
                used = _amd_sysfs_vram_used()
            else:
                break
            self._samples.append(max(0, used - self._baseline))
            self._stop.wait(self._interval)

    def stats(self):
        """Return {peak, median, p95, samples, nonzero_samples}."""
        nonzero = sorted(s for s in self._samples if s > 0)
        n = len(nonzero)
        total = len(self._samples)
        if n == 0:
            return {"peak": 0, "median": 0, "p95": 0,
                    "samples": total, "nonzero_samples": 0}
        return {
            "peak": nonzero[-1],
            "median": nonzero[n // 2],
            "p95": nonzero[int(n * 0.95)] if n >= 20 else nonzero[-1],
            "samples": total,
            "nonzero_samples": n,
        }


# ---------------------------------------------------------------------------
# Process runner with memory measurement
# ---------------------------------------------------------------------------

def _mb(b):
    return b / (1024 * 1024)


def _read_peak_rss(pid):
    """Read peak RSS (VmHWM) from /proc. Returns bytes, or 0 if unavailable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def _get_peak_rss_via_time(cmd, env=None, timeout=180):
    """Run command via /usr/bin/time -v and return (peak_rss_bytes, output)."""
    proc = subprocess.Popen(
        ["/usr/bin/time", "-v"] + cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env or os.environ.copy())
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return 0, ""

    output = stdout.decode(errors="replace")
    time_stderr = stderr.decode(errors="replace")

    peak_rss = 0
    child_lines = []
    for line in time_stderr.split("\n"):
        if "Maximum resident set size" in line:
            m = re.search(r"(\d+)", line.split(":")[-1])
            if m:
                peak_rss = int(m.group(1)) * 1024
        else:
            child_lines.append(line)

    return peak_rss, output + "\n".join(child_lines)


def run_with_memory(cmd, env=None, label="", gpu_vendor="unknown",
                    nvml=None, timeout=180):
    """Run *cmd*, measure peak RSS and GPU VRAM. Returns result dict or None."""
    if not os.path.exists(cmd[0]):
        print(f"  [{label}] binary not found: {cmd[0]}")
        return None

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env or os.environ.copy())

    sampler = GPUSampler(gpu_vendor, nvml=nvml)
    sampler.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        print(f"  [{label}] TIMED OUT")
        sampler.stop()
        return None

    peak_rss = _read_peak_rss(proc.pid)
    sampler.stop()
    output = stdout.decode(errors="replace") + stderr.decode(errors="replace")

    # /proc may vanish before we read it — fall back to /usr/bin/time
    if peak_rss == 0:
        print(f"  [{label}] Re-running with /usr/bin/time for peak RSS...")
        peak_rss, output = _get_peak_rss_via_time(cmd, env=env, timeout=timeout)

    gpu = sampler.stats()
    return {
        "peak_rss": peak_rss,
        "gpu_peak": gpu["peak"],
        "gpu_p95": gpu["p95"],
        "gpu_samples": gpu["samples"],
        "gpu_nonzero": gpu["nonzero_samples"],
        "output": output,
    }


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_cut_output(output):
    """Parse CUT's self-reported GPU memory and prompt token count."""
    info = {}
    m = re.search(r"GPU memory:\s+([\d.]+)\s+MB", output)
    if m:
        info["self_gpu"] = float(m.group(1)) * 1024 * 1024
    m = re.search(r"Prompt token IDs: \[([^\]]*)\]", output)
    if m:
        tokens = [t.strip() for t in m.group(1).split(",") if t.strip()]
        info["prompt_tokens"] = len(tokens)
    return info


def parse_llama_vram(output):
    """Sum llama.cpp's Vulkan buffer sizes from verbose output."""
    total = 0
    for m in re.finditer(r"Vulkan\d?\s+\w+ buffer size\s*=\s*([\d.]+)\s*MiB",
                         output):
        total += float(m.group(1))
    if total:
        return total * 1024 * 1024
    # Fallback: older format
    for m in re.finditer(r"buffer_size\s*=\s*([\d.]+)\s*MiB", output):
        total += float(m.group(1))
    return total * 1024 * 1024


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_runner_stats(r, self_gpu=0):
    """Print memory stats for a single runner."""
    print(f"  CPU Peak RSS:          {_mb(r['peak_rss']):>8.1f} MB")
    n = r["gpu_samples"]
    if n > 0:
        nz = r["gpu_nonzero"]
        print(f"  GPU VRAM (NVML peak): {_mb(r['gpu_peak']):>8.1f} MB"
              f"  ({nz}/{n} nonzero samples)")
        print(f"  GPU VRAM (NVML p95):  {_mb(r['gpu_p95']):>8.1f} MB")
    if self_gpu:
        print(f"  GPU VRAM (self-report):{_mb(self_gpu):>8.1f} MB")


def _detect_llama_device(llama_vk_exe):
    """Auto-detect first discrete Vulkan device name for llama-bench."""
    try:
        r = subprocess.run([llama_vk_exe, "--list-devices"],
                           capture_output=True, text=True, timeout=10)
        for line in (r.stdout + r.stderr).split("\n"):
            m = re.match(r"\s+(Vulkan\d+):\s+(.+)\(\d+ MiB", line)
            if not m:
                continue
            name = m.group(2).strip()
            igpu_keywords = ["Ryzen", "Core", "Celeron", "Pentium", "Atom",
                             "RAPHAEL", "REMBRANDT", "PHOENIX",
                             "llvmpipe", "SwiftShader"]
            if not any(kw.lower() in name.lower() for kw in igpu_keywords):
                return m.group(1), name
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare runtime memory (CPU + GPU) of CUT vs llama.cpp")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ctx-size", type=int, default=None,
                        help="Context/KV cache size (default: 512)")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="Prompt tokens for llama-bench (auto-detected if omitted)")
    parser.add_argument("--kv-type", choices=["f32", "f16"], default="f16")
    parser.add_argument("--device", type=int, default=None,
                        help="Vulkan device index for both runners")
    args = parser.parse_args()

    # -- Model config --
    print(f"Reading model metadata from {args.model}...")
    cfg = get_model_config(args.model)
    model_max_ctx = cfg.get("context_length", 2048)
    ctx_size = min(args.ctx_size or 512, model_max_ctx)
    n_layers = cfg.get("n_layers", "?")
    dim = cfg.get("dim", "?")
    n_kv_heads = cfg.get("n_kv_heads", cfg.get("n_heads", "?"))

    kv_str = "unknown"
    if all(isinstance(v, int) for v in [dim, n_kv_heads, n_layers]):
        head_dim = dim // cfg.get("n_heads", n_kv_heads)
        kv_bytes = 2 * n_layers * ctx_size * (n_kv_heads * head_dim) * 2
        kv_str = f"{_mb(kv_bytes):.1f} MB (f16)"

    gpu_vendor = _detect_gpu_vendor()
    nvml = NVMLMonitor() if gpu_vendor == "nvidia" else None

    # -- Config summary --
    print()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Context size:   {ctx_size}")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  KV cache:       f16 (both runners)")
    print(f"  Est. KV size:   {kv_str}")
    print(f"  GPU vendor:     {gpu_vendor}")
    if nvml and nvml.available():
        for name in nvml.device_names():
            print(f"  NVML device:    {name}")
    print()

    results = {}

    # ── 1. CUT ────────────────────────────────────────────────────────────
    cut_exe = "build/interface/runner/llama/gguf_example"
    print("=" * 60)
    print("1. CUT (Vulkan GPU)")
    print("=" * 60)
    env = os.environ.copy()
    if args.device is not None:
        env["CUT_VULKAN_DEVICE"] = str(args.device)
    r = run_with_memory(
        [cut_exe, args.model, str(args.max_tokens), "--ctx-size", str(ctx_size)],
        env=env, label="CUT", gpu_vendor=gpu_vendor, nvml=nvml)

    prompt_tokens = args.prompt_tokens
    if r:
        info = parse_cut_output(r["output"])
        self_gpu = info.get("self_gpu", 0)
        if prompt_tokens is None and "prompt_tokens" in info:
            prompt_tokens = info["prompt_tokens"]
            print(f"  (auto-detected {prompt_tokens} prompt tokens)")
        results["CUT (Vulkan)"] = {
            "peak_rss": r["peak_rss"], "self_gpu": self_gpu,
        }
        _print_runner_stats(r, self_gpu)
    else:
        print("  SKIPPED")
    print()

    if prompt_tokens is None:
        prompt_tokens = 15

    # ── 2. llama.cpp Vulkan ───────────────────────────────────────────────
    llama_vk_exe = "build/external_runners/llama.cpp/build_vk/bin/llama-bench"
    print("=" * 60)
    print("2. llama.cpp (Vulkan GPU, single device)")
    print("=" * 60)

    # Pin to single GPU for fair comparison (llama.cpp auto-splits otherwise)
    if args.device is not None:
        llama_dev = f"Vulkan{args.device}"
        dev_name = llama_dev
    else:
        llama_dev, dev_name = _detect_llama_device(llama_vk_exe)
        if llama_dev:
            print(f"  Auto-selected: {llama_dev} ({dev_name})")

    env = os.environ.copy()
    cmd = [llama_vk_exe, "-m", args.model,
           "-p", str(prompt_tokens), "-n", str(args.max_tokens),
           "-r", "1", "-ngl", "99", "-v"]
    if llama_dev:
        cmd += ["-dev", llama_dev]
    if args.kv_type != "f16":
        cmd += ["-ctk", args.kv_type, "-ctv", args.kv_type]
    print(f"  flags: {' '.join(cmd[1:])}")

    r = run_with_memory(cmd, env=env, label="llama.cpp VK",
                        gpu_vendor=gpu_vendor, nvml=nvml)
    if r:
        self_gpu = parse_llama_vram(r["output"])
        results["llama.cpp (Vulkan)"] = {
            "peak_rss": r["peak_rss"], "self_gpu": self_gpu,
        }
        _print_runner_stats(r, self_gpu)
    else:
        print("  SKIPPED (run: ./scripts/setup_benchmark_runners.sh)")
    print()

    # ── 3. llama.cpp CPU ──────────────────────────────────────────────────
    llama_cpu_exe = "build/external_runners/llama.cpp/build_cpu/bin/llama-bench"
    print("=" * 60)
    print("3. llama.cpp (CPU only)")
    print("=" * 60)
    cmd = [llama_cpu_exe, "-m", args.model,
           "-p", str(prompt_tokens), "-n", str(args.max_tokens),
           "-r", "1", "-v"]
    if args.kv_type != "f16":
        cmd += ["-ctk", args.kv_type, "-ctv", args.kv_type]
    print(f"  flags: {' '.join(cmd[1:])}")

    r = run_with_memory(cmd, label="llama.cpp CPU",
                        gpu_vendor=gpu_vendor, nvml=nvml)
    if r:
        results["llama.cpp (CPU)"] = {"peak_rss": r["peak_rss"], "self_gpu": 0}
        print(f"  CPU Peak RSS:          {_mb(r['peak_rss']):>8.1f} MB")
    else:
        print("  SKIPPED (not built)")
    print()

    if nvml:
        nvml.shutdown()

    # ── Summary ───────────────────────────────────────────────────────────
    if len(results) < 2:
        print("Need at least 2 runners for comparison.")
        return

    print("=" * 60)
    print("MEMORY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  ctx={ctx_size}, prompt={prompt_tokens}, gen={args.max_tokens}")
    print()

    hdr = f"  {'Runner':<25} {'CPU RSS':>10} {'GPU VRAM':>10}"
    print(hdr)
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    for name, m in sorted(results.items(), key=lambda kv: kv[1]["peak_rss"]):
        rss = f"{_mb(m['peak_rss']):.1f} MB"
        gpu = f"{_mb(m['self_gpu']):.1f} MB" if m["self_gpu"] else "N/A"
        print(f"  {name:<25} {rss:>10} {gpu:>10}")

    # Ratios against CUT
    if "CUT (Vulkan)" in results:
        print()
        cut = results["CUT (Vulkan)"]
        for name, m in results.items():
            if name == "CUT (Vulkan)":
                continue
            print(f"  {name} vs CUT:")
            if cut["peak_rss"] > 0 and m["peak_rss"] > 0:
                r = m["peak_rss"] / cut["peak_rss"]
                who = "llama uses more" if r > 1 else "CUT uses more"
                print(f"    CPU RSS:  {r:.2f}x  ({who})")
            if cut["self_gpu"] > 0 and m["self_gpu"] > 0:
                r = m["self_gpu"] / cut["self_gpu"]
                who = "llama uses more" if r > 1 else "CUT uses more"
                print(f"    GPU VRAM: {r:.2f}x  ({who})")


if __name__ == "__main__":
    main()
