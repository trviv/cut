#!/usr/bin/env python3
"""Model-level benchmark: CUT vs llama.cpp, one case per model / backend / shape.

This is the model-scale sibling of benchmarks/vendor: there the reference is the
kernel library the GPU vendor ships and the question is "how far off vendor-peak
is this operator"; here the reference is the runtime everyone actually measures
LLM inference with, and the question is "how far off llama.cpp is a whole
forward pass". Methodology — what is held equal between the two sides, and what
could not be — is in benchmarks/model/README.md.

    ./scripts/bench/model_bench.sh --quick          # build + run + compare
    python3 scripts/bench/model_compare.py model_bench_results/*.json
    python3 scripts/bench/render_model_bench.py --snapshot

Two measurements per case, named the way llama-bench names them:

  * prefill (pp) — process N prompt tokens into an empty KV cache.
  * decode  (tg) — generate M tokens with N already in the cache.

They are separate cases because they are separate regimes: prefill is a GEMM
over the whole prompt and decode is a GEMV per token, and a runtime can be
strong at one and weak at the other. Reporting only their sum would hide that.

The device invocation of each side is NOT reimplemented here — run_cut and
run_llama come from benchmark_gpus.py, so there is exactly one place that knows
how each runner is launched and parsed. A second copy is a copy that drifts,
which is the same reason the vendor benches share CublasBench.h.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_gpus import (enumerate_gpus, run_capturing_mem, run_cut,
                            run_llama, short_gpu_name)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------- the case table
#
# Shapes are llama-bench's own defaults and a smaller tier, so a CUT number can
# be read next to any llama.cpp result published anywhere. pp512/tg128 is the
# figure the llama.cpp project itself quotes; pp128/tg32 exists because the full
# tier costs minutes per model on the slower backend and a smoke run should not.
TIERS = [
    # (label, prompt tokens, generated tokens, in the --quick set)
    ("pp128 tg32", 128, 32, True),
    ("pp512 tg128", 512, 128, False),
]

# Models are named by file, and the file is the whole specification: the
# quantisation is in the name and the architecture is in the GGUF. This set
# spans the quantisations CUT dequantises (f16, Q4_K, Q5_K, Q6_K, Q8_0) and
# keeps one qwen2 model, which is the only non-llama attention path in the zoo
# (its QKV bias) and therefore the only coverage of it.
#
# `quick` marks the ones a smoke run uses; `notes` explains a model's presence
# where the file name does not.
MODELS = [
    # (filename, quick, notes)
    ("SmolLM2-135M-Instruct-Q8_0.gguf", True, "smallest model that still generates language"),
    ("tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf", True, "Q4_K_S, SPM tokenizer"),
    ("Llama-3.2-1B-Instruct-f16.gguf", True, "unquantised weights"),
    ("Qwen2.5-1.5B-Instruct-Q6_K.gguf", False, "only qwen2 in the set: covers the QKV-bias path"),
    ("Llama-3.2-3B-Instruct-Q4_K_M.gguf", False, ""),
    ("Llama-3.2-3B-Instruct-Q8_0.gguf", False, ""),
    ("Mistral-7B-Instruct-v0.3-Q8_0.gguf", False, "largest that fits a 24 GB card with room for KV"),
]

# The coherence check's prompt. Deliberately a question with a short factual
# answer: a runner that has broken in the way quantisation bugs break it stops
# producing language at all, which is visible in one line of output.
CHECK_PROMPT = "What is the capital of France?"
CHECK_TOKENS = 24

# A check fails below this many distinct token ids out of CHECK_TOKENS. It is a
# collapse detector: a runner whose weights are wrong emits one token forever.
# Well above what any working model produces (measured: 15-20 distinct of 24)
# and well below what a degenerate one does (1-2).
CHECK_MIN_DISTINCT = 4


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def free_vram_gb(idx):
    """Free VRAM on NVIDIA device @p idx, or None when it cannot be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits", "-i", str(idx)],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().split("\n")[0]) / 1024.0
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None


def timeout_for(model_bytes, pp, tg):
    """A generous per-run ceiling that still catches a hang.

    Scaled by model size because the slow combination — a 7B model prefilling
    512 tokens on the Vulkan backend — is minutes of legitimate work, and a
    fixed timeout tuned for it would let a hung 135M model burn ten of them.
    """
    return int(120 + (model_bytes / 1e9) * 60 + (pp + tg) * 0.5)


def coherence_check(cut_binary, model_path, backend, dev_idx, nvidia_idx, timeout):
    """Run CUT for real — sampling, argmax readback, detokenisation — and judge
    whether what came out is language.

    This is the model-scale answer to the vendor suite's correctness gate, and
    it is a WEAKER instrument; benchmarks/model/README.md says so at length. It
    cannot compare against llama.cpp token for token, because the llama.cpp
    build here exposes no logits and applies its own chat template, so the two
    runners are not answering the same question at the token level. What it does
    catch is the failure that actually happens — a quantisation or kernel bug
    that leaves the runner emitting one token forever while still reporting a
    perfectly good tok/s.
    """
    env = os.environ.copy()
    env["CUT_DEVICES"] = f"{backend}:{dev_idx}"
    env.pop("CUT_BENCH_SYNTHETIC", None)   # the point is the REAL decode path
    cmd = [cut_binary, model_path, str(CHECK_TOKENS), CHECK_PROMPT]
    try:
        stdout, stderr, _ = run_capturing_mem(cmd, env, timeout, None)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"status": "FAIL", "detail": f"generation did not finish ({e})"}

    # Parsed from stdout alone. The runner's diagnostics go to stderr, and
    # reading the concatenation would splice them onto the end of the generated
    # text — which then reads as output the model produced.
    ids_match = re.search(r"Generated token IDs: \[([^\]]*)\]", stdout)
    text_match = re.search(r"Decoded text:\n(.*)", stdout, re.S)
    if not ids_match:
        return {"status": "FAIL",
                "detail": "the runner produced no token list "
                          f"({(stderr or stdout).strip().splitlines()[-1:] or ['no output']})"}

    all_ids = [int(t) for t in ids_match.group(1).split(",") if t.strip()]
    prompt_match = re.search(r"Prompt token IDs: \[([^\]]*)\]", stdout)
    prompt_len = len(re.findall(r"\d+", prompt_match.group(1))) if prompt_match else 0
    generated = all_ids[prompt_len:] if prompt_len else all_ids
    text = text_match.group(1).strip() if text_match else ""
    # detokenize() is given prompt + generation, so the prompt comes back out
    # with it; keep the assistant's turn, which is the part under test.
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1].strip()

    distinct = len(set(generated))
    # Whitespace-collapsed: the snippet is quoted on one line in the table and
    # in the JSON, and a model that answers with a list would otherwise break it.
    result = {"distinct": distinct, "generated": len(generated),
              "text": " ".join(text.split())[:200]}
    if not generated:
        result.update(status="FAIL", detail="no tokens generated")
    elif not text:
        result.update(status="FAIL", detail="generated tokens detokenised to nothing")
    # Diversity is only asked of a run long enough for the question to mean
    # something. A model that answers in three tokens and stops has demonstrated
    # a working EOS, not a collapse; a collapsed one never stops, so it always
    # reaches the full length and is always judged.
    elif len(generated) >= 2 * CHECK_MIN_DISTINCT and distinct < CHECK_MIN_DISTINCT:
        result.update(status="FAIL",
                      detail=f"degenerate output: {distinct} distinct token(s) "
                             f"in {len(generated)} generated")
    else:
        result.update(status="ok", detail=None)
    return result


def measure(cut_binary, llama_exe, model_path, backend, cut_idx, llama_idx,
            pp, tg, runs, warmup, nvidia_idx, timeout):
    """One case: both phases from CUT, both phases from llama.cpp.

    CUT reports prefill and decode from a single generation, so one process
    covers both. llama.cpp needs two, and that is not a convenience but the
    point: `-p N -n 0 -d 0` prefills into an empty cache, and `-p 0 -n M -d N`
    generates with N tokens already cached — which is the depth CUT's decode
    actually runs at, since its decode follows its own prefill. Asking
    llama-bench for both phases in one call would measure its generation from an
    empty context and quietly credit it with the cheaper attention.
    """
    # tg+1: CUT counts the first token as prefill's output and generates
    # max_new_tokens-1 more, so tg+1 makes its decode loop exactly tg tokens
    # against llama-bench's -n tg.
    cut = run_cut(cut_binary, model_path, backend, cut_idx, tg + 1,
                  runs=runs, warmup=warmup, synthetic=True,
                  nvidia_idx=nvidia_idx, prompt_tokens=pp,
                  ctx_size=pp + tg + 8, timeout=timeout)

    ref_pp = run_llama(llama_exe, model_path, backend, llama_idx, 0, runs=runs,
                       nvidia_idx=nvidia_idx, n_prompt=pp, n_depth=0,
                       timeout=timeout)
    ref_tg = run_llama(llama_exe, model_path, backend, llama_idx, tg, runs=runs,
                       nvidia_idx=nvidia_idx, n_prompt=0, n_depth=pp,
                       timeout=timeout)
    return cut, ref_pp, ref_tg


def case_entry(cut_tps, ref_tps, cut_mib, ref_mib, tokens, status, detail):
    """One comparison, stored as BOTH a rate and a time.

    tok/s is what a reader of this benchmark wants; milliseconds is what the
    renderer shares with the vendor page, where a cell is ref_ms/cut_ms. Storing
    both means neither consumer has to know the other's convention, and the two
    can never disagree because they are computed from the same measurement here.
    """
    entry = {
        "cut_tps": cut_tps,
        "ref_tps": ref_tps,
        "cut_ms": (1000.0 * tokens / cut_tps) if cut_tps else None,
        "ref_ms": (1000.0 * tokens / ref_tps) if ref_tps else None,
        "tokens": tokens,
        "ref": "llama.cpp",
        "status": status,
    }
    if cut_mib is not None:
        entry["cut_mib"] = cut_mib
    if ref_mib is not None:
        entry["ref_mib"] = ref_mib
    if detail:
        entry["detail"] = detail
    return entry


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*",
                    help="GGUF paths to measure instead of the built-in set")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "model_bench_results"),
                    help="where to write one JSON per GPU (default: model_bench_results/)")
    ap.add_argument("--cut-binary", default=None, help="path to gguf_example")
    ap.add_argument("--llama-dir", default="build/external_runners/llama.cpp",
                    help="llama.cpp build directory")
    ap.add_argument("--runs", type=int, default=5,
                    help="timed runs per measurement; the MEDIAN is reported (default 5)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed runs before the clock starts (default 1)")
    ap.add_argument("--quick", action="store_true",
                    help="the small models and the small tier only — a couple of minutes")
    ap.add_argument("--tier", action="append", metavar="ppN,tgM",
                    help="measure this shape instead of the built-in tiers; repeatable")
    ap.add_argument("--backend", action="append", choices=["cuda", "vulkan"],
                    help="restrict to one backend (default: every one the GPU has)")
    ap.add_argument("--gpu", metavar="SUBSTRING",
                    help="restrict to GPUs whose name contains this")
    ap.add_argument("--include-igpu", action="store_true",
                    help="also measure integrated GPUs (excluded by default)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the coherence check; every case then reports status 'unchecked'")
    args = ap.parse_args()

    # A full sweep is tens of minutes and is usually watched through a redirect
    # or a pipe, where Python's stdout is block-buffered: without this the run
    # shows nothing at all and then everything at once, which is
    # indistinguishable from a hang.
    sys.stdout.reconfigure(line_buffering=True)

    cut_binary = args.cut_binary
    if cut_binary is None:
        for cand in ("build-cuda-rel/interface/runner/llama/gguf_example",
                     "build-release/interface/runner/llama/gguf_example",
                     "build-cuda/interface/runner/llama/gguf_example",
                     "build/interface/runner/llama/gguf_example"):
            if os.path.exists(cand):
                cut_binary = cand
                break
    if not cut_binary or not os.path.exists(cut_binary):
        sys.exit("error: gguf_example not found — build it or pass --cut-binary")

    llama_vk = os.path.join(args.llama_dir, "build_vk/bin/llama-bench")
    llama_cuda = os.path.join(args.llama_dir, "build_cuda/bin/llama-bench")

    # Tiers
    if args.tier:
        tiers = []
        for spec in args.tier:
            m = re.fullmatch(r"pp(\d+),tg(\d+)", spec.strip())
            if not m:
                sys.exit(f"error: --tier wants ppN,tgM (got {spec!r})")
            pp, tg = int(m.group(1)), int(m.group(2))
            tiers.append((f"pp{pp} tg{tg}", pp, tg))
    else:
        tiers = [(label, pp, tg) for label, pp, tg, quick in TIERS
                 if quick or not args.quick]

    # Models: explicit paths win; otherwise the built-in set, filtered to what
    # is actually on disk. A missing model is reported rather than assumed
    # absent on purpose — the set is small and named, so a gap is a fact.
    if args.models:
        model_paths = args.models
    else:
        model_paths = []
        for name, quick, _notes in MODELS:
            if args.quick and not quick:
                continue
            path = os.path.join(PROJECT_DIR, "models", name)
            if os.path.exists(path):
                model_paths.append(path)
            else:
                print(f"  missing, skipped: models/{name}", file=sys.stderr)
    model_paths = [p for p in model_paths if os.path.exists(p)]
    if not model_paths:
        sys.exit("error: no models to measure")

    gpus = enumerate_gpus(cut_binary, llama_vk)
    if not args.include_igpu:
        gpus = [g for g in gpus if not g["igpu"]]
    if args.gpu:
        gpus = [g for g in gpus if args.gpu.lower() in g["name"].lower()]
    if not gpus:
        sys.exit("error: no GPUs to measure (integrated ones need --include-igpu)")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"CUT binary: {cut_binary}")
    print(f"Models:     {len(model_paths)}  Tiers: {', '.join(t[0] for t in tiers)}")
    print(f"Runs:       {args.warmup} warmup + {args.runs} timed, median reported")
    print()

    exit_code = 0
    for gpu in gpus:
        backends = []
        if gpu["cuda"] is not None and os.path.exists(llama_cuda):
            backends.append(("cuda", gpu["cuda"], gpu["cuda"], llama_cuda))
        elif gpu["cuda"] is not None:
            print(f"  {gpu['name']}: llama.cpp CUDA bench not built — CUDA cases skipped",
                  file=sys.stderr)
        if gpu["cut_vk"] is not None and gpu["llama_vk"] is not None and os.path.exists(llama_vk):
            backends.append(("vulkan", gpu["cut_vk"], gpu["llama_vk"], llama_vk))
        if args.backend:
            backends = [b for b in backends if b[0] in args.backend]
        if not backends:
            print(f"  {gpu['name']}: no backend both runtimes can use — skipped",
                  file=sys.stderr)
            continue

        snapshot = {"schema": 1, "gpu": gpu["name"], "runs": args.runs,
                    "warmup": args.warmup, "cases": {}, "checks": {}}
        headroom = free_vram_gb(gpu["cuda"]) if gpu["cuda"] is not None else None

        print("=" * 78)
        print(f"{gpu['name']}  ({', '.join(b[0] for b in backends)})")
        print("=" * 78)

        for model_path in model_paths:
            label = os.path.basename(model_path)[:-len(".gguf")]
            model_bytes = os.path.getsize(model_path)
            if headroom is not None and model_bytes / 1e9 > headroom:
                print(f"  {label}: {model_bytes / 1e9:.1f} GB of weights against "
                      f"{headroom:.1f} GB free — skipped", file=sys.stderr)
                continue

            for backend, cut_idx, llama_idx, llama_exe in backends:
                check = {"status": "unchecked", "detail": None}
                if not args.no_check:
                    check = coherence_check(cut_binary, model_path, backend, cut_idx,
                                            gpu["cuda"], timeout_for(model_bytes, 0, CHECK_TOKENS))
                    snapshot["checks"][f"{label} {backend}"] = check
                    if check["status"] == "FAIL":
                        print(f"  {label} {backend}: COHERENCE FAILURE — {check['detail']}",
                              file=sys.stderr)
                        exit_code = 2

                for tier_label, pp, tg in tiers:
                    started = time.time()
                    cut, ref_pp, ref_tg = measure(
                        cut_binary, llama_exe, model_path, backend, cut_idx, llama_idx,
                        pp, tg, args.runs, args.warmup, gpu["cuda"],
                        timeout_for(model_bytes, pp, tg))

                    ref_mib = max([m for m in (ref_pp.get("mem_mib") if ref_pp else None,
                                               ref_tg.get("mem_mib") if ref_tg else None)
                                   if m is not None], default=None)
                    cut_mib = cut.get("mem_mib") if cut else None

                    # A side that produced no number fails ITS OWN case, not
                    # the pair's: llama.cpp is invoked once per phase and either
                    # invocation can die on its own. Recording the survivor as
                    # "ok" with an empty column would be indistinguishable on
                    # the page from a case this snapshot never ran, and only one
                    # of those is a problem.
                    def verdict(cut_side, ref_side):
                        if cut_side is None:
                            return "FAIL", "CUT produced no result"
                        if ref_side is None:
                            return "FAIL", "llama.cpp produced no result"
                        # Otherwise the case inherits the model's coherence
                        # verdict, which is what decides whether its numbers are
                        # quotable at all.
                        return check["status"], check["detail"]

                    pp_status, pp_detail = verdict(
                        (cut or {}).get("prefill_tps"), (ref_pp or {}).get("prefill_tps"))
                    tg_status, tg_detail = verdict(
                        (cut or {}).get("decode_tps"), (ref_tg or {}).get("decode_tps"))
                    if "FAIL" in (pp_status, tg_status):
                        exit_code = 2

                    snapshot["cases"][f"prefill/{label} {backend} pp{pp}"] = case_entry(
                        (cut or {}).get("prefill_tps"), (ref_pp or {}).get("prefill_tps"),
                        cut_mib, ref_mib, pp, pp_status, pp_detail)
                    snapshot["cases"][f"decode/{label} {backend} tg{tg}@{pp}"] = case_entry(
                        (cut or {}).get("decode_tps"), (ref_tg or {}).get("decode_tps"),
                        cut_mib, ref_mib, tg, tg_status, tg_detail)
                    status = "FAIL" if "FAIL" in (pp_status, tg_status) else pp_status

                    def ratio(a, b):
                        return f"{a / b:.2f}x" if a and b else "-"

                    print(f"  {label:<38} {backend:<7} {tier_label:<12} "
                          f"pp {ratio((cut or {}).get('prefill_tps'), (ref_pp or {}).get('prefill_tps')):>7}  "
                          f"tg {ratio((cut or {}).get('decode_tps'), (ref_tg or {}).get('decode_tps')):>7}  "
                          f"[{time.time() - started:.0f}s]"
                          f"{'  ' + status if status != 'ok' else ''}")

        path = os.path.join(args.out_dir, f"{slug(gpu['name'])}.json")
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"\n  wrote {os.path.relpath(path, PROJECT_DIR)} "
              f"({len(snapshot['cases'])} comparisons)\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
