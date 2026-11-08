# Model comparison benchmarks

How far off llama.cpp is CUT, on this model, on this GPU? One level up from
[`../vendor/`](../vendor/README.md): there the reference is the kernel library
the GPU vendor ships and a case is one operator on one shape; here the reference
is the runtime everyone actually measures LLM inference with, and a case is a
whole forward pass over real weights.

Both questions are worth asking and neither answers the other. An operator that
reaches 0.9x of cuBLAS says nothing about how a model built from thirty of them
schedules, and a model that reaches 0.9x of llama.cpp says nothing about which
kernel to fix next.

Measured results are not kept here — they are recorded as columns in
[`../results/model_history.json`](../results/model_history.json) and rendered on
[the results page](../README.md).

## Layout

Unlike the vendor suite, nothing here is a C++ binary. The two runtimes are
separate processes with their own CLIs, so the "bench" is the driver that runs
them and the case table lives in it.

```
scripts/bench/
├── model_bench.sh           build the runner, run the matrix, print the table
├── model_bench.py           the CASE TABLE, both sides' invocation, the checks
├── model_compare.py         join the per-GPU JSON into one table; exit 2 on failure
├── render_model_bench.py    fold a run into model_history.json, redraw the page
└── benchmark_gpus.py        run_cut / run_llama / GPU enumeration, shared
```

`model_bench.py` does not reimplement how either runtime is launched or parsed:
`run_cut` and `run_llama` come from `benchmark_gpus.py`, which is the one place
that knows that CUT reports `Prefill:`/`Decode:` lines and that llama-bench's
JSON carries `samples_ts` per repetition. This is the same discipline
`CublasBench.h` exists for in the vendor suite — a second copy of "how the
reference is called" is a copy that drifts, and it drifts silently, as a
plausible number.

## Building and running

```bash
./scripts/bench/model_bench.sh --quick                  # small models, small tier
./scripts/bench/model_bench.sh                          # the full matrix
python3 scripts/bench/render_model_bench.py --snapshot  # record a column, redraw the page
```

`model_bench.sh` builds `gguf_example` by default, and that matters for the same
reason it matters in the vendor suite: the operator variant table is compiled in
and `tuning_data.json` is read by index at runtime, so a stale binary measures a
different kernel than the one you think you are testing.

The reference has to be built first — `./scripts/setup/setup_benchmark_runners.sh`
produces `build_vk/`, `build_cuda/` and `build_cpu/` llama.cpp trees. The CUDA
one needs `nvcc`; without it the NVIDIA CUDA rows are skipped with a note, while
CUT's own CUDA backend still runs, since it compiles through NVRTC and needs no
toolkit.

Useful flags: `--quick`, `--runs N` (default 5), `--warmup N` (default 1),
`--tier ppN,tgM` (repeatable, replaces the built-in tiers), `--backend
cuda|vulkan`, `--gpu SUBSTRING`, `--include-igpu`, `--no-check`, `--no-build`,
`--out-dir DIR`. Trailing arguments are GGUF paths and replace the model set.

### The case table

Two shapes per model, named the way llama-bench names them:

| tier | prompt tokens | generated tokens | in `--quick` |
|---|---|---|---|
| `pp128 tg32` | 128 | 32 | yes |
| `pp512 tg128` | 512 | 128 | no |

`pp512 tg128` is llama-bench's own default, which is why it is here: it is the
shape almost every published llama.cpp number uses, so a CUT figure at that
shape can be read next to results from machines this repository will never see.

The model set spans the quantisations CUT dequantises (f16, Q4_K, Q5_K, Q6_K,
Q8_0) from 135M to 7B, and keeps one qwen2 model because its QKV bias is the
only non-LLaMA attention path in the zoo and therefore its only coverage.
Models are skipped, loudly, when their weights alone exceed free VRAM.

## Methodology

Getting this comparison honest matters more than getting it favourable.

### Both sides run the same workload

- **The same number of prompt tokens.** This is the one that was wrong before
  this suite existed. CUT used to prefill its tokenized chat prompt while
  llama-bench was told `-p 15`, and the two matched only because SmolLM2's
  ChatML wrap happens to come to 15 tokens — on any other tokenizer the
  comparison silently drifted, and the old table carried a note telling readers
  not to trust the prefill column. `gguf_example --bench-prompt-tokens N` now
  builds a synthetic prompt of exactly N tokens from a fixed-seed LCG, bypassing
  the template and the tokenizer, which is what llama-bench's `pp` test does
  too. The prefill column means something now.
- **The same number of generated tokens, at the same KV depth.** CUT counts the
  first token as prefill's output and generates `max_new_tokens - 1` more, so
  the harness asks it for `tg + 1`. Decode is measured after CUT's own prefill,
  with N tokens in the cache, so llama-bench is asked for `-n M -d N` — its
  `-d` places N tokens untimed before the test. Without that flag the reference
  would generate from an empty context and be credited with the cheaper
  attention. Prefill is the mirror image: `-p N -n 0 -d 0`, into an empty cache.
  This is why the reference is invoked twice per case and CUT once.
- **The same early-stopping rule, which is none.** CUT stops at EOS by default,
  which on a chat model meant decoding ~15 of the requested 32 tokens and
  amortising the fixed first-token cost over half as many — a real and
  substantial self-inflicted loss. `--no-stop` forces the full length, matching
  llama-bench's `-n`, which never stops early.
- **The same decode path, minus the parts llama-bench does not run.**
  `CUT_BENCH_SYNTHETIC=1` skips the per-token repetition-penalty upload, the
  argmax readback and the printing, because llama-bench's `tg` test does none of
  those either — it measures forward passes. `forward()` itself is untouched.
  Turn it off (`benchmark_gpus.py --cut-real-gen`) and CUT's Vulkan decode drops
  about 30%: that gap is the sampling round trip, and it is a real cost to a
  real user, but charging it to one side of a two-sided comparison would not
  measure it, it would just hide it in the ratio.
- **Full offload on both sides.** llama.cpp gets `-ngl 99`; CUT places every
  layer on the device. Flash attention is left off on the reference (`-fa 0`,
  the default in this build) because CUT has no equivalent, and turning on an
  algorithm only one side has makes the number a fact about that algorithm.
- **The same process ordering.** Each case is a fresh process per side, and CUT
  is always measured first, so if anything the reference inherits a warmer card.

### Timing

- **Both sides warm up, then report a median.** CUT runs `--bench-warmup 1
  --bench-runs 5` inside a single process; llama-bench runs `-r 5` with its own
  internal warmup and the harness takes the median of its per-repetition
  `samples_ts`. Five runs and a median, on both sides, from the same harness.
- **The warmup must be in-process, and this is not a detail.** CUT compiles
  kernels per process — Vulkan's pipeline cache is in-memory only and the CUDA
  backend goes through NVRTC at runtime — so relaunching the binary N times
  leaves every run cold and a cross-process "warmup" discards nothing. That is
  why `--bench-runs` exists in the runner at all rather than in this script.
- **The clock is the host's, on both sides.** Unlike the vendor suite there is
  no GPU-event pair here: a forward pass is hundreds of dispatches and what a
  caller waits for is wall time. CUT times prefill and decode around its own
  submission-and-synchronise; llama-bench times its own. Both include their
  host-side scheduling, which at this scale is part of the product rather than
  measurement overhead.
- **Peak VRAM is sampled from outside.** Neither runner reports it, so the
  harness polls `nvidia-smi` while the child runs and keeps the peak —
  per-process where the driver attributes it, falling back to the rise in
  whole-GPU used memory, which is only meaningful when nothing else is
  allocating. NVIDIA only; other vendors report nothing rather than a guess.
  **Free the card first**: a resident Ollama model will make CUT OOM or report
  nonsense, and the VRAM column will describe the wrong process.

### Correctness is a gate, and it is a weaker one than the vendor suite's

Once per model and backend, outside every timed region, CUT generates real text
from a real prompt — sampling, argmax readback, detokenisation, no synthetic
mode — and the run is judged on what came out. No tokens, tokens that
detokenise to nothing, or fewer than four distinct token ids fails the case,
marks every timing for that model and backend, and makes the process exit 2.
The diversity rule is only applied to runs of at least eight tokens: a model
that answers in three and stops has demonstrated a working EOS, while a
collapsed one never stops and so is always judged at full length.
`model_compare.py` then prints `VOID` instead of a speedup, exactly as
`vendor_compare.py` does for a case that disagreed with cuBLAS.

**This detects collapse, not drift.** The vendor suite can hold CUT's output
next to the reference's element by element; this cannot, and the reason is
worth writing down rather than leaving as an apparent oversight:

- llama-bench exposes no logits and no generated text — it is a stopwatch.
- The non-interactive completion binary that could produce a comparable greedy
  continuation is not built in the vendored llama.cpp here; `llama-cli` has
  dropped `--no-conversation` in favour of a separate `llama-completion`.
- Even with it, the two runners apply different chat templates to the same
  prompt (llama.cpp reads the GGUF's, CUT wraps ChatML when the tokens exist),
  so a token-level disagreement would usually be a fact about templating.

So the gate catches the failure that actually happens — the class where a
dequantisation or kernel bug leaves a model that loads cleanly, reports a
perfectly healthy tok/s, and emits one token forever — and it does not catch a
runner that is subtly less accurate. The generated text is recorded in the
snapshot so a human can read what the runner said rather than trust a boolean.
A greedy-continuation comparison against `llama-completion` is the obvious
upgrade when that binary is available.

### Reading the table

Cells are **reference over CUT**, so bigger is better and `1.00x` is parity —
the same convention as the operator page, and the same bands (🔵 ≥1x, 🟢 ≥0.95x,
🟡 ≥0.5x, 🟠 ≥0.2x, 🔴 below). Each family's figure is the geometric mean of its
cases, since these are ratios.

Two things the ratio does not say, both of which the absolute columns do:

- **A ratio has no scale.** 0.9x is 900 tok/s against 1000 or 9 against 10, and
  only one of those is worth an afternoon. `model_compare.py` prints both sides'
  tok/s next to the ratio, and the newest column on the page carries CUT's
  absolute rate under the cell for the same reason.
- **Prefill and decode are different machines.** Prefill is a GEMM over the
  whole prompt and saturates the card; decode is a GEMV per token and is bound
  by weight bandwidth and launch overhead. They are separate families on the
  page because a runtime can be strong at one and weak at the other, and their
  average is a number about neither.

## Adding a model or a shape

1. Drop the GGUF in `models/` and add its file name to `MODELS` in
   `scripts/bench/model_bench.py`, with a note saying what it covers that the
   set does not already — a model that adds no quantisation, architecture or
   size is minutes per run for no new information.
2. Check CUT can actually load it. The runner assumes a LLaMA-shaped block, and
   the dequantiser handles F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K —
   Q2_K, Q3_K and the IQ types throw on load.
3. Run only the new model (`./scripts/bench/model_bench.sh models/<file>.gguf`)
   and record it with `render_model_bench.py --snapshot --merge-latest`, which
   folds it into the existing column instead of starting one whose other cells
   would be copied from an older run and read as fresh measurements.
4. A new shape is `--tier ppN,tgM` for a one-off, or an entry in `TIERS` if it
   earns a permanent place. Prefer llama-bench's own shapes: a number nobody
   else measures cannot be compared with anything.
