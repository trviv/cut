# Vendor comparison benchmarks

Measures how CUT's operators stack up against the highly-tuned libraries the GPU
vendors ship — cuBLAS/cuDNN/cuVS on NVIDIA, rocBLAS/MIOpen on AMD. The point is
a defensible number for each crucial compute operation: *how far off vendor-peak
is CUT, on this shape, on this GPU?*

This is deliberately separate from `../autotune/`, which compares CUT's own
shader variants against each other to build a dispatch table. Here the reference
is external and the question is absolute competitiveness, not relative variant
choice.

## Layout

```
vendor/
├── common/VendorBench.h          Backend-neutral harness over Google Benchmark
├── cuda/
│   ├── CudaBenchCommon.h         CUDA-only helpers: event timing, VRAM budget, sampling
│   ├── cublas_matmul_bench.cpp   f32 GEMM/GEMV vs cuBLAS
│   ├── cublas_extras_bench.cpp   f16 GEMM vs cublasGemmEx, transpose vs cublasSgeam
│   ├── cudnn_softmax_bench.cpp   softmax vs cudnnSoftmaxForward
│   ├── cudnn_conv_bench.cpp      conv2d vs cudnnConvolutionForward
│   ├── cub_scan_sort_bench.cpp   scan + radix sort vs CUB
│   └── cub_wrappers.cu           CUB behind a C ABI (needs nvcc)
└── amd/
    ├── rocblas_matmul_bench.cpp     f32 GEMM/GEMV vs rocBLAS
    ├── rocblas_wrappers.hip         rocBLAS + all HIP calls behind a C ABI
    ├── rocprim_scan_sort_bench.cpp  scan + radix sort vs rocPRIM
    └── rocprim_wrappers.hip         rocPRIM + all HIP calls behind a C ABI
```

`common/VendorBench.h` must stay free of CUDA and HIP headers so both sides can
include it. It wraps Google Benchmark: `registerPair` registers the two halves of
one comparison, and the vendor half is handed in as a `TimedFn` — a
`std::function<double()>` that runs one launch and returns its GPU milliseconds.
That indirection is what keeps the header vendor-neutral; the HIP-event
`hipTimed` factory lives in the executables that use it, and the CUDA-event
`cudaTimed` in `cuda/CudaBenchCommon.h` alongside the other CUDA-only helpers
the model-scale cases need.

Google Benchmark supplies `benchmark::benchmark`, found with `find_package` and
fetched from GitHub (v1.9.1) if absent — the same pattern `tests/` uses for Google
Test.

CUB and rocPRIM are header-only *device* template libraries: only nvcc/hipcc can
compile them, while the rest of the suite is host C++. Each is therefore
confined to one wrapper translation unit exposing a plain `extern "C"` ABI, so
no benchmark `.cpp` needs a device-compiler dialect. The AMD wrapper goes
further and hides the entire HIP surface (allocation, copies, events), so
`rocprim_scan_sort_bench.cpp` builds with a plain host compiler.

## Building and running

Vendor targets are optional and self-guarding — if the SDK is missing, the
target is skipped and the default build is unaffected. Configure prints which:

```
-- vendor/cuda: cuBLAS /path/to/libcublas.so.13
-- vendor/amd: skipped (rocBLAS not found)
```

### Everything at once

```bash
cmake -B build-cuda-rel -DENABLE_CUDA_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
./scripts/bench/vendor_bench.sh
```

That builds every vendor target, runs each binary that exists, and prints one
combined comparison table across all of them. Targets whose SDK was not found are
skipped rather than treated as failures, so on an NVIDIA machine the two AMD
benches simply report `skipped (not built)`. A benchmark that fails is reported and
the rest still run; the script exits nonzero if any did.

Useful flags: `--quick` (skip the two largest `sort_radix` sizes — see below),
`--no-large` (skip the model-scale cases — see below), `--repetitions N`
(default 3), `--filter REGEX`, `--min-time T`, `--no-build`, `--build-dir DIR`,
`--out-dir DIR`. Per-benchmark JSON lands in `--out-dir` (default
`vendor_bench_results/`).

### The model-scale cases

Alongside the round-number shapes, each bench carries a tier of cases lifted
from real networks at real batch sizes: `hgemm_large`, `transpose_large`,
`softmax_large`, and every case in `cudnn_conv_bench`. They are named after
where they come from — `llama3-8b-ffn-up`, `flux-dit-mlp`,
`sd-vae-decoder-128ch-2048px` — because "M=32768 K=4096 N=28672" is a fact
about a matrix and "the Llama-3-8B fused gate/up projection over a 32k-token
batch" is a fact about a workload.

The models they come from are the 10-20 GB class: Llama-3-8B and Llama-2-13B,
Qwen2.5-14B, FLUX.1, SD3.5-Large, ViT-g/14, the Stable Diffusion VAE. The cases
themselves hold **2-20 GB of VRAM**, which is the point — an operator that
looks fine on a 4096² buffer is being asked here whether it still holds up when
nothing fits in any cache.

Three consequences worth knowing before running them:

- **They are slow.** A single `hgemm_large` iteration runs for 0.6 s at the
  largest shape and a `conv2d` iteration for 2.5 s, so the full model-scale
  tier is minutes rather than seconds. `--no-large` drops it.
- **Only one is resident at a time.** Operands are built on first use and freed
  when the next case needs the memory (`registerPairLazy`), so a binary that
  registers a dozen 10-GB cases still only ever holds one. Google Benchmark runs
  benchmarks in registration order and both halves of a pair are adjacent, so
  the CUT half builds the case, the vendor half reuses it, and repetitions reuse
  it too — a multi-GB upload happens once per case, not once per repetition.
- **Cases that do not fit are skipped, loudly.** Each one declares its footprint
  and is checked against what is actually free at startup, less headroom. On a
  smaller card the table is shorter and stderr says exactly which cases went and
  why. `CUT_VENDOR_BENCH_VRAM_GB=8` forces a budget, which is how to reproduce
  another card's selection — or to check that the skip path still works without
  finding a smaller GPU.

```bash
# Just the model-scale GEMMs, with aggregates
./build-cuda-rel/benchmarks/vendor/cuda/cublas_extras_bench \
    --benchmark_filter='hgemm_large' --benchmark_repetitions=3

# Everything except the model-scale tier
./scripts/bench/vendor_bench.sh --no-large
```

**A full run is slow, and it is CUT's fault.** The multi-pass `sortRadix` takes
783 ms per call at N=1M and ~13 s at N=16M, against CUB's 0.14 ms and 1.48 ms, so
those two cases dominate everything else combined. `--quick` excludes them and
brings a full sweep down to a couple of minutes; the other 43 comparisons are
unaffected.

### One benchmark at a time

```bash
cmake --build build-cuda-rel --target \
    cublas_matmul_bench cublas_extras_bench cudnn_softmax_bench cub_scan_sort_bench
./build-cuda-rel/benchmarks/vendor/cuda/cublas_matmul_bench \
    [--benchmark_repetitions=N] [--benchmark_filter=REGEX] \
    [--benchmark_out=PATH --benchmark_out_format=json]
```

Every bench is a standard Google Benchmark binary and takes the full flag set —
`--benchmark_list_tests`, `--benchmark_min_time`, `--benchmark_report_aggregates_only`,
and the rest. The three flags above are the ones that matter here:

- `--benchmark_repetitions=5` turns on the `_mean` / `_median` / `_stddev` / `_cv`
  aggregate rows. Without it you get a single sample and no idea how noisy it was.
- `--benchmark_filter` selects a subset. Every case is registered twice, as
  `cut/<op>/<shape>` and `<vendor>/<op>/<shape>`, so `--benchmark_filter='^cut/'`
  runs only the CUT half and `--benchmark_filter='sgemv'` only the decode shapes.
- `--benchmark_out` writes JSON, which is what the comparison script below reads.

Google Benchmark prints each benchmark on its own line and has no notion that two
of them are two halves of one comparison. To get the side-by-side table with the
speedup and correctness columns, run the JSON through:

```bash
./build-cuda-rel/benchmarks/vendor/cuda/cublas_matmul_bench \
    --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
    --benchmark_out=sgemm.json --benchmark_out_format=json
python3 scripts/bench/vendor_compare.py sgemm.json
```

```
op        shape                 vendor  cut_ms  ref_ms  cut_rate  ref_rate  unit    speedup  cv     ref_mag  max_diff
transpose M=4096_N=4096         cuBLAS  0.165   0.165   811.16    813.95    GB/s    1.00x    0.02%  0.500    0.00e+00
hgemm     M=1_K=8192_N=8192     cuBLAS  0.963   0.158   139.41    851.76    GFLOPS  0.16x    0.13%  24.134   8.37e+13
```

The script accepts several JSON files at once, `--sort {speedup,op,shape}`, and
`--csv`. Google Benchmark also ships its own `tools/compare.py`, which can diff two
filters (`compare.py filters ./bench '^cut/' '^cuBLAS/'`) and adds a U-test for
statistical significance — better for "did my change move the needle", but it does
not carry the correctness columns.

`vendor/cuda` needs a full CUDA toolkit include tree, not just the pip
`nvidia-cu13` wheel — the wheel is runtime-only and ships no `crt/` headers,
which `cuda_runtime.h` requires. The build picks headers from the toolkit and
libraries from the pip tree (which `CUTLib` already puts on the rpath; linking
`libcublas.so.13` from a second tree that also provides it makes CMake's rpath
ordering unsatisfiable).

cuDNN ships separately from the toolkit and is searched for independently, so
`cudnn_softmax_bench` skips on its own without affecting the cuBLAS targets. The
pip `nvidia-cudnn` wheel works and is what the recorded numbers were measured
against. `libcudnn.so.9` `dlopen`s its `libcudnn_*.so.9` sub-libraries from its
own directory at runtime, so that directory goes on the target's rpath too.

## Methodology

Getting this comparison honest matters more than getting it favourable.

- **Both sides are timed on the GPU.** CUT via
  `Runtime::lastDispatchTimings()` (hardware timestamps), the vendor via CUDA
  events. Neither number includes host-side submit/wait or tensor setup. Every
  benchmark therefore runs under Google Benchmark's `UseManualTime()`: the
  library owns the loop and the statistics, but the clock stays on the GPU.
  Without it Google Benchmark would report host wall-clock, which for an
  asynchronous GPU submit measures little more than the cost of enqueueing.
- **Same process, same context.** CUT creates its own CUDA driver context and
  makes it current, so the CUDA runtime API and cuBLAS bind to that same
  context. Both sides run on the same device with the same clocks.
- **Iteration count is adaptive, and the median is reported** across
  `--benchmark_repetitions`. Each benchmark body also does three untimed calls
  first, which absorb CUT's one-time kernel compilation — Google Benchmark's
  iteration-count estimation has no way to know that first call should be
  discarded. The model-scale cases turn that down to one warmup and pin their
  iteration count: at 0.6 s a call, three warmups cost more than the
  measurement, and each issue allocates a multi-GB output tensor whose host-side
  cost the manual-time loop cannot see.
- **Setup is hoisted out of the timed region.** Operands are uploaded once at
  registration and the timed callable is the dispatch alone. This matters twice
  over: the vendor side uploads its operands once, so leaving CUT to re-upload
  per iteration would tilt the comparison; and because the upload is invisible to
  the GPU timestamp but not to the wall clock, the adaptive loop would keep
  iterating to accumulate manual time and never finish. Measured before the fix:
  a 34 us GEMV carried 3152 us of hidden per-iteration setup and ran 20129
  iterations.
- **CUT runs its default variant** — whatever the autotuned dispatch table
  picks. That is what a caller actually gets, so that is what gets compared.
- **No implicit precision advantage.** cuBLAS is set to `CUBLAS_DEFAULT_MATH`
  so it does not silently drop to TF32 against CUT's true f32 kernels. Any
  reduced-precision comparison should be its own explicitly-labelled case.
- **Correctness is checked once, outside every timed region, and it is a
  gate — not a column.** It runs at registration and rides along as the
  `max_diff`, `ref_mag` and `max_diff_allowed` counters on both halves of the
  pair, so a JSON consumer never has to join two sources to find out whether a
  fast number was also a correct one. `ref_mag` is the mean magnitude of the
  reference output: without it, two silently-empty buffers would agree
  perfectly. A zero `ref_mag`, or a `max_diff` over the case's tolerance, fails
  the case via `SkipWithError`, marks **both** halves of the pair, and makes the
  binary exit **2**.

  This used to be advisory, and that was a real hole: the f16 GEMV at M=1
  returns garbage (`max_diff` 8.4e13 against a `ref_mag` of 24) and for a while
  it printed a GFLOPS figure and exited 0 like any passing case. A number that
  is fast and wrong is the one number a benchmark must never quietly report.
- **Above ~12M elements the check is sampled, not exhaustive.** The largest
  output in the suite is nearly two billion elements; holding both sides of it
  on the host as f32 would cost more memory than the GPU side of the whole
  benchmark. `planSample` reads a bounded prefix, middle and suffix from both
  sides at identical offsets instead. This is a real weakening — a kernel that
  is wrong only in the unsampled region would pass — but a GEMM that is wrong is
  essentially never wrong only there, and the alternative was not checking the
  large cases at all. Small cases are still compared element for element.
- **A case declares what it holds.** `CaseSpec::footprintBytes` is reported as
  the `vram_gb` counter and is what the budget check gates on. It is not always
  a sum of the obvious buffers: CUT's `softmax` is a composition that
  materialises the broadcast max, the shifted input and the exponentials at full
  size, so the softmax cases charge eight times their element count and not
  four. That multiplier is measured against `nvidia-smi`, not counted off the
  source — a case sized as though CUT needed only an input and an output aborted
  the run at 21 GB resident.
- **The rate column adapts to the operator.** Set `CaseSpec::flops` for
  compute-bound ops and `CaseSpec::bytes` for memory-bound ones; the counter then
  reads `FLOPS` or `bytes_per_second`. Scan and sort are bandwidth-bound, so
  quoting GFLOPS for them would be meaningless. Note the units differ between the
  two views: Google Benchmark's console prints `bytes_per_second` with 1024-based
  `Gi/s` suffixes, while `vendor_compare.py` reports decimal GB/s (1e9).
- **Destructive operators pin their iteration count.** CUT's `sortRadix` family
  sorts in place, so each iteration must re-upload its input or it would re-sort
  sorted data and report a misleadingly fast number. That refill is exactly the
  hidden per-iteration cost the adaptive loop cannot see, so those cases set
  `CaseSpec::iterations` to a small fixed number and lean on
  `--benchmark_repetitions` for stability instead.

### Tolerances

Every case declares the largest `max_diff` it accepts, via `CaseSpec::tolerance`.
A case passes when `max_diff <= max(absolute, relative * ref_mag)`.

These are **garbage detectors, not precision certificates**. The gap between real
rounding error and a broken kernel is many orders of magnitude — f16 rounding
lands at ~2e-3 relative, the broken GEMV at ~1e12 — so each bound is set well
above measured error. A loose gate that is always on beats a tight one that gets
switched off the first time it cries wolf.

| Operator | Tolerance | Measured worst case | Why |
|---|---|---|---|
| f32 GEMM/GEMV | `rel(1e-4)` | 0 | Different accumulation order than cuBLAS; in practice bit-exact here |
| f16 GEMM (both tiers) | `rel(1e-2)` | ~2e-3 | Half carries ~3 decimal digits |
| Transpose (both tiers) | `exact()` | 0 | Moves values, computes none — no rounding to allow for |
| Softmax (both tiers) | `rel(1e-4)` | ~4e-6 | Relative, since a 152064-wide row produces values around 1/152064 |
| Conv2D | `rel(1e-3)` | ~3.3e-5 | Accumulates `C*k*k` products in a different order than cuDNN |
| Prefix scan | `rel(1e-3)` | ~9e-6 | Parallel reordering drift, which grows with N |
| Radix sort | `exact()` | 0 | `max_diff` is a mismatch **count**, not a float delta |

The default for a case that sets nothing is `rel(1e-2)` — deliberately a gate
rather than an exemption, so a new benchmark is checked by default and its author
tightens it. `Tolerance::reportOnly()` disables the gate and should carry a
comment saying why; an ungated case is a case that cannot fail.

Two per-operator notes worth repeating when quoting numbers:

- **Sort `max_diff` is a mismatch count, not a float delta.** Sorted uint32 keys
  must agree with the vendor exactly, hence `exact()`.
- **f16 GEMM `max_diff` has to be read against `ref_mag`.** Half carries ~3
  decimal digits, so a diff a few thousandths of `ref_mag` is ordinary rounding
  where the same figure would be alarming in the f32 table.

### What a failure looks like

The binary names the failing comparisons and exits 2:

```
cut/hgemm/M=1_K=4096_N=4096/manual_time  ERROR OCCURRED: 'INCORRECT: CUT disagrees
  with the reference — max_diff 1.198e+13 exceeds the 1.735e-01 allowed against ref_mag 1.735e+01'

=== CORRECTNESS FAILURES (2 comparison(s)) ===
  hgemm/M=1_K=4096_N=4096
  hgemm/M=1_K=8192_N=8192
```

`vendor_compare.py` adds a `status` column, prints the failures to stderr, and
**withholds the speedup** — a failed row reads `VOID`, not a ratio, so there is
no number on the line to quote by accident:

```
status | op    | shape              | ... | speedup | ref_mag | max_diff
ok     | hgemm | M=16_K=4096_N=4096 | ... | 0.41x   | 16.997  | 3.10e-02
FAIL   | hgemm | M=1_K=4096_N=4096  | ... | VOID    | 0       | 0.00e+00
```

(A failed row's counters read 0 because Google Benchmark drops every user counter
from an errored row — the real numbers are in the failure block above the table,
which reads them from `error_message`.)

Exit codes are **0** clean, **1** a benchmark failed to run, **2** a comparison
failed its correctness gate. `vendor_bench.sh` propagates the same three, and
correctness outranks run failure: a benchmark that did not run is a gap, but one
that ran fast and wrong is a wrong answer. Note that a binary exiting 2 still has
its JSON compared — dropping it would hide exactly the failures the gate exists
to surface.

Note on layout: CUT is row-major (`C[M,N] = A[M,K] * B[K,N]`), cuBLAS is
column-major. The benches compute `C^T = B^T * A^T`, which yields the row-major
result with no transposes and no extra copies — so the reference is not charged
for a layout conversion CUT never performs.

## Current coverage

| Operation | NVIDIA reference | AMD reference | Status |
|---|---|---|---|
| f32 GEMM / GEMV | cuBLAS `cublasSgemm` | rocBLAS `rocblas_sgemm` | both written |
| f16 GEMM / GEMV | cuBLAS `cublasGemmEx` | rocBLAS `rocblas_hgemm` | NVIDIA done; AMD not yet |
| Transpose | cuBLAS `cublasSgeam` | rocBLAS `rocblas_sgeam` | NVIDIA done; AMD not yet |
| Softmax | cuDNN `cudnnSoftmaxForward` | MIOpen `miopenSoftmaxForward` | NVIDIA done; AMD not yet |
| Conv2D | cuDNN `cudnnConvolutionForward` | MIOpen `miopenConvolutionForward` | NVIDIA done; AMD not yet |
| Prefix scan (incl/excl) | CUB `DeviceScan` | rocPRIM `inclusive_scan` / `exclusive_scan` | both written |
| Radix sort (key+value) | CUB `DeviceRadixSort::SortPairs` | rocPRIM `radix_sort_pairs` | both written |
| RMSNorm / LayerNorm | — | — | no legacy-API vendor equivalent; cuDNN v9 exposes normalization only through the graph API, so this needs a hand-written bandwidth-peak reference rather than a library call |
| Quantized matmul (Q4/Q8) | — | — | no direct vendor equivalent; compare against dequant + SGEMM |
| KNN / brute-force search | cuVS (RAFT) or FAISS | — | not yet — CUT has no KNN operator; it would be composed from matmul + topk over `impl/sort` |

**The AMD sources have never been compiled or run** — this machine has no ROCm
install. `rocprim_scan_sort_bench.cpp` and `rocblas_matmul_bench.cpp` are
syntax-checked with a host compiler (both deliberately include no HIP header),
but `rocprim_wrappers.hip` and `rocblas_wrappers.hip` have not been through
hipcc, and the rocPRIM and rocBLAS call signatures are written from the API
docs, not verified against a build. Expect to fix something the first time you
build them on real hardware.

## Findings

Measured on an RTX 3090, CUDA 13.0 (cuDNN 9.20), median across
`--benchmark_repetitions=3`. Recorded here so regressions are visible; re-run
rather than trusting these numbers indefinitely. Every figure below is from
`vendor/cuda` — nothing in `vendor/amd` has been run.

Measurement noise on this machine is low: `cv` is under 1% for almost every case
(a couple of the smallest shapes reach ~2.7%), so differences of more than a few
percent below are real. Note that CPU scaling is enabled here, which Google
Benchmark warns about on every run; it affects the wall-clock column, not the
GPU-timestamp column these numbers come from.

- **f32 GEMM sits at ~0.5–0.6x cuBLAS** on square shapes (0.59x at 1024³, 0.61x
  at 4096³, 0.47x at 512³), rising to 0.92x at the tiny 128³ where neither side
  fills the GPU.
- **The skinny shapes are where CUT actually loses**: 0.37x at M=16 K=N=4096 and
  0.36x at the M=1 K=N=8192 GEMV. The mid-size GEMVs are much healthier than
  previously recorded — 0.67x and 0.70x at K=N=2048 and 4096.

  **This corrects an earlier finding.** The pre-Google-Benchmark harness reported
  0.33–0.37x across all GEMV shapes and called it "the widest gap and the one that
  matters most for LLM decode". That was a measurement artifact: the old timed
  lambda called `createTensor` on every iteration, so `lastDispatchTimings()`
  summed the operand upload — 16 MB of B matrix at K=N=2048 — into the GEMV's
  time. With the upload hoisted to registration, two of the three GEMV shapes
  roughly double. The gap at K=N=8192 is real and still the worst GEMV case.
- **f16 GEMM is uniformly worse than f32 relative to cuBLAS: 0.31–0.41x** on
  every shape, dropping to 0.13x at the 8192 GEMV (0.17x before the GEMV
  correctness fix below, which cost some speed to buy a right answer). Both
  sides accumulate in f32
  (`CUBLAS_COMPUTE_32F`), so this is not a precision trade — cuBLAS reaches
  71.8 TFLOPS at 4096³ against CUT's 27.3, i.e. cuBLAS is using tensor cores
  and CUT is not.
- **f16 GEMV (M=1) was numerically WRONG on the CUDA backend — FIXED.**
  `max_diff` was 1.2e13 at M=1 K=4096 N=4096 and 8.4e13 at M=1 K=8192 N=8192
  against a `ref_mag` of ~17 and ~24: not near-misses but garbage (2.9e3,
  -2.0e9, 3.3e10, …).

  The cause was a dtype mismatch in kernel selection, not arithmetic.
  `MatMulOpNode::outputDtype()` is always `Float32`, but the kernel lookup asked
  for an output dtype matching the *operand* dtype — so Float16 A and B selected
  the Float16-output kernel, which writes 2 bytes per element into the 4-byte
  Float32 tensor. Readers saw pairs of halves reinterpreted as floats, which is
  exactly the kind of garbage observed.

  It hid because `shouldUseCoopMat` requires `M % 16 == 0 && M >= 16`: every
  aligned M took the cooperative-matrix path, which always asked for Float32.
  Only M=1 reached the scalar path — and so did *any* f16 matmul on a device
  without coopmat, which is the broader half of the bug. Every kernel lookup now
  passes `outputDtype()`, and the f16/f16 case resolves to the
  Float32-A/Float16-B kernel: the activation is cast (one row at M=1, ~9 us) and
  the weight matrix stays f16. Covered by
  `MatrixOpsTest.MatMul_F16xF16_WritesFloat32Output`, which tests M=1, 3 and 15
  against the coopmat M=16 — a test at aligned M alone would have passed against
  the bug.

  Accuracy is now better than the f32 GEMM's at the same shapes (the accumulator
  went from half to f32): `max_diff` 0.031 against a tolerance of 0.173. The
  measured time went from 955 us to 1256 us at K=N=8192, but the old figure
  timed a kernel that produced garbage — 1256 us is the first valid measurement,
  and only 9 us of it is the added cast.
- **Transpose is at parity with cuBLAS: 0.98–1.07x**, and bit-exact (`max_diff`
  is 0.00e+00 on every shape). Both sides run at 740–830 GB/s, which is ~85–95%
  of the 3090's 936 GB/s peak, so the operator is saturated and there is nothing
  left to win here. Aspect ratio does not move it — tall, wide and square all
  land within a few percent.
- **Softmax is 0.01–0.04x cuDNN**, the second-largest gap in the suite after
  prefix scan. On many-short-rows shapes CUT holds 24–33 GB/s against cuDNN's
  712–819 GB/s (which is at bandwidth peak). The few-very-long-rows shapes are
  worse in absolute terms — 0.24 GB/s at rows=1 cols=152064, a 5.1 ms softmax
  over a single vocabulary row — but cuDNN is also far off peak there (8.6
  GB/s), since one row cannot fill the GPU. The ratio, not the rate, is the
  signal: CUT is ~25-30x slower on shapes where the vendor is saturated.
  Numerically CUT is fine throughout (`max_diff` 1e-08 to 1e-11).
- **Prefix scan is 0.04–0.08x CUB** at large N — CUT holds ~35 GB/s against
  CUB's ~817 GB/s at N=16M, i.e. roughly 4% of achievable bandwidth. This is
  the largest relative gap anywhere in the suite, and it widens with N: 0.36x at
  N=65K, 0.08x at N=1M, 0.05x at N=4M, 0.04x at N=16M. CUT's own throughput is
  roughly flat at 35–45 GB/s across all four sizes while CUB climbs from 77 to
  817 GB/s, so CUT is not scaling with the problem at all. Unaffected by the
  hoisting correction above — these figures match the pre-Google-Benchmark
  harness almost exactly.
- **`sortRadix` (the multi-pass path) is pathologically slow on CUDA**: 13.2 s
  at N=16M versus CUB's 1.48 ms, and 35 ms at N=65K. It scales linearly at a
  flat ~0.02 GB/s, so it is a constant-factor problem, not a complexity one.
  Independently confirmed against the pre-existing `benchmarks/benchmark.cpp`
  sort bench (32.6 ms vs this suite's 32.2 ms at N=65K), so it is a real CUT
  issue rather than a harness artifact.
- **`sortRadixSinglePass` and `sortRadixOneSweep` are healthy** — 0.36–0.44x of
  CUB at N >= 1M, and bit-exact against it at every size. They track each other
  to within a fraction of a percent. The small-N end is better than the large-N
  end: 0.75x at N=65K, 0.44x at N=1M, 0.39x at N=4M, 0.36x at N=16M. Only the
  N=65K figure moved in the Google Benchmark rewrite (it read 0.36–0.44x before);
  the old harness created the key and value tensors *inside* its timed lambda, so
  `lastDispatchTimings()` summed the upload dispatches into the sort's time, and
  that overhead dominated only at the smallest size. The rewrite creates the
  tensors once and refills them via `copyToTensor`, leaving the sort kernels
  alone in the measurement.

### Model-scale findings

Same machine and method, median across `--benchmark_repetitions=3`. `cv` is
under 1% on every case in this tier — well under the differences below — so
these ratios are solid despite each case running only three iterations.

- **f16 GEMM lands at 0.32-0.35x cuBLAS on every model shape**, without
  exception across eleven of them: 25-27.6 TFLOPS against cuBLAS's 76-79.4.
  This is the same gap the round-number shapes show at 4096³ (0.38x) and the
  same cause — cuBLAS is on tensor cores, CUT is not — but the model shapes make
  it much better behaved. Where the small table swings from 0.17x to 0.41x
  depending on shape, every real projection here sits in a three-point band,
  from a 32k x 4096 x 6144 fused QKV to a 65536 x 4096 x 28672 FFN forty times
  its size. Nothing about being large, skinny, or vocabulary-wide moves it.
  Aspect ratio does not matter, and neither does the 16.6 GB working set of the
  largest case: **CUT's f16 GEMM is at a flat third of cuBLAS and the missing
  factor of three is tensor cores.**
- **Transpose stays at parity right up to a 17 GB working set**: 0.98x at 8k²
  through 32k x 16k, and 1.00x at 32768², where both sides reach 821 GB/s — 88%
  of the 3090's 936 GB/s peak. Bit-exact throughout. This is the one operator
  that gains nothing from being given a bigger problem, because it was already
  saturated.
- **Softmax is 0.07-0.11x cuDNN at model scale**, against 0.01-0.04x on the
  small shapes — and CUT did not improve. CUT holds 23-33 GB/s here, the same
  21-33 GB/s it manages on the small shapes; what changed is cuDNN, which falls
  from 712-819 GB/s to 293-333 GB/s once the buffers stop fitting in cache. Read
  the absolute rate, not the ratio: CUT is ~4% of achievable bandwidth on an
  operator that is pure bandwidth, at every size tried.
- **Conv2D is 0.03-0.04x cuDNN across every vision shape** — patch embedding,
  U-Net 3x3, VAE decoder, ResNet block, 1x1 projection. CUT holds 0.63-1.31
  TFLOPS against cuDNN's 19.6-41.3, a 25-33x gap that is remarkably insensitive
  to shape. The 1x1 projection is the worst case at 629 GFLOPS, which is telling:
  a 1x1 convolution *is* a GEMM, and CUT's own f32 matmul does 15.4 TFLOPS at
  4096³ — 24 times faster on the same arithmetic at the same precision. The gap
  is the convolution path, not the hardware. Numerically CUT is fine
  (`max_diff` 0 to 9.3e-04 against `ref_mag` 4-28).
- **CUT's conv2d cannot dispatch past 1,048,560 output rows** (N x C_out x
  H_out). The default variant linearises those three onto the grid's y axis in
  blocks of 16, and CUDA caps `gridDim.y` at 65535, so crossing it is a
  `CUDA_ERROR_INVALID_VALUE` from `cuLaunchKernel` — not a fallback, not a
  slower path. This binds long before memory does on the shapes vision models
  actually use: a ViT-L/14 patch embedding at batch 512 wants 8.4M rows, eight
  times the limit, and even batch 64 is sixteen rows over. Every conv case here
  is sized under the ceiling and `cudnn_conv_bench` skips anything above it with
  a message rather than crashing, but the fix belongs in `Conv2DOpNode` —
  folding the batch into a second grid axis, or tiling the y axis.

## Adding an operator

1. Add a source next to its vendor's siblings (`cuda/`, `amd/`).
2. On CUDA, include `CudaBenchCommon.h` for `cudaTimed` and the model-scale
   helpers; on HIP, copy `hipTimed` from a sibling. Either wraps one vendor
   launch in an event pair and returns its GPU milliseconds.
3. Write one `register<Op>Case` per shape. Upload both sides' operands **once**,
   before any timing, and make the CUT issue lambda the dispatch alone. Capture
   `Runtime` by reference and everything else by value: registration happens in
   `main` and the lambdas outlive that scope.
4. Run the correctness check there too, outside any timed region, and store the
   `compareBuffers` result in `CaseSpec::check`.
5. Set `CaseSpec::tolerance`. Run the case once, look at the `max_diff` it
   actually produces against its `ref_mag`, and set the bound an order of
   magnitude above that — see *Tolerances*. Leaving it unset gets the `rel(1e-2)`
   default, which catches garbage but is looser than most operators deserve.
6. Fill the rest of the `CaseSpec`: `flops` for a compute-bound op or `bytes` for
   a memory-bound one — set exactly one, it picks the counter — then call
   `registerPair`.
7. End `main` with `const int rc = cutbench::runAll(argc, argv);` followed by
   `runtime.shutdown();` and `return rc;`. That order is required: `runAll`
   clears the benchmark registry so the captured `cut::Tensor` handles are
   released while the runtime is still up, and letting the `Runtime` destructor
   run at end of `main` instead of calling `shutdown()` segfaults. Returning
   `rc` is what carries a correctness failure into the exit status.
8. Device buffers, vendor handles and descriptors are deliberately leaked — they
   must outlive every registered lambda, and the process exits immediately after.
9. Register the target in that directory's `CMakeLists.txt`, linking
   `benchmark::benchmark`, guarded by a `find_*` so a missing SDK skips rather
   than breaks the build.

### Adding a model-scale case

Same shape of code, four differences:

1. Call `registerPairLazy` with a factory instead of `registerPair` with
   operands. The factory allocates, uploads, runs the correctness check, and
   returns a `LazyCase`; it is called on first use and its `release` runs when
   the next lazy case needs the memory. Install `release` as soon as the device
   pointers exist, so an early return still frees them.
2. Allocate with `tryDeviceAlloc`, which returns null instead of aborting, and
   return an empty `shared_ptr` if it does — the case is then skipped rather
   than taking the run down with it.
3. Set `footprintBytes` to everything the case holds **on both sides**, and gate
   registration on `fitsVramBudget`. Measure the number rather than deriving it
   from the obvious buffers if the CUT op is a composition: check the case
   against `nvidia-smi` once and use what you see.
4. Set `warmupIterations = 1` and pin `iterations`, and use `planSample` /
   `sampleTensor` / `sampleDeviceFloats` for the correctness check rather than
   reading whole buffers back.
