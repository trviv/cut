# Vendor comparison benchmarks

How far off vendor-peak is CUT, on this shape, on this GPU? The references are
the tuned libraries the GPU vendors ship — cuBLAS/cuDNN/CUB on NVIDIA,
rocBLAS/rocPRIM on AMD.

Separate from `../autotune/`, which compares CUT's own shader variants against
each other to build a dispatch table. Here the reference is external and the
question is absolute competitiveness, not relative variant choice.

Measured results are not kept here — run the suite and read the table it prints.

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
include it. `registerPair` registers the two halves of one comparison, and the
vendor half is handed in as a `TimedFn` — a `std::function<double()>` that runs
one launch and returns its GPU milliseconds. That indirection is what keeps the
header vendor-neutral.

CUB and rocPRIM are header-only *device* template libraries: only nvcc/hipcc can
compile them, while the rest of the suite is host C++. Each is confined to one
wrapper translation unit exposing a plain `extern "C"` ABI. The AMD wrapper goes
further and hides the entire HIP surface, so `rocprim_scan_sort_bench.cpp` builds
with a plain host compiler.

**The AMD sources have never been compiled or run** — no ROCm on this machine.
The `.cpp` files are syntax-checked with a host compiler, but the `.hip` wrappers
have not been through hipcc and the call signatures are written from the API
docs. Expect to fix something the first time you build them on real hardware.

## Building and running

Vendor targets are optional and self-guarding — if the SDK is missing, the
target is skipped and the default build is unaffected. Configure prints which:

```
-- vendor/cuda: cuBLAS /path/to/libcublas.so.13
-- vendor/amd: skipped (rocBLAS not found)
```

### Everything at once

```bash
./scripts/bench/vendor_bench.sh
```

That is the whole command — it configures the build directory if missing, builds
every vendor target this configuration defines, runs each binary, and prints one
combined comparison table.

It configures with `-DENABLE_CUDA_BACKEND=ON -DCMAKE_BUILD_TYPE=Release`. Vulkan
needs no flag (`find_package(Vulkan REQUIRED)` makes it mandatory), so CUDA is
the only opt-in backend. It also re-configures a directory built *without*
`ENABLE_CUDA_BACKEND`, since that state builds none of the `vendor/cuda` targets
and is otherwise indistinguishable from a missing SDK — an empty table with no
explanation. A benchmark that fails is reported and the rest still run.

Useful flags: `--quick` (skip the two largest `sort_radix` sizes), `--no-large`
(skip the model-scale cases), `--interleave` (remove the cut-then-vendor ordering
bias; implies `--no-large`), `--repetitions N` (default 3), `--filter REGEX`,
`--min-time T`, `--no-build`, `--build-dir DIR`, `--out-dir DIR`. Per-benchmark
JSON lands in `--out-dir` (default `vendor_bench_results/`).

**A full run is slow, and it is CUT's fault.** The multi-pass `sortRadix` takes
hundreds of milliseconds per call at N=1M and seconds at N=16M, against CUB's
sub-millisecond, so those two cases dominate everything else combined. `--quick`
excludes them and brings a full sweep down to a couple of minutes.

### The model-scale cases

Alongside the round-number shapes, each bench carries a tier lifted from real
networks at real batch sizes: `hgemm_large`, `transpose_large`, `softmax_large`,
and every case in `cudnn_conv_bench`. They are named after where they come from —
`llama3-8b-ffn-up`, `flux-dit-mlp` — because "M=32768 K=4096 N=28672" is a fact
about a matrix and "the Llama-3-8B fused gate/up projection over a 32k-token
batch" is a fact about a workload. The cases hold **2-20 GB of VRAM**, which is
the point: an operator that looks fine on a 4096² buffer is asked here whether it
still holds up when nothing fits in any cache.

Three consequences worth knowing before running them:

- **They are slow.** A single `hgemm_large` iteration runs for a good fraction of
  a second at the largest shape, so the tier is minutes rather than seconds.
  `--no-large` drops it.
- **Only one is resident at a time.** Operands are built on first use and freed
  when the next case needs the memory (`registerPairLazy`). Google Benchmark runs
  benchmarks in registration order and both halves of a pair are adjacent, so the
  CUT half builds the case, the vendor half reuses it, and repetitions reuse it
  too — a multi-GB upload happens once per case, not once per repetition. This is
  also why `--interleave` implies `--no-large`.
- **Cases that do not fit are skipped, loudly.** Each declares its footprint and
  is checked against what is actually free at startup, less headroom. On a
  smaller card the table is shorter and stderr says which cases went and why.
  `CUT_VENDOR_BENCH_VRAM_GB=8` forces a budget, which is how to reproduce another
  card's selection — or to check the skip path without finding a smaller GPU.

### One benchmark at a time

```bash
cmake -B build-cuda-rel -DENABLE_CUDA_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-rel --target \
    cublas_matmul_bench cublas_extras_bench cudnn_softmax_bench \
    cudnn_conv_bench cub_scan_sort_bench
./build-cuda-rel/benchmarks/vendor/cuda/cublas_matmul_bench \
    --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
    --benchmark_out=sgemm.json --benchmark_out_format=json
python3 scripts/bench/vendor_compare.py sgemm.json
```

Every bench is a standard Google Benchmark binary and takes the full flag set.
The three that matter:

- `--benchmark_repetitions=5` turns on the `_mean` / `_median` / `_stddev` /
  `_cv` aggregate rows. Without it you get one sample and no idea how noisy.
- `--benchmark_filter` selects a subset. Every case is registered twice, as
  `cut/<op>/<shape>` and `<vendor>/<op>/<shape>`, so `'^cut/'` runs only the CUT
  half and `'sgemv'` only the decode shapes.
- `--benchmark_out` writes the JSON `vendor_compare.py` reads.

Google Benchmark prints each benchmark on its own line and has no notion that two
of them are halves of one comparison; `vendor_compare.py` joins them into the
side-by-side table with the speedup and correctness columns:

```
op        shape                 vendor  cut_ms  ref_ms  cut_rate  ref_rate  unit    speedup  cv     ref_mag  max_diff
transpose M=4096_N=4096         cuBLAS  0.165   0.165   811.16    813.95    GB/s    1.00x    0.02%  0.500    0.00e+00
```

It accepts several JSON files at once, `--sort {speedup,op,shape}`, and `--csv`.
Google Benchmark also ships `tools/compare.py`, which can diff two filters and
adds a U-test — better for "did my change move the needle", but it does not carry
the correctness columns.

`vendor/cuda` needs a full CUDA toolkit include tree, not just the pip
`nvidia-cu13` wheel — the wheel is runtime-only and ships no `crt/` headers,
which `cuda_runtime.h` requires. The build picks headers from the toolkit and
libraries from the pip tree (linking `libcublas.so.13` from a second tree that
also provides it makes CMake's rpath ordering unsatisfiable).

cuDNN ships separately and is searched for independently, so `cudnn_softmax_bench`
skips on its own without affecting the cuBLAS targets. The pip `nvidia-cudnn`
wheel works; `libcudnn.so.9` `dlopen`s its sub-libraries from its own directory at
runtime, so that directory goes on the target's rpath too.

## Methodology

Getting this comparison honest matters more than getting it favourable.

### What is compared against what

| Operation | NVIDIA reference | AMD reference |
|---|---|---|
| f32 GEMM / GEMV | cuBLAS `cublasSgemm` | rocBLAS `rocblas_sgemm` |
| f16 GEMM / GEMV | cuBLAS `cublasGemmEx` | not yet |
| Transpose | cuBLAS `cublasSgeam` | not yet |
| Softmax | cuDNN `cudnnSoftmaxForward` | not yet |
| Conv2D | cuDNN `cudnnConvolutionForward` | not yet |
| Prefix scan (incl/excl) | CUB `DeviceScan` | rocPRIM `inclusive_scan` / `exclusive_scan` |
| Radix sort (key+value) | CUB `DeviceRadixSort::SortPairs` | rocPRIM `radix_sort_pairs` |

RMSNorm and LayerNorm have no legacy-API vendor equivalent (cuDNN v9 exposes
normalization only through the graph API), so they need a hand-written
bandwidth-peak reference rather than a library call. Quantized matmul has no
direct equivalent either; the comparison would be against dequant + SGEMM.

### Both sides run the same operator

- **Same result storage.** Same dtype, same element count, same
  in-place-or-not contract — otherwise the two halves are not running the same
  operator and the correctness gate measures the reference instead of CUT. This
  is easy to get wrong and was wrong once: the f16 GEMM asked cuBLAS for an f16
  `C` while `MatMulOpNode::outputDtype()` writes f32, so cuBLAS moved half the
  output bytes and every `max_diff` was dominated by the reference's own storage
  rounding (~2e-3 relative) rather than by CUT's arithmetic. With both writing
  f32, every aligned-M shape is **bit-exact** against cuBLAS and the worst case
  anywhere drops to 7.5e-5 relative.

  Where the APIs genuinely differ, check that the difference costs nothing rather
  than assuming it. CUT's radix sort is in place and CUB's cannot be, but both are
  fixed-destination sorts over four passes of 8-bit digits, and four is even — so
  each ping-pongs against a single alternate buffer and lands on its mandated side
  with no extra copy, at identical per-pass traffic. Neither is given CUB's
  cheaper `DoubleBuffer` contract, which permits finishing in whichever buffer the
  last pass reached.
- **Same math.** cuBLAS is set to `CUBLAS_DEFAULT_MATH` and cuDNN's convolutions
  to `CUDNN_FMA_MATH`, so neither silently drops to TF32 against CUT's true f32
  kernels. The f16 GEMM passes `CUBLAS_COMPUTE_32F`, matching the f32 accumulator
  CUT's coopmat path uses — both sides run f16 operands on tensor cores and
  accumulate in f32. Nothing is compiled with fast math: CUT's kernels go through
  NVRTC with no `--use_fast_math`, so IEEE division and square root and the
  default FMA contraction are in force on both sides. Any reduced-precision
  comparison should be its own explicitly-labelled case.
- **Same layout, no conversion charged to either side.** CUT is row-major
  (`C[M,N] = A[M,K] * B[K,N]`), cuBLAS column-major. The benches compute
  `C^T = B^T * A^T`, which yields the row-major result with no transposes and no
  extra copies. Conv2D configures cuDNN with the NCHW input and `[Cout, C, kH,
  kW]` weights CUT already takes, and with `CUDNN_CROSS_CORRELATION`, since CUT
  does not flip the kernel.
- **CUT runs its default variant** — whatever the autotuned dispatch table picks.
  That is what a caller actually gets, so that is what gets compared.
- **Same process, same context.** CUT creates its own CUDA driver context and
  makes it current, so the CUDA runtime API and cuBLAS bind to that same context.
  Both sides run on the same device with the same clocks.

### Timing

- **Both sides are timed on the GPU, with one event pair around one launch.**
  The vendor's pair is recorded by `cudaTimed` around the library call. CUT's is
  recorded by the command buffer around the whole submission and read back via
  `Runtime::lastSubmitSpanMicros()`; per-dispatch timestamps are switched *off*
  for these runs, because they sit between kernels and would widen the very gap
  they are inside. Every benchmark runs under Google Benchmark's
  `UseManualTime()`: the library owns the loop and the statistics, but the clock
  stays on the GPU. Without it Google Benchmark would report host wall-clock,
  which for an asynchronous GPU submit measures little more than enqueueing cost.
- **What each window contains.** CUT's span covers every dispatch the op expands
  to *and the gaps between them* — for `softmax` (a multi-kernel composition) or
  `scan` (a descriptor zero-fill plus the scan kernel) that is the whole latency a
  caller waits for. Summing per-dispatch timings instead would report a smaller,
  unachievable figure. The vendor's window is the same shape: `DeviceScan` clears
  its own scratch inside the call it is timed around, so both sides are charged
  for their setup kernels.
- **The one asymmetry that remains, and its size.** CUT's host-side op building —
  node construction, kernel lookup, output allocation, encode — happens in
  `issue()` *before* `rt.flush()` records the start event, whereas a vendor
  library's host-side work (cuBLAS heuristic dispatch, cuDNN descriptor
  validation) happens after `cudaEventRecord(start)`, on an idle GPU, and lands
  inside the measurement. Measured on an RTX 3090 as the gap between a per-call
  event pair and a steady-state batch of 50:

  | call | per-pair | batched | difference |
  |---|---|---|---|
  | empty kernel (the floor both sides pay) | 2.94 µs | 1.58 µs | +1.37 µs |
  | `cublasSgemm` 128³ | 8.19 µs | 5.71 µs | +2.48 µs (+43%) |
  | `cublasSgemm` 1024³ | 110.4 µs | 107.9 µs | +2.52 µs (+2.3%) |
  | `cublasSgeam` 4096² | 164.9 µs | 163.7 µs | +1.19 µs (+0.7%) |
  | CUB `InclusiveSum` N=65536 | 6.85 µs | 4.94 µs | +1.91 µs (+39%) |
  | CUB `SortPairs` N=65536 | 51.2 µs | 50.0 µs | +1.21 µs (+2.4%) |

  The ~1.4 µs floor is idle-GPU launch latency CUT's span contains too, once per
  dispatch — so a multi-dispatch CUT op pays it more than once. The **net** tilt
  in CUT's favour is the residual: ~1.1 µs against cuBLAS, ~0.5 µs against CUB, ~0
  against `sgeam`, which has no host-side heuristic to run. That is up to ~10% on
  the sub-20 µs cases and under 1% above ~100 µs. Rows in that band should not be
  quoted to the percent; if they ever need to be, time a batch of N launches
  inside one event pair on both sides and divide.
- **Order matters slightly, and CUT always goes first.** `registerPair` registers
  `cut/<case>` before `<vendor>/<case>` and Google Benchmark runs in registration
  order, so the vendor half always inherits a card the CUT half just warmed.
  Measured on the 3090: a compute-bound `sgemm 1024³` from a 10 s idle runs
  114.7 µs on the first call, 110.5 µs shortly after, and 104.4 µs once ~0.3 s of
  GPU work has accumulated — about 5%, biasing *against* whichever side runs
  first. A bandwidth-bound `sgeam 4096²` shows none of it, holding 819 GB/s from
  call 0. Median-across-repetitions absorbs most of what is left; `--interleave`
  removes the ordering entirely.
- **Setup is hoisted out of the timed region.** Operands are uploaded once at
  registration and the timed callable is the dispatch alone. This matters twice
  over: the vendor side uploads once, so leaving CUT to re-upload per iteration
  would tilt the comparison; and because the upload is invisible to the GPU
  timestamp but not to the wall clock, the adaptive loop would keep iterating to
  accumulate manual time and never finish. Measured before the fix: a 34 µs GEMV
  carried 3152 µs of hidden per-iteration setup and ran 20129 iterations.
- **Iteration count is adaptive, and the median is reported** across
  `--benchmark_repetitions`. Each body also does three untimed calls first, which
  absorb CUT's one-time kernel compilation. The model-scale cases turn that down
  to one warmup and pin their iteration count: at 0.6 s a call, three warmups cost
  more than the measurement, and each issue allocates a multi-GB output tensor
  whose host-side cost the manual-time loop cannot see.
- **Destructive operators pin their iteration count.** CUT's `sortRadix` family
  sorts in place, so each iteration must re-upload its input or it would re-sort
  sorted data and report a misleadingly fast number. That refill is exactly the
  hidden per-iteration cost the adaptive loop cannot see, so those cases set
  `CaseSpec::iterations` and lean on `--benchmark_repetitions` for stability.

### Reading the rate columns

Set `CaseSpec::flops` for compute-bound ops and `CaseSpec::bytes` for
memory-bound ones; the counter then reads `FLOPS` or `bytes_per_second`. Quoting
GFLOPS for scan or sort would be meaningless. Units differ between the two views:
Google Benchmark's console prints `bytes_per_second` with 1024-based `Gi/s`
suffixes, while `vendor_compare.py` reports decimal GB/s (1e9).

**A rate here is a fraction of an ideal, not a measured bandwidth or FLOP rate.**
Both halves are divided by the same denominator, so the *ratio* is exactly the
ratio of their times and always means what it says. The absolute figure often
does not:

- **Softmax GB/s understates both sides, but not equally.** The denominator is
  one read and one write per element; a stable softmax reads twice, and CUT's
  composition also materialises the broadcast max, the shifted input and the
  exponentials at full size — which is why the large cases charge it 8x its
  element count against cuDNN's 2x. Part of CUT's gap here is that it moves more
  bytes, not that it moves them slower.
- **conv2d GFLOPS overstates a Winograd row.** cuDNN's heuristic may pick
  Winograd, which computes a 3x3 with ~2.25x fewer multiplies than the nominal
  count both sides are charged, so its reported FLOP/s can exceed the device's
  fp32 FMA peak. Which algorithm ran is recorded per case as the `cudnn_algo`
  counter in the JSON (and named on stderr at registration), so a row can be
  checked rather than assumed like-for-like.

### Correctness is a gate, not a column

It runs once at registration, outside every timed region, and rides along as the
`max_diff`, `ref_mag` and `max_diff_allowed` counters on both halves of the pair,
so a JSON consumer never has to join two sources to find out whether a fast
number was also a correct one. `ref_mag` is the mean magnitude of the reference
output: without it, two silently-empty buffers would agree perfectly. A zero
`ref_mag`, or a `max_diff` over the case's tolerance, fails the case via
`SkipWithError`, marks **both** halves, and makes the binary exit **2**.

This used to be advisory, and that was a real hole: the f16 GEMV at M=1 returned
garbage (`max_diff` 8.4e13 against a `ref_mag` of 24) and for a while it printed a
GFLOPS figure and exited 0 like any passing case. A number that is fast and wrong
is the one number a benchmark must never quietly report.

**Above ~12M elements the check is sampled, not exhaustive.** The largest output
in the suite is nearly two billion elements; holding both sides of it on the host
as f32 would cost more memory than the GPU side of the whole benchmark.
`planSample` reads a bounded prefix, middle and suffix from both sides at
identical offsets instead. This is a real weakening — a kernel wrong only in the
unsampled region would pass — but a GEMM that is wrong is essentially never wrong
only there, and the alternative was not checking the large cases at all.

**A case declares what it holds.** `CaseSpec::footprintBytes` is reported as the
`vram_gb` counter and is what the budget check gates on. It is not always a sum of
the obvious buffers — the softmax cases charge eight times their element count,
measured against `nvidia-smi` rather than counted off the source, because a case
sized as though CUT needed only an input and an output aborted the run at 21 GB
resident.

### Tolerances

Every case declares the largest `max_diff` it accepts, via `CaseSpec::tolerance`.
A case passes when `max_diff <= max(absolute, relative * ref_mag)`.

These are **garbage detectors, not precision certificates**. Real rounding error
and a broken kernel are many orders of magnitude apart, so each bound is set well
above measured error. A loose gate that is always on beats a tight one that gets
switched off the first time it cries wolf.

| Operator | Tolerance | Measured worst case | Why |
|---|---|---|---|
| f32 GEMM/GEMV | `rel(1e-4)` | 0 | Different accumulation order than cuBLAS; in practice bit-exact here |
| f16 GEMM (both tiers) | `rel(1e-3)` | 7.5e-5 | f16 operands, f32 accumulate on both sides; aligned M is bit-exact, only the shapes that miss the coopmat tiling drift |
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
- **f16 GEMM `max_diff` is now comparable with the f32 table's.** It used not to
  be: with an f16 reference `C` every row sat at ~2e-3 relative no matter what CUT
  computed, because that is what f16 storage costs.

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
**withholds the speedup** — a failed row reads `VOID`, not a ratio, so there is no
number on the line to quote by accident:

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
that ran fast and wrong is a wrong answer. A binary exiting 2 still has its JSON
compared — dropping it would hide exactly the failures the gate exists to
surface.

## Adding an operator

1. Add a source next to its vendor's siblings (`cuda/`, `amd/`).
2. On CUDA, include `CudaBenchCommon.h` for `cudaTimed` and the model-scale
   helpers; on HIP, copy `hipTimed` from a sibling.
3. Write one `register<Op>Case` per shape. Upload both sides' operands **once**,
   before any timing, and make the CUT issue lambda the dispatch alone. Capture
   `Runtime` by reference and everything else by value: registration happens in
   `main` and the lambdas outlive that scope.
4. **Check that the two sides are running the same operator**, before caring what
   either one costs. Same input dtypes, same *output* dtype and element count,
   same in-place-or-not contract, and no math mode that lets one side drop
   precision the other keeps. Where the APIs cannot be made identical, work out
   what the difference costs and write it down. A mismatch here does not show up
   as a wrong answer; it shows up as a plausible speedup and a `max_diff` that is
   really a measurement of the reference.
5. Run the correctness check outside any timed region and store the
   `compareBuffers` result in `CaseSpec::check`.
6. Set `CaseSpec::tolerance` from the `max_diff` the case actually produces
   against its `ref_mag`, an order of magnitude above it. Leaving it unset gets
   the `rel(1e-2)` default, which catches garbage but is looser than most
   operators deserve.
7. Fill the rest of the `CaseSpec`: `flops` for a compute-bound op or `bytes` for
   a memory-bound one — set exactly one — then call `registerPair`. If the
   reference made a choice a reader needs in order to judge the row, put it in
   `CaseSpec::counters` so it reaches the JSON.
8. End `main` with `const int rc = cutbench::runAll(argc, argv);` followed by
   `runtime.shutdown();` and `return rc;`. That order is required: `runAll` clears
   the registry so the captured `cut::Tensor` handles are released while the
   runtime is still up, and letting the `Runtime` destructor run at end of `main`
   instead segfaults.
9. Device buffers, vendor handles and descriptors are deliberately leaked — they
   must outlive every registered lambda, and the process exits immediately after.
10. Register the target in that directory's `CMakeLists.txt`, linking
    `benchmark::benchmark`, guarded by a `find_*` so a missing SDK skips rather
    than breaks the build.

### Adding a model-scale case

Same shape of code, four differences:

1. Call `registerPairLazy` with a factory instead of `registerPair` with
   operands. The factory allocates, uploads, runs the correctness check, and
   returns a `LazyCase`; install `release` as soon as the device pointers exist,
   so an early return still frees them.
2. Allocate with `tryDeviceAlloc`, which returns null instead of aborting, and
   return an empty `shared_ptr` if it does — the case is then skipped rather than
   taking the run down with it.
3. Set `footprintBytes` to everything the case holds **on both sides**, and gate
   registration on `fitsVramBudget`. Measure it against `nvidia-smi` rather than
   deriving it from the obvious buffers if the CUT op is a composition.
4. Set `warmupIterations = 1`, pin `iterations`, and use `planSample` /
   `sampleTensor` / `sampleDeviceFloats` for the correctness check.
