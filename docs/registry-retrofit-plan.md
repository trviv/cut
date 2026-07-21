# Op-case registry retrofit — handoff brief

**Objective:** migrate the existing operator tests in `tests/runtime/RuntimeTests.cpp`
(~239 tests) onto the shared **gtest-free op-case registry** so ONE `OpCase`
definition drives both the correctness gtest AND the `op_bench` perf path —
**without reducing coverage or breaking the green suite.**

## Current state (already committed on `main`)
- `tests/harness/OpRegistry.h` — gtest-free header, `namespace cut::opregistry`:
  ```
  struct VerifyResult { bool ok; std::string detail; };
  struct OpCase { std::string name, family;
    std::function<Tensor(Runtime&, int variant)> run;
    std::function<VerifyResult(Runtime&, const Tensor&)> verify; };
  const std::vector<OpCase>& allOpCases();   // built by buildOpCases()
  ```
  Because `verify` returns a plain result (no gtest macros), both the test binary
  and `op_bench` link it. Currently covers only binary (add/sub/mul/max/min) +
  unary (neg/abs/square/relu), F32, exact refs — a DEMO, not the full matrix.
- Correctness consumer: `tests/runtime/OpRegistryTests.cpp`
  (`TEST(OpRegistry, AllBuiltinCasesMatchReference)`).
- Perf consumer: `op_bench` `benchRegistry()` (times each case's `run`).
- Whole suite: **462/462 green on Vulkan AND CUDA.**

## HARD RULES (do not violate)
1. **All code generation goes through the local Ollama model** per repo
   `CLAUDE.md`: plan → write plan to a temp file → `python3 scripts/dev/ollama_code.py
   --context <files> /tmp/plan.txt` → review → apply with Edit/Write, fixing what
   the model got wrong. Never hand-author operator/library code. (If the server
   is down: `nohup ollama serve >/tmp/ollama.log 2>&1 &`; model = `devstral-small-2:24b`.)
   Trivial mechanical assembly (splitting model output into files, inserting a
   verified fragment) is fine.
2. **Commit messages via Ollama** (see CLAUDE.md snippet). **Never add a
   `Co-Authored-By: Claude` trailer** (repo directive).
3. **Keep both backends green after every family.** Validate with:
   `bash scripts/build/run_tests_both_backends.sh` → must print `RESULT: vulkan=PASS
   cuda=PASS`. Baseline is 462 and only grows.
4. **Commit incrementally — one family per commit.** If context runs low, the
   committed families persist; a fresh session resumes from this doc.
5. **Never reduce coverage.** The registry must cover a family as comprehensively
   as the existing test BEFORE the old test is replaced. When unsure, keep both.
6. **Pin model opus on any sub-subagents.** (Fable subagents blew the spend limit.)
7. Free VRAM before CUDA test runs: `ollama stop
   devstral-small-2:24b` (it reloads on the next codegen call). The Vulkan device
   and CUDA device are both the RTX 3090 on this box.

## Design
### Step A — extract shared references (do FIRST)
The reference math + helpers live in `tests/runtime/RuntimeTests.cpp`'s anon
namespace. Move them VERBATIM (protect the math!) into a new gtest-free header
`tests/harness/OpRefs.h` (inline functions, `namespace cut::opref` or keep
`cut`), then `#include` it from both `OpRegistry.h` and `RuntimeTests.cpp`
(replacing the inline copies there). Functions/consts to extract (with
`RuntimeTests.cpp` line hints — verify current lines, the file shifts):
- half utils: `floatToHalf` :161, `halfToFloat` :179, `floatsToHalves` :207,
  `halvesToFloats` :214 (also `f32_to_f16`, `packNibbles` used by quant tests).
- shape/data: `generateShapes` :366 (NOTE: forces innermost dim to a multiple of
  4 for multi-dim), `totalElements` :407, `shapeToString` :416,
  `generateTestData<T>` :1227 (float uniform [0.1,10], int [1,100]).
- refs: `binaryVecVecRef` :432, `binaryVecScalarRef` :598, `unaryRef` :797,
  `ternaryClampRef` :1153, `ternarySelectRef` :1159, `reduceRef` :1165,
  `dimReduceRef` :2360, `normDimRef` :2420.
- config arrays: `kBinaryVecVecOps` (29), `kIntBinaryVecVecOps` (22),
  `kBinaryVecScalarOps` (33), `kUnaryOps` (55), `kAllDataTypes`, `kTestDimSizes`,
  `kDimReductionOps`.
To move verbatim safely: pass RuntimeTests.cpp as `--context` to ollama and ask
it to reproduce the named functions in a header, then DIFF each reproduced body
against the original and fix any drift before applying. Rebuild `tests` and
confirm still 462 green after the extraction (pure refactor, no behavior change).

### Step B — grow the registry to full coverage, family by family
Extend `buildOpCases()` (or split into per-family builders in
`tests/harness/OpRegistryCases.cpp` if the header gets big) so each case mirrors
the existing test's matrix. Order (biggest/most-uniform first):
1. binary vec-vec — 29 ops × {F32,F16,U32,I32} × `generateShapes(1..4)`, cmp ops
   → uint32 output, shift ops clamp rhs; F16 relaxed tol; use `binaryVecVecRef`.
2. binary vec-scalar — `kBinaryVecScalarOps`, `binaryVecScalarRef`.
3. unary — 55 ops (`kUnaryOps`), F32/I32/U32, `unaryRef` (skip non-finite refs).
4. ternary clamp/select — `ternaryClampRef`/`ternarySelectRef`.
5. reduce (global) + reducedim — `reduceRef`/`dimReduceRef`; argmax/argmin.
6. matmul (+ variant sweep via `spec`), transpose (+ variants), dot.
7. conv1d/conv2d (+ variants), maxpool/avgpool (+ variants), adaptive avgpool.
8. softmax/logSoftmax (2D only — plain softmax crashes on true 1D input),
   rmsNorm, layernorm, batchnorm, norm/normdim, embedding, pad.
9. cumsum/cumprod, prefix scan, sort (bitonic/radix), dequant, quant matmul
   (Q8/Q4 — gate `getCompiledMatMulQ8(vi,F32,F16,F32)`; SKIP variant names with
   "Dot"/"CoopMat": those need pre-packed operands — `MatMulQ8GemvDot` returns
   garbage with the plain [K,N] int8 layout, a suspected real bug).
Tolerances: F32 elementwise 1e-5; F32 accumulating `abs(exp)*1e-4+1e-5`; F16 rel
1e-2; int exact; cmp → uint32 `(ref!=0)`.

### Step C — retrofit the existing tests
For each family, AFTER the registry covers it and the registry gtest is green:
replace the old test body in `RuntimeTests.cpp` with a thin driver that iterates
the family's registry cases (run + verify), OR delete the old test in favor of
the registry-driven test. Keep `SCOPED_TRACE(c.name)`. Re-run
`run_tests_both_backends.sh` and confirm the total count didn't DROP relative to
the coverage the family had (it's fine for the number of gtest cases to change as
long as the assertion surface is preserved). Commit the family.

## Validation loop (every family)
```
./scripts/build/format.sh
cmake --build build --target tests -j$(nproc)                 # Vulkan ASan
GTEST_FILTER='OpRegistry*:<Family>*' bash scripts/build/run_tests_both_backends.sh
# expect: RESULT: vulkan=PASS cuda=PASS
```
Then a full `bash scripts/build/run_tests_both_backends.sh` before committing.

## Known pitfalls
- `generateShapes` pads innermost dim to a multiple of 4; `copyFromTensor`
  returns LOGICAL (unpadded) data — compare against logical refs.
- Plain `softmax(a,dim)`/`logSoftmax` throw on true 1D input (reduce→unsqueeze→
  expand makes a 2D intermediate). Use 2D `{1,D}` shapes.
- Quant `getCompiledMatMulQ8/Q4` default scales dtype is Float32 but real scales
  are **Float16** — pass `(vi, Float32, Float16, Float32)`.
- Half-float util must match the GPU bit math (copy verbatim from RuntimeTests).
- Device caps: `runtime.store().caps().integerDotProduct` /
  `.cooperativeMatrix` gate hw-specific variants.

## Findings from session 1 (7 families done: refs extract, binary vec-vec,
## binary vec-scalar, unary, ternary, global reduce, dim-reduce, normdim —
## commits dd15d16..bb194b3). Bake these into your approach:
- **Ollama (`devstral-small-2:24b`) pitfalls, recurring:** it emits C++20
  designated initializers (`.name=…`) and `const char* + const char*` name
  concatenation, and sometimes omits a referenced struct def. WRITE PLANS THAT
  DEMAND C++17: build cases as `OpCase c; c.name = std::string("…") +
  operatorName(op) + …; c.family = "…"; c.run = …; c.verify = …;
  cases.push_back(std::move(c));` and include every struct. Review/fix each
  generation for these before applying. Fragment-A (math/sweep helpers) is
  usually faithful; Fragment-B (the case-builder) needs the most fixing.
- **`VerifyResult` must be UNQUALIFIED** inside `namespace cut::opregistry`
  (the model writes `cut::VerifyResult`, which won't compile).
- **Double-execution (address this):** `OpRegistry.AllBuiltinCasesMatchReference`
  verifies EVERY family, and each per-family driver verifies its family AGAIN, so
  migrated families run their verify twice and the suite keeps getting slower.
  FIX: make `AllBuiltinCasesMatchReference` run-only (call `run` for coverage of
  perf-path compilation but drop its `verify` loop), leaving the per-family
  drivers as the single correctness path. Do this early in session 2.
- Pattern used per family: `run(rt,variant)` = one representative dispatch (for
  op_bench timing); `verify(rt,out)` = the FULL original sweep (all shapes/dtypes/
  tolerances) so no coverage is lost; the old per-family gtest becomes a thin
  driver iterating that family (`filter on c.family + a name substring`).
- `createTensor` takes a single `const std::vector<uint32_t>&` — braced `{64,64}`
  args compile fine.
- Run big Ollama generations in the BACKGROUND (its ~300s timeout + stream-to-file).
  NEVER run codegen during a CUDA test pass — the ~20 GB model + CUDA tests
  contend on the 24 GB 3090; `ollama stop devstral-small-2:24b` before each CUDA pass.

## Build/run cheat-sheet
- Vulkan ASan dev build: `build/` ; CUDA release build: `build-cuda-rel/`.
- Reconfigure after adding a source: `cmake -S . -B build` (and `build-cuda-rel`).
- Single test bin run: `ASAN_OPTIONS=detect_leaks=0 build/tests/tests
  --gtest_filter='...'` (Vulkan) ; add `CUT_TEST_BACKEND=cuda` +
  `build-cuda-rel/tests/tests` for CUDA.

## Session 2 progress (commits `0359c51`..`495677e`). Bake these into your approach:
- **Double-execution FIXED first** (commit `0359c51`): `OpRegistry.AllBuiltinCasesMatchReference`
  is now RUN-ONLY (calls `c.run` for perf-path coverage, no `verify` loop). Per-family
  drivers in `RuntimeTests.cpp` are the single correctness path.
- **Families migrated this session (one family-group per commit):**
  - `f67e983` matmul (default + variant sweeps), transpose (default + variants), dot.
  - `3aa2b00` conv1d, conv2d (default + variants).
  - `e1f354b` maxpool, avgpool, adaptive avgpool.
  - `0368bae` global norm, rms (global+dim), logsumexp (global+dim).
  - `d028e48` embedding, pad.
  - `912186d` layernorm, batchnorm.
  - `2c05135` softmax, logsoftmax (fused cross-checked vs composite; 2D+).
  - `1c59290` cumsum/cumprod ("cumulative" family, 1D/2D/3D + large multipass).
  - `495677e` prefix scan (exclusive/inclusive), sort (bitonic float + radix uint32).
  - `528c144` dequant (BF16 round-trip + Q4_K/Q6_K super-block dequant); removed the
    now-unused `f16_bits_to_f32` static helper from RuntimeTests.
  - `6485e25` quantized matmul (Q8/Q4: simple/withscales/vs-reference + gated all-variant
    sweeps). Gates on `getCompiledMatMulQ8/Q4(vi, Float32, Float16, Float32)` and skips
    "Dot"/"CoopMat" variant names (need pre-packed operands). The batched-vs-per-row
    Mistral-geometry Q8 tests and the DISABLED_ test were left as-is (intricate).
  Test count held at **462/462 green on BOTH backends after every commit** (the migrated
  gtests keep their names, just become thin drivers, so the total does not change).
- **ALL families from the brief are now migrated.** The op-case registry (family strings
  in `OpRegistry.h`): binary_vecvec, binary_vecscalar, unary, ternary, reduce, dimreduce,
  normdim (session 1) + matmul, transpose, dot, conv1d, conv2d, maxpool, avgpool, norm,
  rms, logsumexp, embedding, pad, layernorm, batchnorm, softmax, logsoftmax, cumulative,
  prefixscan, sort, dequant, quantmatmul (session 2).
- **Intentionally NOT migrated** (too intricate, per brief): attention, rope, multi-device,
  and the Q8 batched-vs-per-row Mistral-geometry / DISABLED tests.
- **Workflow that worked well this session (use it):**
  - Pass a SMALL context to Ollama, NOT the growing `OpRegistry.h` (now ~2000 lines).
    Use `scratchpad/pattern_ref.txt` (a ~40-line pattern reference). When the full file is
    passed as context the model tends to regurgitate + hit its 16k `num_predict` cap and
    truncate mid-output (this bit the pooling generation).
  - Ask Ollama to emit just two marker-delimited sections: `===HELPERS===` and `===CASES===`
    (plus `===INCLUDES===` if new generated-variant headers are needed). Write near-final
    C++17 in the plan; the model reproduces it faithfully at small context. Review then apply.
  - `scratchpad/replace_tests.py <file> <spec.json>` mechanically rewrites each
    `TEST_F(Fixture, Name){...}` into a thin driver that iterates `opregistry::allOpCases()`
    filtered on an exact `c.name`. spec.json = list of `[fixture, test_name, case_name]`.
    This is the fast path for the driver conversion (fixtures use `runtime_`). Fixture
    member ref-helpers (e.g. `conv1dRef`) left unused are harmless (no -Wunused for members).
  - Poll for codegen completion by grepping the OUTPUT FILE for the last case name — do NOT
    `pgrep -f ollama_code.py` from a bash wrapper (its own command line self-matches → the
    wait loop never exits). Foreground `sleep` is blocked; use background bash `until` loops.
- **Pre-existing (not introduced here):** a UBSan `signed integer overflow` note at
  `OpRefs.h:1035` (int `ReduceProd` reference) prints during runs but is non-fatal; suite
  stays green. Leave it or fix the reference to accumulate in a wider type in a separate commit.
- `Norm`, `CumSum`, `CumProd`, `PrefixScanExclusiveSum`, `PrefixScanInclusiveSum` are
  `OperatorEnum` values usable UNQUALIFIED inside `namespace cut::opregistry`.
