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
   `CLAUDE.md`: plan → write plan to a temp file → `python3 scripts/ollama_code.py
   --context <files> /tmp/plan.txt` → review → apply with Edit/Write, fixing what
   the model got wrong. Never hand-author operator/library code. (If the server
   is down: `nohup ollama serve >/tmp/ollama.log 2>&1 &`; model = `devstral-small-2:24b`.)
   Trivial mechanical assembly (splitting model output into files, inserting a
   verified fragment) is fine.
2. **Commit messages via Ollama** (see CLAUDE.md snippet). **Never add a
   `Co-Authored-By: Claude` trailer** (repo directive).
3. **Keep both backends green after every family.** Validate with:
   `bash scripts/run_tests_both_backends.sh` → must print `RESULT: vulkan=PASS
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
./scripts/format.sh
cmake --build build --target tests -j$(nproc)                 # Vulkan ASan
GTEST_FILTER='OpRegistry*:<Family>*' bash scripts/run_tests_both_backends.sh
# expect: RESULT: vulkan=PASS cuda=PASS
```
Then a full `bash scripts/run_tests_both_backends.sh` before committing.

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

## Build/run cheat-sheet
- Vulkan ASan dev build: `build/` ; CUDA release build: `build-cuda-rel/`.
- Reconfigure after adding a source: `cmake -S . -B build` (and `build-cuda-rel`).
- Single test bin run: `ASAN_OPTIONS=detect_leaks=0 build/tests/tests
  --gtest_filter='...'` (Vulkan) ; add `CUT_TEST_BACKEND=cuda` +
  `build-cuda-rel/tests/tests` for CUDA.
