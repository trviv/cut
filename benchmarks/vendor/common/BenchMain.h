/// The main() every vendor bench ends with.
///
/// Worth a header rather than seven copies because the ORDER is load-bearing and
/// not obvious: runAll must clear the benchmark registry before the runtime is
/// shut down, since the registered lambdas hold refcounted cut::Tensor handles
/// and CUT treats a non-empty buffer container at shutdown as fatal. Letting the
/// Runtime destructor run at end of main instead of calling shutdown() segfaults.
/// Seven copies of that is seven chances to get it wrong once.
#pragma once

#include "VendorBench.h"

#include <Runtime.h>

#include <cstdlib>
#include <functional>
#include <iostream>

namespace cutbench {

/// Brings up `backend`, lets `registerCases` register every comparison, runs
/// them, and tears down in the required order. Returns the process exit code:
/// 0 clean, 1 the backend was unavailable or the arguments were bad, 2 a case
/// failed its correctness gate.
inline int runVendorBenchMain(
    int argc, char **argv, cut::BackendType backend,
    const std::function<void(cut::Runtime &)> &registerCases) {
  // CUT logs every dispatch to stderr otherwise, which buries the table.
  setenv("CUT_PROFILE_QUIET", "1", 1);

  cut::Runtime runtime;
  if (backend == cut::BackendType::CUDA && !runtime.isCudaAvailable()) {
    std::cerr << "CUDA backend unavailable "
                 "(build with -DENABLE_CUDA_BACKEND=ON)\n";
    return 1;
  }
  runtime.init(backend);
  // Required for the submit-span timing timeCutOnce reads.
  runtime.setProfilingEnabled(true);

  registerCases(runtime);

  const int rc = runAll(argc, argv);

  // See the note at the top: this order is not a style choice.
  runtime.shutdown();
  return rc;
}

} // namespace cutbench
