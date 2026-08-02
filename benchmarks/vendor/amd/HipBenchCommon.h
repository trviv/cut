/// HIP-side timing shared by the AMD benches.
///
/// Parameterised on the event functions rather than calling them directly,
/// because the two wrappers expose their own C ABIs (`rocb*` for rocBLAS,
/// `rocp*` for rocPRIM) and neither .cpp may include a HIP header. Passing the
/// four function pointers in is what lets one copy of the timing logic serve
/// both without unifying those ABIs.
#pragma once

#include "VendorBench.h"

#include <functional>

namespace cutbench {

/// The event API one wrapper exposes, as plain function pointers.
struct HipEventApi {
  void *(*create)();
  void (*record)(void *);
  void (*synchronize)(void *);
  float (*elapsedMs)(void *, void *);
};

/// Times one vendor launch on both clocks — see TimedResult.
///
/// `gpuMs` is the interval between two stream events, so the library's own
/// host-side work lands inside it as GPU idle; `wallMs` adds the synchronise.
/// The CUT side measures its wall with the same helper, so the pair is
/// comparable even though the device clocks are not the same kind.
///
/// The events are created once and owned by the returned closure, so the
/// per-iteration cost is a record/sync pair rather than an allocation. They are
/// deliberately never destroyed — the closure outlives main().
inline TimedFn hipTimed(const HipEventApi &api, std::function<void()> launch) {
  void *start = api.create();
  void *stop = api.create();
  return [api, start, stop, launch]() {
    const double wallUs = wallMicros([&] {
      api.record(start);
      launch();
      api.record(stop);
      api.synchronize(stop);
    });
    return TimedResult{static_cast<double>(api.elapsedMs(start, stop)),
                       wallUs / 1000.0};
  };
}

} // namespace cutbench
