#include <gtest/gtest.h>
#include <Runtime.h>
#include <SharedRuntime.h>
#include "harness/OpRegistry.h"

namespace cut {
namespace {

// Exercises the run() path of EVERY registry case (the SAME OpCase definitions
// consumed by the op benchmark's perf path), keeping every perf-path definition
// compiled and dispatched. Correctness verification is intentionally NOT done
// here: each per-family driver in RuntimeTests.cpp is the single correctness
// path (calls verify() once), so cases are not verified twice.
TEST(OpRegistry, AllBuiltinCasesMatchReference) {
  Runtime *rt = test::sharedRuntime();
  if (!rt) GTEST_SKIP() << "No compute backend available";
  const auto &cases = opregistry::allOpCases();
  ASSERT_FALSE(cases.empty());
  for (const auto &c : cases) {
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*rt, -1);
    (void)out;
  }
  rt->flush();
}

} // namespace
} // namespace cut
