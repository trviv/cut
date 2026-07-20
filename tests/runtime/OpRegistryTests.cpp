#include <gtest/gtest.h>
#include <Runtime.h>
#include <SharedRuntime.h>
#include "harness/OpRegistry.h"

namespace cut {
namespace {

// Drives EVERY registry case through the correctness path. The SAME OpCase
// definitions are consumed by the op benchmark's perf path (via run()), so op
// tests and perf timing share one source of truth.
TEST(OpRegistry, AllBuiltinCasesMatchReference) {
  Runtime *rt = test::sharedRuntime();
  if (!rt) GTEST_SKIP() << "No compute backend available";
  const auto &cases = opregistry::allOpCases();
  ASSERT_FALSE(cases.empty());
  for (const auto &c : cases) {
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*rt, -1);
    if (!c.verify) continue;
    opregistry::VerifyResult vr = c.verify(*rt, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
  rt->flush();
}

} // namespace
} // namespace cut
