
#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <cstdint>
#include <optional>
#include <vector>

// Generated dispatch table, forward declarations, and lookup functions.
#include "MatMulVariants.generated.h"

namespace cut {

/// Checks if an operator is the matmul op.
inline bool isMatMulOp(OperatorEnum op) {
  return op == MatMul;
}

} // namespace cut
