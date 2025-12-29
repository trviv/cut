#pragma once

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <ComputeStructs.h>

#include <vector>

namespace cut {

// Forward declarations
class ComputeInterface;

/**
 * Dispatcher class for encoding math operators to the compute backend.
 * Generates compute dispatches based on operator enums, inferring dtype
 * and workgroup size from buffer bindings.
 */
class Dispatcher {
public:
  /**
   * Constructs a Dispatcher with the given compute interface.
   * @param iface The compute interface to use for encoding dispatches.
   */
  explicit Dispatcher(ComputeInterface *iface);

  /**
   * Encodes a math operator dispatch to the compute backend.
   * Infers dtype from buffer bindings (validates they match) and computes
   * workgroup size from the total element count in the buffer shapes.
   *
   * @param op The operator to execute (from OperatorEnum).
   * @param bindings Vector of compute bindings (buffers and data).
   * @throws std::runtime_error if buffer dtypes don't match or shapes are
   * incompatible.
   */
  void encode(OperatorEnum op, const std::vector<ComputeBinding> &bindings);

  /**
   * Returns a string name for the given operator.
   * @param op The operator enum.
   * @return A human-readable name for the operator.
   */
  static const char *operatorName(OperatorEnum op);

private:
  ComputeInterface *iface_;
};

} // namespace cut
