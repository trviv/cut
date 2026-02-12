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
   *
   * @param op The operator to execute (from OperatorEnum).
   * @param bindings Vector of compute bindings (buffers and data).
   * @param shader Pre-created shader handle for this operator.
   * @param executionSize Number of elements to process (from buffer execution
   * size).
   */
  void encode(OperatorEnum op,
              const std::vector<ComputeBinding> &bindings,
              const ComputeHandle &shader,
              size_t executionSize);

private:
  ComputeInterface *iface_;
};

} // namespace cut
