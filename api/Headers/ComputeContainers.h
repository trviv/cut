#pragma once

#include <ComputeContainer.h>
#include <ComputeStructs.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;
class CommandBufferContainer;

/**
 * Records compute dispatch operations for later submission.
 * Created via ComputeInterface::beginCommandBuffer() and finalized
 * via ComputeInterface::endCommandBuffer().
 */
class ComputeDispatchContainer final : public ComputeDataContainer<ComputeDispatch> {
public:
  /**
   * Encodes a dispatch operation to this command buffer.
   * @param dispatch The compute dispatch to encode (moved).
   * @return Handle to the encoded dispatch.
   */
  ComputeHandle encode(ComputeDispatch &&dispatch);

private:
  friend class ComputeInterface;
  friend class CommandBufferContainer;

  /** Constructs a ComputeDispatchContainer with type ID 1. */
  ComputeDispatchContainer() : ComputeDataContainer<ComputeDispatch>(1) {}
};

/**
 * Container for managing ComputeDispatchContainer objects via handles.
 */
class CommandBufferContainer final
    : public ComputeDataContainer<ComputeDispatchContainer *> {
  /**
   * Creates a new ComputeDispatchContainer and returns its handle.
   * @return Handle to the created ComputeDispatchContainer.
   */
  ComputeHandle create() {
    ComputeDispatchContainer *ptr = new ComputeDispatchContainer();
    return ComputeDataContainer<ComputeDispatchContainer *>::create(std::move(ptr));
  }

private:
  friend class ComputeInterface;

  CommandBufferContainer() : ComputeDataContainer<ComputeDispatchContainer *>(2) {}
};

} // namespace cut
