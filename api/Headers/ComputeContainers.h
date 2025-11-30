#pragma once

#include <ComputeContainer.h>
#include <ComputeStruct.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;
class CommandBufferContainer;

/**
 * Records compute dispatch operations for later submission.
 * Created via ComputeInterface::beginCommandBuffer() and finalized
 * via ComputeInterface::endCommandBuffer().
 */
class CommandBuffer final : public ComputeDataContainer<ComputeDispatch> {
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

  /** Constructs a CommandBuffer with type ID 1. */
  CommandBuffer() : ComputeDataContainer<ComputeDispatch>(1) {}
};

/**
 * Container for managing CommandBuffer objects via handles.
 */
class CommandBufferContainer final
    : public ComputeDataContainer<CommandBuffer *> {
  /**
   * Creates a new CommandBuffer and returns its handle.
   * @return Handle to the created CommandBuffer.
   */
  ComputeHandle create() { return createHandle(new CommandBuffer()); }

private:
  friend class ComputeInterface;

  CommandBufferContainer() : ComputeDataContainer<CommandBuffer *>(2) {}
};

} // namespace cut
