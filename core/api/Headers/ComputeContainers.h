#pragma once

#include <ComputeContainer.h>
#include <ComputeStructs.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;

/**
 * Abstract container for managing CommandBuffer objects via handles.
 * Derived classes must implement createCommandBuffer() to provide
 * backend-specific CommandBuffer instances.
 */
class CommandBufferContainer : public ComputeDataContainer<CommandBuffer *> {
public:
  CommandBufferContainer() = default;

  virtual ~CommandBufferContainer() = default;

  /**
   * Creates a new backend-specific CommandBuffer instance.
   * @return ComputeHandle to the created CommandBuffer.
   */
  virtual ComputeHandle createCommandBuffer() = 0;

private:
  friend class ComputeInterface;
};

} // namespace cut
