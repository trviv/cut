#pragma once

#include <ComputeContainer.h>
#include <ComputeStructs.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;
class CommandBufferContainer;

/**
 * Abstract container for managing CommandBuffer objects via handles.
 * Derived classes must implement createCommandBuffer() to provide
 * backend-specific CommandBuffer instances.
 */
class CommandBufferContainer
    : public ComputeDataContainer<std::unique_ptr<CommandBuffer>> {
public:
  CommandBufferContainer()
      : ComputeDataContainer<std::unique_ptr<CommandBuffer>>(2) {}

  virtual ~CommandBufferContainer() = default;

protected:
  /**
   * Creates a new backend-specific CommandBuffer instance.
   * @return Unique pointer to the created CommandBuffer.
   */
  virtual std::unique_ptr<CommandBuffer> createCommandBuffer() = 0;

private:
  friend class ComputeInterface;
};

} // namespace cut
