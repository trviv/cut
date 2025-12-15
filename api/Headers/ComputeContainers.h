#pragma once

#include <ComputeContainer.h>
#include <ComputeStructs.h>

namespace cut {

/** Forward declarations. */
class ComputeInterface;
class CommandBufferContainer;

/**
 * Container for managing CommandBuffer objects via handles.
 */
class CommandBufferContainer
    : public ComputeDataContainer<std::unique_ptr<CommandBuffer>> {
public:
  CommandBufferContainer()
      : ComputeDataContainer<std::unique_ptr<CommandBuffer>>(2) {}

private:
  friend class ComputeInterface;
};

} // namespace cut
