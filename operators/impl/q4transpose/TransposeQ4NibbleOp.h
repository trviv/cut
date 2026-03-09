#pragma once

#include "OpNode.h"

namespace cut {

/// GPU Q4_0 nibble transpose: [N, K/2] packed -> [K, N/2] packed.
/// Performs unpack + transpose + repack in a single GPU dispatch.
class TransposeQ4NibbleOpNode : public OpNode {
public:
  TransposeQ4NibbleOpNode(TensorStore &store,
                          const Tensor &packedInput,
                          uint32_t N,
                          uint32_t K,
                          std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  uint32_t N_, K_;
  uint32_t strideInHalf_;
  uint32_t strideOutHalf_;
};

} // namespace cut
