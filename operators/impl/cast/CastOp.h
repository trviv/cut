#pragma once

#include "OpNode.h"

namespace cut {

class CastOpNode : public OpNode {
public:
  CastOpNode(TensorStore &store,
             const Tensor &input,
             DataType targetDtype,
             std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  size_t shaderKey() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  std::string displayName() const override;

private:
  DataType srcDtype_;
  DataType dstDtype_;
  std::vector<uint32_t> shape_;
  uint32_t actualInner_;
  uint32_t alignedInner_;
  uint32_t totalElements_;
};

} // namespace cut
