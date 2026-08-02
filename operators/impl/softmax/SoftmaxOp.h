#pragma once

#include "OpNode.h"

namespace cut {

/// Threads that cooperate on one softmax row in the native CUDA kernel: a warp
/// for short rows, the whole 256-thread block for long ones.
///
/// MUST stay identical to cut_softmax_threads_per_row() in SoftmaxCommon.cuh.
/// The host sizes the grid with this and the kernel derives its row mapping
/// with that, from the same pc.reduceSize; if the two rules disagree, blocks
/// and rows stop lining up and the result is silently wrong, not a crash.
inline constexpr uint32_t kSoftmaxWgSize = 256;
inline constexpr uint32_t kSoftmaxWarpRowMaxCols = 512;

inline constexpr uint32_t softmaxThreadsPerRow(uint32_t reduceSize) {
  return reduceSize <= kSoftmaxWarpRowMaxCols ? 32u : kSoftmaxWgSize;
}

class SoftmaxOpNode : public OpNode {
public:
  /// op should be Softmax or LogSoftmax.
  SoftmaxOpNode(OperatorEnum op,
                TensorStore &store,
                const Tensor &a,
                int dim,
                std::optional<uint32_t> spec = {});

  DataType outputDtype() const override;
  std::optional<std::vector<uint32_t>> shader() const override;
  std::vector<uint32_t> outputShape() const override;
  ThreadSize dispatchSize() const override;
  std::vector<uint8_t> pushConstants() const override;
  size_t shaderKey() const override;

private:
  DataType dtype_;
  int dim_;
  uint32_t outerSize_;
  uint32_t reduceSize_;
  uint32_t innerSize_;
  uint32_t inOuterStride_;
  uint32_t inReduceStride_;
  uint32_t bufInnerDim_;
  uint32_t alignedBufInner_;
  bool cudaRowMapping_ = false;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
