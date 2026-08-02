#pragma once

#include "OpNode.h"

namespace cut {

/// Variant indices, in the order shaders.json declares them. Append only: an
/// insert would renumber the rest, and the index is what selects the shader.
enum SoftmaxVariant : uint32_t {
  kSoftmaxVariantSoftmax = 0,
  kSoftmaxVariantLogSoftmax = 1,
  kSoftmaxVariantSoftmaxWide = 2,
  kSoftmaxVariantLogSoftmaxWide = 3,
};

/// The two knobs the native CUDA kernel's row mapping depends on. Both MUST
/// stay identical to their counterparts in SoftmaxCommon.cuh
/// (CUT_SOFTMAX_WARP_ROW_MAX and CUT_SOFTMAX_NVEC): the host sizes the grid from
/// these and the kernel derives its row mapping from those, both off the same
/// pc.reduceSize. If the two disagree the blocks and the rows stop lining up and
/// the output is silently wrong — rows are left unwritten rather than the
/// dispatch failing. The softmax test sweep pins the boundaries on both sides.
inline constexpr uint32_t kSoftmaxWarpRowMaxCols = 512;
inline constexpr uint32_t kSoftmaxNVec = 8;

/// Threads that cooperate on one row: a warp for short rows, the whole block
/// for long ones. wgSize is the variant's block size, not a constant.
inline constexpr uint32_t softmaxThreadsPerRow(uint32_t reduceSize,
                                               uint32_t wgSize) {
  return reduceSize <= kSoftmaxWarpRowMaxCols ? 32u : wgSize;
}

/// Longest row a wgSize-thread block can hold in registers, and therefore read
/// exactly once. Past this the kernel streams, which costs a second read of the
/// whole row — 3 passes over DRAM instead of 2, measured at almost exactly the
/// 1.5x that implies.
inline constexpr uint32_t softmaxRegisterResidentCols(uint32_t wgSize) {
  return wgSize * 4u * kSoftmaxNVec;
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
  uint32_t variant_ = kSoftmaxVariantSoftmax;
  uint32_t wgSize_ = 256;
  bool cudaRowMapping_ = false;
  std::vector<uint32_t> outShape_;
};

} // namespace cut
