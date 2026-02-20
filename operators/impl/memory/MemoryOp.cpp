#include "MemoryOp.h"
#include "Runtime.h"

namespace cut {

// --- CopyOpNode ---

CopyOpNode::CopyOpNode(Runtime &runtime,
                       const Tensor &src,
                       std::vector<uint32_t> &&dstShape,
                       std::optional<uint32_t> spec)
    : OpNode(Copy, runtime, spec) {
  const auto &buf = runtime.getTensor(src);
  const auto srcShape = buf.getShape();
  dtype_ = buf.getDtype();
  dstShape_ = std::move(dstShape);
  srcInner_ = srcShape.empty() ? 1 : srcShape.back();
  srcAlignedInner_ = (srcInner_ + 3) & ~static_cast<uint32_t>(3);
  dstInner_ = dstShape_.empty() ? 1 : dstShape_.back();
  dstAlignedInner_ = (dstInner_ + 3) & ~static_cast<uint32_t>(3);
  totalElements_ = 1;
  for (auto d : dstShape_)
    totalElements_ *= d;
  if (dstShape_.empty())
    totalElements_ = 1;
  inputs_ = {src};
  output_ = runtime.createTensorEmpty(dstShape_, dtype_);
  hasOutput_ = true;
}

DataType CopyOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> CopyOpNode::outputShape() const {
  return dstShape_;
}

ThreadSize CopyOpNode::dispatchSize() const {
  return {totalElements_, 1, 1};
}

std::vector<uint8_t> CopyOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t srcAlignedInner;
    uint32_t srcActualInner;
    uint32_t dstAlignedInner;
    uint32_t dstActualInner;
    uint32_t totalElements;
  } pc{srcAlignedInner_, srcInner_, dstAlignedInner_, dstInner_,
       totalElements_};
  return toBytes(pc);
}

// --- EmbeddingOpNode ---

EmbeddingOpNode::EmbeddingOpNode(Runtime &runtime,
                                 const Tensor &indices,
                                 const Tensor &weight,
                                 std::optional<uint32_t> spec)
    : OpNode(Embedding, runtime, spec) {
  const auto &idxBuf = runtime.getTensor(indices);
  const auto &wBuf = runtime.getTensor(weight);
  const auto idxShape = idxBuf.getShape();
  const auto wShape = wBuf.getShape();
  dtype_ = wBuf.getDtype();
  if (wShape.size() != 2) {
    throw std::runtime_error(
        "embedding: weight must be 2D [num_embeddings, embedding_dim]");
  }
  embDim_ = wShape[1];
  numIndices_ = 1;
  for (auto d : idxShape)
    numIndices_ *= d;
  outShape_ = idxShape;
  outShape_.push_back(embDim_);
  inputs_ = {indices, weight};
  output_ = runtime.createTensorEmpty(outputShape(), DataType::Float32);
  hasOutput_ = true;
}

DataType EmbeddingOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> EmbeddingOpNode::outputShape() const {
  return outShape_;
}

ThreadSize EmbeddingOpNode::dispatchSize() const {
  uint32_t alignedDim4 = ((embDim_ + 3) & ~3u) / 4;
  uint32_t totalOutputs = numIndices_ * alignedDim4;
  uint32_t gridX = ((totalOutputs + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> EmbeddingOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numIndices;
    uint32_t embDim;
  } pc{numIndices_, embDim_};
  return toBytes(pc);
}

// --- PadOpNode ---

PadOpNode::PadOpNode(Runtime &runtime,
                     const Tensor &input,
                     std::vector<uint32_t> &&padWidths,
                     float value,
                     std::optional<uint32_t> spec)
    : OpNode(Pad, runtime, spec) {
  const auto &buf = runtime.getTensor(input);
  const auto shape = buf.getShape();
  dtype_ = buf.getDtype();
  padWidths_ = std::move(padWidths);
  value_ = value;
  int ndim = static_cast<int>(shape.size());
  if (padWidths_.size() % 2 != 0 ||
      static_cast<int>(padWidths_.size() / 2) > ndim) {
    throw std::runtime_error("pad: invalid padWidths length");
  }
  int numPaddedDims = static_cast<int>(padWidths_.size() / 2);

  outShape_ = shape;
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    outShape_[dim] += padWidths_[2 * i] + padWidths_[2 * i + 1];
  }

  totalOutputElements_ = 1;
  for (auto d : outShape_)
    totalOutputElements_ *= d;

  std::memset(&params_, 0, sizeof(params_));
  params_.ndim = static_cast<uint32_t>(ndim);
  params_.totalElements = totalOutputElements_;
  std::memcpy(&params_.fillValue, &value_, sizeof(float));
  for (int i = 0; i < ndim; ++i) {
    params_.inShape[i] = shape[i];
    params_.outShape[i] = outShape_[i];
    params_.padBefore[i] = 0;
  }
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    params_.padBefore[dim] = padWidths_[2 * i];
  }
  inputs_ = {input};
  output_ = runtime.createTensorEmpty(outputShape(), outputDtype());
  hasOutput_ = true;
}

DataType PadOpNode::shaderDtype() const {
  return dtype_;
}

std::vector<uint32_t> PadOpNode::outputShape() const {
  return outShape_;
}

ThreadSize PadOpNode::dispatchSize() const {
  uint32_t gridX = ((totalOutputElements_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> PadOpNode::pushConstants() const {
  return toBytes(params_);
}

} // namespace cut
