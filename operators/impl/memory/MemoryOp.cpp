#include "MemoryOp.h"

namespace cut {

// --- TransposeOpNode ---

TransposeOpNode::TransposeOpNode(std::vector<uint32_t> shape, DataType dtype)
    : OpNode(Transpose), shape_(std::move(shape)), dtype_(dtype) {
  if (shape_.size() != 2) {
    throw std::runtime_error("transpose requires a 2D matrix");
  }
  M_ = shape_[0];
  N_ = shape_[1];
}

DataType TransposeOpNode::shaderDtype() const {
  return dtype_;
}
DataType TransposeOpNode::outputDtype() const {
  return DataType::Float32;
}

std::vector<uint32_t> TransposeOpNode::outputShape() const {
  return {N_, M_};
}

ThreadSize TransposeOpNode::dispatchSize() const {
  const uint32_t tileSize = 16;
  uint32_t strideOut = (M_ + 3) & ~3u;
  uint32_t strideOut4 = strideOut / 4;
  uint32_t gridX = (N_ + tileSize - 1) / tileSize * tileSize;
  uint32_t gridY = (strideOut4 + tileSize - 1) / tileSize * tileSize;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> TransposeOpNode::pushConstants() const {
  uint32_t strideIn = (N_ + 3) & ~3u;
  uint32_t strideOut = (M_ + 3) & ~3u;
  struct PushConstants {
    uint32_t M, N, strideIn, strideOut;
  } pc{M_, N_, strideIn, strideOut};
  return toBytes(pc);
}

// --- CopyOpNode ---

CopyOpNode::CopyOpNode(std::vector<uint32_t> srcShape,
                       std::vector<uint32_t> dstShape,
                       DataType dtype)
    : OpNode(Copy), srcShape_(std::move(srcShape)),
      dstShape_(std::move(dstShape)), dtype_(dtype) {
  srcInner_ = srcShape_.empty() ? 1 : srcShape_.back();
  srcAlignedInner_ = (srcInner_ + 3) & ~static_cast<uint32_t>(3);
  dstInner_ = dstShape_.empty() ? 1 : dstShape_.back();
  dstAlignedInner_ = (dstInner_ + 3) & ~static_cast<uint32_t>(3);
  totalElements_ = 1;
  for (auto d : dstShape_)
    totalElements_ *= d;
  if (dstShape_.empty())
    totalElements_ = 1;
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

EmbeddingOpNode::EmbeddingOpNode(std::vector<uint32_t> idxShape,
                                 std::vector<uint32_t> wShape,
                                 DataType weightDtype)
    : OpNode(Embedding), idxShape_(std::move(idxShape)),
      wShape_(std::move(wShape)), dtype_(weightDtype) {
  if (wShape_.size() != 2) {
    throw std::runtime_error(
        "embedding: weight must be 2D [num_embeddings, embedding_dim]");
  }
  embDim_ = wShape_[1];
  numIndices_ = 1;
  for (auto d : idxShape_)
    numIndices_ *= d;
  outShape_ = idxShape_;
  outShape_.push_back(embDim_);
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

PadOpNode::PadOpNode(std::vector<uint32_t> shape,
                     std::vector<uint32_t> padWidths,
                     float value,
                     DataType dtype)
    : OpNode(Pad), shape_(std::move(shape)), padWidths_(std::move(padWidths)),
      value_(value), dtype_(dtype) {
  int ndim = static_cast<int>(shape_.size());
  if (padWidths_.size() % 2 != 0 ||
      static_cast<int>(padWidths_.size() / 2) > ndim) {
    throw std::runtime_error("pad: invalid padWidths length");
  }
  int numPaddedDims = static_cast<int>(padWidths_.size() / 2);

  outShape_ = shape_;
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    outShape_[dim] += padWidths_[2 * i] + padWidths_[2 * i + 1];
  }

  totalOutputElements_ = 1;
  for (auto d : outShape_)
    totalOutputElements_ *= d;

  // Build PadParams struct
  std::memset(&params_, 0, sizeof(params_));
  params_.ndim = static_cast<uint32_t>(ndim);
  params_.totalElements = totalOutputElements_;
  std::memcpy(&params_.fillValue, &value_, sizeof(float));
  for (int i = 0; i < ndim; ++i) {
    params_.inShape[i] = shape_[i];
    params_.outShape[i] = outShape_[i];
    params_.padBefore[i] = 0;
  }
  for (int i = 0; i < numPaddedDims; ++i) {
    int dim = ndim - 1 - i;
    params_.padBefore[dim] = padWidths_[2 * i];
  }
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
