#include "MemoryOp.h"
#include "MemoryShaders.generated.h"
#include "Shaders.h"
#include "TensorStore.h"

namespace cut {

// --- CopyOpNode ---

CopyOpNode::CopyOpNode(TensorStore &store,
                       const Tensor &src,
                       std::vector<uint32_t> &&dstShape,
                       std::optional<uint32_t> spec)
    : OpNode(Copy, store, spec) {
  const auto &buf = store.getTensor(src);
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
  output_ = store.createTensorEmpty(dstShape_, dtype_);
}

DataType CopyOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> CopyOpNode::shader() const {
  auto compiled = compiledCopy(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

EmbeddingOpNode::EmbeddingOpNode(TensorStore &store,
                                 const Tensor &indices,
                                 const Tensor &weight,
                                 const Tensor &preallocOutput,
                                 std::optional<uint32_t> spec)
    : OpNode(Embedding, store, spec) {
  const auto &idxBuf = store.getTensor(indices);
  const auto &wBuf = store.getTensor(weight);
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
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outputShape(), dtype_);
}

DataType EmbeddingOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> EmbeddingOpNode::shader() const {
  auto compiled = compiledEmbedding(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

EmbeddingColOpNode::EmbeddingColOpNode(TensorStore &store,
                                       const Tensor &indices,
                                       const Tensor &matrix,
                                       const Tensor &scales,
                                       const Tensor &preallocOutput,
                                       std::optional<uint32_t> spec)
    : OpNode(EmbeddingCol, store, spec) {
  const auto &idxBuf = store.getTensor(indices);
  const auto &mBuf = store.getTensor(matrix);
  const auto &sBuf = store.getTensor(scales);
  const auto idxShape = idxBuf.getShape();
  const auto mShape = mBuf.getShape();
  const auto sShape = sBuf.getShape();

  if (mShape.size() != 2) {
    throw std::runtime_error(
        "embeddingCol: matrix must be 2D [dim, vocab]");
  }

  embDim_ = mShape[0];
  vocabStride_ = (mShape[1] + 3) & ~3u;
  scaleStride_ = sShape.size() == 2 ? ((sShape[1] + 3) & ~3u) : vocabStride_;
  outStride_ = (embDim_ + 3) & ~3u;

  numIndices_ = 1;
  for (auto d : idxShape)
    numIndices_ *= d;

  format_ = mBuf.getDtype() == DataType::Int8 ? 1 : 0;
  if (mBuf.getDtype() != DataType::Float16 && mBuf.getDtype() != DataType::Int8) {
    throw std::runtime_error(
        "embeddingCol: matrix must be Float16 or Int8");
  }

  outShape_ = idxShape;
  outShape_.push_back(embDim_);

  inputs_ = {indices, matrix, scales};
  output_ = preallocOutput ? preallocOutput
                           : store.createTensorEmpty(outputShape(), DataType::Float32);
}

DataType EmbeddingColOpNode::outputDtype() const {
  return DataType::Float32;
}

std::optional<std::vector<uint32_t>> EmbeddingColOpNode::shader() const {
  auto compiled = compiledEmbeddingCol(DataType::UInt32, DataType::Float32);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, format_);
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> EmbeddingColOpNode::outputShape() const {
  return outShape_;
}

ThreadSize EmbeddingColOpNode::dispatchSize() const {
  uint32_t total = numIndices_ * embDim_;
  uint32_t gridX = ((total + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> EmbeddingColOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t numIndices;
    uint32_t embDim;
    uint32_t vocabStride;
    uint32_t scaleStride;
    uint32_t outStride;
  } pc{numIndices_, embDim_, vocabStride_, scaleStride_, outStride_};
  return toBytes(pc);
}

// --- PadOpNode ---

PadOpNode::PadOpNode(TensorStore &store,
                     const Tensor &input,
                     std::vector<uint32_t> &&padWidths,
                     float value,
                     std::optional<uint32_t> spec)
    : OpNode(Pad, store, spec) {
  const auto &buf = store.getTensor(input);
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
  output_ = store.createTensorEmpty(outputShape(), outputDtype());
}

DataType PadOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> PadOpNode::shader() const {
  auto compiled = compiledPad(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
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

// --- ExpandOpNode ---

ExpandOpNode::ExpandOpNode(TensorStore &store,
                           const Tensor &src,
                           const std::vector<uint32_t> &targetShape,
                           std::optional<uint32_t> spec)
    : OpNode(Expand, store, spec) {
  const auto &buf = store.getTensor(src);
  const auto srcShape = buf.getShape();
  dtype_ = buf.getDtype();
  int ndim = static_cast<int>(targetShape.size());

  if (ndim < 1 || ndim > 4) {
    throw std::runtime_error("expand: only 1-4 dimensions supported");
  }
  if (static_cast<int>(srcShape.size()) != ndim) {
    throw std::runtime_error(
        "expand: input ndim (" + std::to_string(srcShape.size()) +
        ") must match target ndim (" + std::to_string(ndim) + ")");
  }

  outShape_ = targetShape;
  for (int i = 0; i < ndim; ++i) {
    if (srcShape[i] != 1 && srcShape[i] != targetShape[i]) {
      throw std::runtime_error("expand: input dim " + std::to_string(i) +
                               " is " + std::to_string(srcShape[i]) +
                               ", must be 1 or match target " +
                               std::to_string(targetShape[i]));
    }
  }

  totalOutputElements_ = 1;
  for (auto d : outShape_)
    totalOutputElements_ *= d;

  std::memset(&params_, 0, sizeof(params_));
  params_.ndim = static_cast<uint32_t>(ndim);
  params_.totalElements = totalOutputElements_;
  for (int i = 0; i < ndim; ++i) {
    params_.inShape[i] = srcShape[i];
    params_.outShape[i] = outShape_[i];
  }

  inputs_ = {src};
  output_ = store.createTensorEmpty(outShape_, dtype_);
}

DataType ExpandOpNode::outputDtype() const {
  return dtype_;
}

std::optional<std::vector<uint32_t>> ExpandOpNode::shader() const {
  auto compiled = compiledExpand(dtype_, dtype_);
  if (compiled.has_value()) {
    auto spirv = std::move(compiled.value());
    patchSpecConstant(spirv, 1, static_cast<uint32_t>(op_));
    return spirv;
  }
  return std::nullopt;
}

std::vector<uint32_t> ExpandOpNode::outputShape() const {
  return outShape_;
}

ThreadSize ExpandOpNode::dispatchSize() const {
  uint32_t gridX = ((totalOutputElements_ + 255) / 256) * 256;
  return {gridX, 1, 1};
}

std::vector<uint8_t> ExpandOpNode::pushConstants() const {
  return toBytes(params_);
}

// --- SliceOpNode ---

SliceOpNode::SliceOpNode(TensorStore &store,
                         const Tensor &src,
                         uint32_t dim,
                         uint32_t start,
                         uint32_t end)
    : OpNode(Copy, store) { // Reuse Copy enum — this node never dispatches
  const auto &buf = store.getTensor(src);
  dtype_ = buf.getDtype();
  auto srcShape = buf.getShape();

  if (dim != 0 || srcShape.size() != 1) {
    throw std::runtime_error(
        "SliceOpNode currently only supports 1D tensors along dim 0");
  }
  if (start >= end || end > srcShape[0]) {
    throw std::runtime_error(
        "SliceOpNode: invalid range [" + std::to_string(start) + ", " +
        std::to_string(end) + ") for shape " + std::to_string(srcShape[0]));
  }

  uint32_t sliceLen = end - start;
  outShape_ = {sliceLen};

  // Compute byte offset. The source tensor's innermost dim is aligned to
  // multiple of 4 elements. For a 1D tensor, the aligned stride is
  // (shape[0] + 3) & ~3. The byte offset for element `start` is:
  size_t elemSize = dataTypeSize(buf.getDtype());
  byteOffset_ = static_cast<size_t>(start) * elemSize;

  inputs_ = {src};
  // Output will be created as a view during graph execution (not here,
  // because the parent tensor may be an arena-planned view itself).
  // For now, allocate a placeholder that gets replaced at execution time.
  output_ = store.createTensorEmpty(outShape_, dtype_);
}

} // namespace cut
