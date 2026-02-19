#pragma once

#include "OpNode.h"

namespace cut {

class TransposeOpNode : public OpNode {
public:
  TransposeOpNode(std::vector<uint32_t> shape, DataType dtype)
      : shape_(std::move(shape)), dtype_(dtype) {
    M_ = shape_[0];
    N_ = shape_[1];
  }

  void validate() const override {
    if (shape_.size() != 2) {
      throw std::runtime_error("transpose requires a 2D matrix");
    }
  }

  OperatorEnum op() const override { return Transpose; }
  DataType shaderDtype() const override { return dtype_; }
  DataType outputDtype() const override { return DataType::Float32; }

  std::vector<uint32_t> outputShape() const override { return {N_, M_}; }

  ThreadSize dispatchSize() const override {
    const uint32_t tileSize = 16;
    uint32_t strideOut = (M_ + 3) & ~3u;
    uint32_t strideOut4 = strideOut / 4;
    uint32_t gridX = (N_ + tileSize - 1) / tileSize * tileSize;
    uint32_t gridY = (strideOut4 + tileSize - 1) / tileSize * tileSize;
    return {gridX, gridY, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    uint32_t strideIn = (N_ + 3) & ~3u;
    uint32_t strideOut = (M_ + 3) & ~3u;
    struct PushConstants {
      uint32_t M, N, strideIn, strideOut;
    } pc{M_, N_, strideIn, strideOut};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> shape_;
  DataType dtype_;
  uint32_t M_, N_;
};

class CopyOpNode : public OpNode {
public:
  CopyOpNode(std::vector<uint32_t> srcShape,
             std::vector<uint32_t> dstShape,
             DataType dtype)
      : srcShape_(std::move(srcShape)), dstShape_(std::move(dstShape)),
        dtype_(dtype) {
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

  void validate() const override {}

  OperatorEnum op() const override { return Copy; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return dstShape_; }

  ThreadSize dispatchSize() const override { return {totalElements_, 1, 1}; }

  std::vector<uint8_t> pushConstants() const override {
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

private:
  std::vector<uint32_t> srcShape_;
  std::vector<uint32_t> dstShape_;
  DataType dtype_;
  uint32_t srcInner_, srcAlignedInner_;
  uint32_t dstInner_, dstAlignedInner_;
  uint32_t totalElements_;
};

class EmbeddingOpNode : public OpNode {
public:
  EmbeddingOpNode(std::vector<uint32_t> idxShape,
                  std::vector<uint32_t> wShape,
                  DataType weightDtype)
      : idxShape_(std::move(idxShape)), wShape_(std::move(wShape)),
        dtype_(weightDtype) {
    embDim_ = wShape_[1];
    numIndices_ = 1;
    for (auto d : idxShape_)
      numIndices_ *= d;
    outShape_ = idxShape_;
    outShape_.push_back(embDim_);
  }

  void validate() const override {
    if (wShape_.size() != 2) {
      throw std::runtime_error(
          "embedding: weight must be 2D [num_embeddings, embedding_dim]");
    }
  }

  OperatorEnum op() const override { return Embedding; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return outShape_; }

  ThreadSize dispatchSize() const override {
    uint32_t alignedDim4 = ((embDim_ + 3) & ~3u) / 4;
    uint32_t totalOutputs = numIndices_ * alignedDim4;
    uint32_t gridX = ((totalOutputs + 255) / 256) * 256;
    return {gridX, 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    struct PushConstants {
      uint32_t numIndices;
      uint32_t embDim;
    } pc{numIndices_, embDim_};
    return toBytes(pc);
  }

private:
  std::vector<uint32_t> idxShape_;
  std::vector<uint32_t> wShape_;
  DataType dtype_;
  uint32_t embDim_;
  uint32_t numIndices_;
  std::vector<uint32_t> outShape_;
};

class PadOpNode : public OpNode {
public:
  PadOpNode(std::vector<uint32_t> shape,
            std::vector<uint32_t> padWidths,
            float value,
            DataType dtype)
      : shape_(std::move(shape)), padWidths_(std::move(padWidths)),
        value_(value), dtype_(dtype) {
    int ndim = static_cast<int>(shape_.size());
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

  void validate() const override {
    int ndim = static_cast<int>(shape_.size());
    if (padWidths_.size() % 2 != 0 ||
        static_cast<int>(padWidths_.size() / 2) > ndim) {
      throw std::runtime_error("pad: invalid padWidths length");
    }
  }

  OperatorEnum op() const override { return Pad; }
  DataType shaderDtype() const override { return dtype_; }

  std::vector<uint32_t> outputShape() const override { return outShape_; }

  ThreadSize dispatchSize() const override {
    uint32_t gridX = ((totalOutputElements_ + 255) / 256) * 256;
    return {gridX, 1, 1};
  }

  std::vector<uint8_t> pushConstants() const override {
    return toBytes(params_);
  }

private:
  struct PadParams {
    uint32_t ndim;
    uint32_t inShape[4];
    uint32_t outShape[4];
    uint32_t padBefore[4];
    uint32_t totalElements;
    float fillValue;
  };

  std::vector<uint32_t> shape_;
  std::vector<uint32_t> padWidths_;
  float value_;
  DataType dtype_;
  std::vector<uint32_t> outShape_;
  uint32_t totalOutputElements_;
  PadParams params_;
};

} // namespace cut
