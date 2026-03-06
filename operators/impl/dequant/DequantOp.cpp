#include "DequantOp.h"
#include "DequantShaders.generated.h"
#include "TensorStore.h"

namespace cut {

DequantOpNode::DequantOpNode(TensorStore &store,
                             const Tensor &rawData,
                             DequantFormat format,
                             uint32_t rows,
                             uint32_t cols,
                             std::optional<uint32_t> spec)
    : OpNode(Dequantize, store, spec), format_(format), rows_(rows),
      cols_(cols) {
  outputStride_ = (cols_ + 3) & ~static_cast<uint32_t>(3);
  inputs_ = {rawData};
  output_ = store.createTensorEmpty({rows_, cols_}, DataType::Float32);
}

DataType DequantOpNode::outputDtype() const {
  return DataType::Float32;
}

size_t DequantOpNode::shaderKey() const {
  size_t key = static_cast<size_t>(op_);
  key |= (static_cast<size_t>(format_) & 0xF) << 16;
  return key;
}

std::optional<std::vector<uint32_t>> DequantOpNode::shader() const {
  switch (format_) {
  case DequantFormat::BF16:
    return compiledDequantBF16(DataType::Int8, DataType::Float32);
  case DequantFormat::Q4_K:
    return compiledDequantQ4K(DataType::Int8, DataType::Float32);
  case DequantFormat::Q5_K:
    return compiledDequantQ5K(DataType::Int8, DataType::Float32);
  case DequantFormat::Q6_K:
    return compiledDequantQ6K(DataType::Int8, DataType::Float32);
  }
  return std::nullopt;
}

std::vector<uint32_t> DequantOpNode::outputShape() const {
  return {rows_, cols_};
}

ThreadSize DequantOpNode::dispatchSize() const {
  // 2D dispatch: X covers cols, Y covers rows.
  // Avoids expensive integer division (gid/cols, gid%cols) in shader.
  return {cols_, rows_, 1};
}

std::vector<uint8_t> DequantOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t rows;
    uint32_t cols;
    uint32_t outputStride;
  } pc{rows_, cols_, outputStride_};
  return toBytes(pc);
}

std::string DequantOpNode::displayName() const {
  return "Dequantize";
}

} // namespace cut
