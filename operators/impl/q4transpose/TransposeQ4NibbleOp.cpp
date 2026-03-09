#include "TransposeQ4NibbleOp.h"
#include "Q4transposeShaders.generated.h"
#include "TensorStore.h"

namespace cut {

TransposeQ4NibbleOpNode::TransposeQ4NibbleOpNode(TensorStore &store,
                                                 const Tensor &packedInput,
                                                 uint32_t N,
                                                 uint32_t K,
                                                 std::optional<uint32_t> spec)
    : OpNode(TransposeQ4, store, spec), N_(N), K_(K) {
  if (N_ % 2 != 0) {
    throw std::runtime_error("TransposeQ4: N must be even");
  }
  if (K_ % 32 != 0) {
    throw std::runtime_error("TransposeQ4: K must be a multiple of 32");
  }
  strideInHalf_ = (K_ / 2 + 3) & ~static_cast<uint32_t>(3);
  strideOutHalf_ = (N_ / 2 + 3) & ~static_cast<uint32_t>(3);
  inputs_ = {packedInput};
  output_ = store.createTensorEmpty({K_, N_ / 2}, DataType::Int8);
}

DataType TransposeQ4NibbleOpNode::outputDtype() const {
  return DataType::Int8;
}

size_t TransposeQ4NibbleOpNode::shaderKey() const {
  return static_cast<size_t>(op_);
}

std::optional<std::vector<uint32_t>> TransposeQ4NibbleOpNode::shader() const {
  return compiledTransposeQ4Nibble(DataType::Int8, DataType::Int8);
}

std::vector<uint32_t> TransposeQ4NibbleOpNode::outputShape() const {
  return {K_, N_ / 2};
}

ThreadSize TransposeQ4NibbleOpNode::dispatchSize() const {
  // [numthreads(64, 4, 1)]
  // X covers uint32 words per output row, Y covers output rows (K)
  uint32_t halfN = N_ / 2;
  uint32_t wordsPerRow = (halfN + 3) / 4;
  uint32_t gridX = ((wordsPerRow + 63) / 64) * 64;
  uint32_t gridY = ((K_ + 3) / 4) * 4;
  return {gridX, gridY, 1};
}

std::vector<uint8_t> TransposeQ4NibbleOpNode::pushConstants() const {
  struct PushConstants {
    uint32_t N;
    uint32_t K;
    uint32_t strideInHalf;
    uint32_t strideOutHalf;
  } pc{N_, K_, strideInHalf_, strideOutHalf_};
  return toBytes(pc);
}

std::string TransposeQ4NibbleOpNode::displayName() const {
  return "TransposeQ4";
}

} // namespace cut
