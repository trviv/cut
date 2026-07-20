#include <gtest/gtest.h>
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <SharedRuntime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cut {
namespace {

inline uint16_t floatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent == 128) { return sign | 0x7C00 | (mantissa ? (mantissa >> 13) | 1 : 0); }
  if (exponent < -14) { return sign; }
  if (exponent > 15) { return sign | 0x7C00; }
  return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}
inline float halfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) { bits = sign; }
    else { exponent = 1; while (!(mantissa & 0x0400)) { mantissa <<= 1; exponent--; }
           mantissa &= 0x03FF; bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13); }
  } else if (exponent == 31) { bits = sign | 0x7F800000 | (mantissa << 13); }
  else { bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13); }
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}

class OpCoverageTest : public ::testing::Test {
protected:
  void SetUp() override {
    rt_ = test::sharedRuntime();
    if (!rt_) GTEST_SKIP() << "No compute backend available";
  }
  void TearDown() override { if (rt_) rt_->flush(); }
  Runtime *rt_ = nullptr;
};

TEST_F(OpCoverageTest, Cast_F32_to_F16) {
  // input values all exactly representable in f16 (no rounding)
  std::vector<float> in = {0.f,1.f,-1.f,2.f,0.5f,-0.5f,1.5f,2.5f,4.f,8.f,0.25f,100.f};
  auto a = rt_->createTensor({(uint32_t)in.size()}, DataType::Float32, in.data());
  auto out = rt_->ops().cast(a, DataType::Float16);
  std::vector<uint16_t> got(in.size());
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(uint16_t));
  for (size_t i=0;i<in.size();++i) ASSERT_FLOAT_EQ(halfToFloat(got[i]), in[i]) << "i="<<i;
}

TEST_F(OpCoverageTest, Cast_F16_to_F32) {
  std::vector<float> vals = {0.f,1.f,-2.f,0.5f,3.5f,-4.f,0.25f,16.f};
  std::vector<uint16_t> in(vals.size());
  for (size_t i=0;i<vals.size();++i) in[i]=floatToHalf(vals[i]);
  auto a = rt_->createTensor({(uint32_t)in.size()}, DataType::Float16, in.data());
  auto out = rt_->ops().cast(a, DataType::Float32);
  std::vector<float> got(vals.size());
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (size_t i=0;i<vals.size();++i) ASSERT_FLOAT_EQ(got[i], halfToFloat(in[i])) << "i="<<i;
}

TEST_F(OpCoverageTest, Cast_I32_to_F32) {
  std::vector<int32_t> in = {0,1,-1,5,-7,100,-128,1000};
  auto a = rt_->createTensor({(uint32_t)in.size()}, DataType::Int32, in.data());
  auto out = rt_->ops().cast(a, DataType::Float32);
  std::vector<float> got(in.size());
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (size_t i=0;i<in.size();++i) ASSERT_FLOAT_EQ(got[i], (float)in[i]) << "i="<<i;
}

TEST_F(OpCoverageTest, Cast_F32_to_I32_Truncates) {
  std::vector<float> in = {0.f,1.7f,-1.7f,2.9f,-2.9f,3.0f,10.5f,-10.5f};
  auto a = rt_->createTensor({(uint32_t)in.size()}, DataType::Float32, in.data());
  auto out = rt_->ops().cast(a, DataType::Int32);
  std::vector<int32_t> got(in.size());
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(int32_t));
  for (size_t i=0;i<in.size();++i) ASSERT_EQ(got[i], (int32_t)in[i]) << "i="<<i;
}

TEST_F(OpCoverageTest, Expand_2D_BroadcastRows) {
  // {1,4} -> {3,4}: each output row equals the single input row
  std::vector<float> in = {1.f,2.f,3.f,4.f};
  auto a = rt_->createTensor({1,4}, DataType::Float32, in.data());
  auto out = rt_->ops().expand(a, {3,4});
  ASSERT_EQ(rt_->getTensor(out).getShape(), (std::vector<uint32_t>{3,4}));
  std::vector<float> got(12);
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (uint32_t r=0;r<3;++r) for (uint32_t c=0;c<4;++c)
    ASSERT_FLOAT_EQ(got[r*4+c], in[c]) << "r="<<r<<" c="<<c;
}

TEST_F(OpCoverageTest, Expand_2D_BroadcastCols) {
  // {3,1} -> {3,4}: each output row is the input scalar repeated
  std::vector<float> in = {5.f,6.f,7.f};
  auto a = rt_->createTensor({3,1}, DataType::Float32, in.data());
  auto out = rt_->ops().expand(a, {3,4});
  std::vector<float> got(12);
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (uint32_t r=0;r<3;++r) for (uint32_t c=0;c<4;++c)
    ASSERT_FLOAT_EQ(got[r*4+c], in[r]) << "r="<<r<<" c="<<c;
}

TEST_F(OpCoverageTest, Expand_3D_BroadcastBatch) {
  // {1,2,4} -> {3,2,4}
  std::vector<float> in = {1,2,3,4, 5,6,7,8};
  auto a = rt_->createTensor({1,2,4}, DataType::Float32, in.data());
  auto out = rt_->ops().expand(a, {3,2,4});
  std::vector<float> got(24);
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (uint32_t b=0;b<3;++b) for (uint32_t i=0;i<8;++i)
    ASSERT_FLOAT_EQ(got[b*8+i], in[i]) << "b="<<b<<" i="<<i;
}

TEST_F(OpCoverageTest, Flatten_Partial) {
  // {2,3,4} flatten dims [1,2] -> {2,12}, row-major data preserved
  std::vector<float> in(24); for (int i=0;i<24;++i) in[i]=(float)(i+1);
  auto a = rt_->createTensor({2,3,4}, DataType::Float32, in.data());
  auto out = rt_->ops().flatten(a, 1, 2);
  ASSERT_EQ(rt_->getTensor(out).getShape(), (std::vector<uint32_t>{2,12}));
  std::vector<float> got(24);
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (int i=0;i<24;++i) ASSERT_FLOAT_EQ(got[i], in[i]) << "i="<<i;
}

TEST_F(OpCoverageTest, Flatten_Full) {
  // {2,3,4} flatten [0,-1] -> {24}
  std::vector<float> in(24); for (int i=0;i<24;++i) in[i]=(float)(i+1);
  auto a = rt_->createTensor({2,3,4}, DataType::Float32, in.data());
  auto out = rt_->ops().flatten(a, 0, -1);
  ASSERT_EQ(rt_->getTensor(out).getShape(), (std::vector<uint32_t>{24}));
  std::vector<float> got(24);
  rt_->copyFromTensor(out, got.data(), got.size()*sizeof(float));
  for (int i=0;i<24;++i) ASSERT_FLOAT_EQ(got[i], in[i]) << "i="<<i;
}

} // namespace
} // namespace cut
