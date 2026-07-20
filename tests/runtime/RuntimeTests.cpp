#include <gtest/gtest.h>

#include "impl/avgpool2d/AvgPool2DVariants.generated.h"
#include "impl/conv1d/Conv1DVariants.generated.h"
#include "impl/conv2d/Conv2DVariants.generated.h"
#include "impl/dequant/DequantOp.h"
#include "impl/matmul/MatMulQ4Variants.generated.h"
#include "impl/matmul/MatMulQ8Variants.generated.h"
#include "impl/matmul/MatMulVariants.generated.h"
#include "impl/maxpool2d/MaxPool2DVariants.generated.h"
#include "impl/reducedim/ReduceDimVariants.generated.h"
#include "impl/transpose/TransposeVariants.generated.h"
#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <Operations.h>
#include <Runtime.h>
#include <SharedRuntime.h>

#include "harness/OpRefs.h"
#include "harness/OpRegistry.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cut {
namespace {

// ============================================================================
// Test Configuration
// ============================================================================

// All dimension counts to test (1D, 2D, 3D, 4D)
constexpr std::array<size_t, 4> kDimensionCounts = {1, 2, 3, 4};

// Dimension sizes to test (1-17)
constexpr uint32_t kMinDimSize = 1;
constexpr uint32_t kMaxDimSize = 17;

// Representative dimension sizes for comprehensive testing
// Include odd values: 1, 3, 7, 9, 13, 17

// All data types

// All binary operators (vec-vec variant tests)

// Binary operators valid for integer types (i/uvec4)

// All binary operators (vec-scalar variant tests)

// All unary operators

// Ternary operators
constexpr std::array<OperatorEnum, 2> kTernaryOps = {TernaryClamp,
                                                     TernarySelect};

// Reduction operators
constexpr std::array<OperatorEnum, 7> kReductionOps = {
    ReduceSum,  ReduceMean, ReduceMin, ReduceMax,
    ReduceProd, ReduceAny,  ReduceAll};

// Matrix operators
constexpr std::array<OperatorEnum, 3> kMatrixOps = {MatMul, Transpose, Dot};

// ============================================================================
// Helper Functions
// ============================================================================

// dataTypeName is provided by ComputeCommon.h

inline const char *backendName(BackendType backend) {
  return backend == BackendType::Vulkan ? "Vulkan" : "Unknown";
}

// IEEE 754 half-precision (float16) conversion utilities

// Helper to check if an operator has Vulkan shader support
// (New operators not yet implemented in shaders)
inline bool hasVulkanShaderSupport(OperatorEnum op) {
  switch (op) {
  // Binary operators with shader support
  case BinaryAdd:
  case BinarySub:
  case BinaryMul:
  case BinaryDiv:
  case BinaryMod:
  case BinaryPow:
  case BinaryFloorDiv:
  case BinaryEqual:
  case BinaryNotEqual:
  case BinaryLess:
  case BinaryLessEqual:
  case BinaryGreater:
  case BinaryGreaterEqual:
  case BinaryMin:
  case BinaryMax:
  case UnaryNeg:
  case UnaryAbs:
  case UnarySqrt:
  case UnarySquare:
  case UnaryReciprocal:
  case UnarySign:
  case UnaryExp:
  case UnaryLog:
  case UnaryLog2:
  case UnaryLog10:
  case UnarySin:
  case UnaryCos:
  case UnaryTan:
  case UnaryAsin:
  case UnaryAcos:
  case UnaryAtan:
  case UnarySinh:
  case UnaryCosh:
  case UnaryTanh:
  case UnaryFloor:
  case UnaryCeil:
  case UnaryRound:
  // Extended binary operators (bitwise, logical, special math)
  case BinaryBitwiseAnd:
  case BinaryBitwiseOr:
  case BinaryBitwiseXor:
  case BinaryLeftShift:
  case BinaryRightShift:
  case BinaryLogicalAnd:
  case BinaryLogicalOr:
  case BinaryLogicalXor:
  case BinaryAtan2:
  case BinaryHypot:
  case BinaryCopysign:
  case BinaryFmod:
  case BinaryLeakyRelu:
  // Extended unary operators
  case UnaryExpm1:
  case UnaryExp2:
  case UnaryLog1p:
  case UnaryCbrt:
  case UnaryDegrees:
  case UnaryRadians:
  case UnaryLogicalNot:
  case UnaryBitwiseNot:
  case UnaryRelu:
  case UnarySigmoid:
  case UnaryGelu:
  case UnarySilu:
  case UnarySoftplus:
  case UnaryIsNan:
  case UnaryIsInf:
  // Ternary operators
  case TernaryClamp:
  case TernarySelect:
  // Reduction operations
  case ReduceSum:
  case ReduceMean:
  case ReduceMin:
  case ReduceMax:
  case ReduceProd:
  case ReduceAny:
  case ReduceAll:
  // Matrix operations
  case MatMul:
  case Transpose:
  case Dot:
  // Tensor creation operations
  case Arange:
  case Linspace:
  case Zeros:
  case Ones:
  case Full:
  // Norm operations
  case Norm:
  // Extended unary activations (Phase 1)
  case UnaryRelu6:
  case UnaryElu:
  case UnarySelu:
  case UnaryCelu:
  case UnaryMish:
  case UnaryHardswish:
  case UnaryHardsigmoid:
  case UnaryHardtanh:
  case UnarySoftsign:
  case UnaryLogSigmoid:
  case UnaryTanhshrink:
  // Extended unary math (Phase 2)
  case UnaryRsqrt:
  case UnaryTrunc:
  case UnaryFrac:
  case UnaryAsinh:
  case UnaryAcosh:
  case UnaryAtanh:
  case UnaryIsFinite:
  // Extended binary operators (activations and log-sum-exp)
  case BinaryPrelu:
  case BinaryHardshrink:
  case BinarySoftshrink:
  case BinaryLogaddexp:
  case BinaryLogaddexp2:
  // Argmax/Argmin reductions
  case ReduceArgmax:
  case ReduceArgmin:
  // Cumulative scan operations
  case CumSum:
  case CumProd:
  // Prefix scan operations
  case PrefixScanExclusiveSum:
  case PrefixScanInclusiveSum:
  // Sort operations
  case SortBitonic:
  case SortRadix:
  // Dim-wise reductions
  case NormDim:
    return true;
  // Operators without shader support yet
  default:
    return false;
  }
}

// Generate shapes for testing given dimension count
// For multi-dimensional shapes, innermost dimension must be multiple of 4
// to avoid buffer padding issues with the current kernel implementation

// ============================================================================
// Reference Implementations for Operators
// ============================================================================






// ============================================================================
// Test Data Generation
// ============================================================================


// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class RuntimeOperatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = test::sharedRuntime();
    if (!runtime_) {
      GTEST_SKIP() << "Vulkan not available on this system";
    }
  }

  void TearDown() override { runtime_->flush(); }

  void initBackend(BackendType /* backend */) {
    // No-op: shared runtime is already initialized
  }

  Runtime *runtime_ = nullptr;
};

// ============================================================================
// Vulkan Backend Tests - All Operators
// ============================================================================

class VulkanBackendTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Test binary vec-vec operators across all data types.
// Retrofitted to drive the shared op-case registry (family "binary_vecvec"),
// so the correctness sweep and the op_bench perf path share one definition.
TEST_F(VulkanBackendTest, BinaryVecVecOperators) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "binary_vecvec")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    ASSERT_TRUE(static_cast<bool>(c.verify));
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0) << "no binary_vecvec cases found in registry";
}

// Test unary operators with Float32
TEST_F(VulkanBackendTest, UnaryOperators_Float32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "unary" || c.name.find("/f32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test unary operators with Int32 (registry-driven).
TEST_F(VulkanBackendTest, UnaryOperators_Int32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "unary" || c.name.find("/i32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test binary vec-scalar operators with Float32 (registry-driven).
TEST_F(VulkanBackendTest, BinaryVecScalarOperators_Float32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "binary_vecscalar" ||
        c.name.find("/f32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Cross-Backend Consistency Tests
// ============================================================================

class DimensionSizeRangeTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(DimensionSizeRangeTest, AllSizes_1D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  for (uint32_t size = kMinDimSize; size <= kMaxDimSize; ++size) {
    std::vector<uint32_t> shape = {size};
    const uint32_t elements = size;
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Size: " + std::to_string(size));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for size " << size;
    }
  }
}

// Test 2D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_2D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  // Outer dimension (h) can be any odd value, inner dimension (w) must be
  // multiple of 4
  constexpr std::array<uint32_t, 6> outerSizes = {1, 3, 7, 9, 13, 17};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t h : outerSizes) {
    for (uint32_t w : innerSizes) {
      std::vector<uint32_t> shape = {h, w};
      const uint32_t elements = h * w;
      const size_t bufferSize = elements * sizeof(float);

      SCOPED_TRACE("Shape: " + std::to_string(h) + "x" + std::to_string(w));

      auto dataA = generateTestData<float>(elements, 42);
      auto dataB = generateTestData<float>(elements, 123);

      auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
      auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

      auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        float expected = dataA[i] * dataB[i];
        ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
      }
    }
  }
}

// Test 3D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_3D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinarySub;

  // Outer dimensions can be odd values, innermost must be multiple of 4
  constexpr std::array<uint32_t, 4> outerSizes = {1, 3, 7, 9};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t d : outerSizes) {
    for (uint32_t h : outerSizes) {
      for (uint32_t w : innerSizes) {
        std::vector<uint32_t> shape = {d, h, w};
        const uint32_t elements = d * h * w;
        const size_t bufferSize = elements * sizeof(float);

        SCOPED_TRACE("Shape: " + std::to_string(d) + "x" + std::to_string(h) +
                     "x" + std::to_string(w));

        auto dataA = generateTestData<float>(elements, 42);
        auto dataB = generateTestData<float>(elements, 123);

        auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
        auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

        auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

        std::vector<float> output(elements);
        runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

        for (uint32_t i = 0; i < elements; ++i) {
          float expected = dataA[i] - dataB[i];
          ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
        }
      }
    }
  }
}

// Test 4D shapes with various outer dimensions and aligned inner dimension
TEST_F(DimensionSizeRangeTest, AllSizes_4D_Float32) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryDiv;

  // Outer dimensions can be odd values, innermost must be multiple of 4
  constexpr std::array<uint32_t, 3> outerSizes = {1, 3, 7};
  constexpr std::array<uint32_t, 4> innerSizes = {4, 8, 12, 16};
  for (uint32_t n : outerSizes) {
    for (uint32_t c : outerSizes) {
      for (uint32_t h : outerSizes) {
        for (uint32_t w : innerSizes) {
          std::vector<uint32_t> shape = {n, c, h, w};
          const uint32_t elements = n * c * h * w;
          const size_t bufferSize = elements * sizeof(float);

          SCOPED_TRACE("Shape: " + std::to_string(n) + "x" + std::to_string(c) +
                       "x" + std::to_string(h) + "x" + std::to_string(w));

          auto dataA = generateTestData<float>(elements, 42);
          auto dataB = generateTestData<float>(elements, 123);
          // Avoid division by zero
          for (auto &v : dataB) {
            if (v < 0.1f)
              v = 0.1f;
          }

          auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
          auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

          auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

          std::vector<float> output(elements);
          runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

          for (uint32_t i = 0; i < elements; ++i) {
            float expected = dataA[i] / dataB[i];
            ASSERT_NEAR(output[i], expected, 1e-4f)
                << "Mismatch at index " << i;
          }
        }
      }
    }
  }
}

// ============================================================================
// Non-Aligned Innermost Dimension Tests
// ============================================================================

// Tests for multi-dimensional shapes where innermost dimension is NOT a
// multiple of 4 (1, 3, 5, 11, 13). These tests verify that the
// calculateAlignedElements function properly handles alignment.

class NonAlignedInnermostTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Test 2D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim1) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 1};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim3) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 3};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim5) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinarySub;

  for (uint32_t outer : {2u, 5u, 7u}) {
    std::vector<uint32_t> shape = {outer, 5};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] - dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim11) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  for (uint32_t outer : {2u, 3u, 5u}) {
    std::vector<uint32_t> shape = {outer, 11};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(NonAlignedInnermostTest, BinaryVecVec_2D_InnermostDim13) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  for (uint32_t outer : {2u, 3u, 5u}) {
    std::vector<uint32_t> shape = {outer, 13};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test 3D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_3D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] + dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test 4D shapes with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, BinaryVecVec_4D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test unary operators with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, Unary_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = UnaryNeg;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {4, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataIn = generateTestData<float>(elements, 42);

    auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());

    auto bufferOut = runtime_->ops().unaryOp(op, bufferIn);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = -dataIn[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test vec-scalar operators with non-aligned innermost dimensions
TEST_F(NonAlignedInnermostTest, VecScalar_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;
  const float scalar = 2.5f;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {4, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalar);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * scalar;
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Vulkan backend tests with non-aligned innermost dimensions
class VulkanNonAlignedInnermostTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_2D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    for (uint32_t outer : {2u, 3u, 5u}) {
      std::vector<uint32_t> shape = {outer, innerDim};
      const uint32_t elements = totalElements(shape);
      const size_t bufferSize = elements * sizeof(float);

      SCOPED_TRACE("Shape: " + shapeToString(shape));

      auto dataA = generateTestData<float>(elements, 42);
      auto dataB = generateTestData<float>(elements, 123);

      auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
      auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

      auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

      std::vector<float> output(elements);
      runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

      for (uint32_t i = 0; i < elements; ++i) {
        float expected = dataA[i] + dataB[i];
        ASSERT_NEAR(output[i], expected, 1e-5f)
            << "Mismatch at index " << i << " for shape "
            << shapeToString(shape);
      }
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_3D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] * dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, BinaryVecVec_4D_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinarySub;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {2, 2, 3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    auto dataB = generateTestData<float>(elements, 123);

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto bufferB = runtime_->createTensor(shape, dtype, dataB.data());

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, bufferB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataA[i] - dataB[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(VulkanNonAlignedInnermostTest, Unary_NonAlignedInnermost) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = UnarySquare;

  // Test innermost dimensions 1, 3, 5, 11, 13
  for (uint32_t innerDim : {1u, 3u, 5u, 11u, 13u}) {
    std::vector<uint32_t> shape = {3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataIn = generateTestData<float>(elements, 42);

    auto bufferIn = runtime_->createTensor(shape, dtype, dataIn.data());

    auto bufferOut = runtime_->ops().unaryOp(op, bufferIn);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = dataIn[i] * dataIn[i];
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

// Test reduction operators with Float32 on Vulkan
TEST_F(VulkanBackendTest, ReductionOperators_Float32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "reduce" || c.name.find("/f32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test ternary clamp operator with Float32 on Vulkan
TEST_F(VulkanBackendTest, TernaryClamp_Float32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("clamp/f32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test ternary select operator with Float32 on Vulkan (registry-driven).
TEST_F(VulkanBackendTest, TernarySelect_Float32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("select/f32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Dimension-wise Reduction Tests
// ============================================================================

// Dim reduction operators


// Test dim reduction operators on 2D tensors reducing along dim 0
TEST_F(VulkanBackendTest, DimReductionOperators_2D_Dim0) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "dimreduce" || c.name.find("/2d_dim0") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test dim reduction operators on 2D tensors reducing along dim 1
TEST_F(VulkanBackendTest, DimReductionOperators_2D_Dim1) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "dimreduce" || c.name.find("/2d_dim1") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test dim reduction on 3D tensor reducing along the middle dimension
TEST_F(VulkanBackendTest, DimReductionOperators_3D_MiddleDim) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "dimreduce" || c.name.find("/3d_mid") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Dimension-wise Norm Tests
// ============================================================================

// Test NormDim on 2D tensors reducing along dim 0
TEST_F(VulkanBackendTest, NormDim_2D_Dim0) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "normdim/2d_dim0")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test NormDim on 2D tensors reducing along dim 1
TEST_F(VulkanBackendTest, NormDim_2D_Dim1) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "normdim/2d_dim1")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test NormDim on 3D tensor reducing along middle dimension
TEST_F(VulkanBackendTest, NormDim_3D_MiddleDim) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "normdim/3d_mid")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// Test NormDim with known values (3-4-5 triangle)
TEST_F(VulkanBackendTest, NormDim_KnownValues) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "normdim/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Buffer View Tests (Runtime Level)
// ============================================================================

TEST_F(VulkanBackendTest, BufferView_BinaryOp) {
  constexpr uint32_t elements = 64;
  constexpr size_t bufferSize = elements * sizeof(float);

  // Create parent tensor with 128 floats of known data
  std::vector<float> parentData(128);
  for (uint32_t i = 0; i < 128; ++i) {
    parentData[i] = static_cast<float>(i);
  }
  auto parent =
      runtime_->createTensor({128}, DataType::Float32, parentData.data());

  // Create two views via TensorStore
  auto viewA = runtime_->store().createTensorView(parent, 0, {elements},
                                                  DataType::Float32);
  auto viewB = runtime_->store().createTensorView(parent, 256, {elements},
                                                  DataType::Float32);

  // Perform add: viewA + viewB -> result
  auto result = runtime_->ops().binaryOp(BinaryAdd, viewA, viewB);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(result, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(i + 64);
    EXPECT_FLOAT_EQ(expected, output[i]) << "Element " << i;
  }
}

TEST_F(VulkanBackendTest, BufferView_RefCounting) {
  auto parent = runtime_->createTensor({256}, DataType::Float32);
  auto view =
      runtime_->store().createTensorView(parent, 0, {64}, DataType::Float32);

  // Drop parent handle — view's parentHandle_ keeps parent alive
  parent.reset();
  EXPECT_TRUE(view);

  // View should still be usable for a binary op with another tensor
  std::vector<float> data(64, 1.0f);
  auto other = runtime_->createTensor({64}, DataType::Float32, data.data());
  auto result = runtime_->ops().binaryOp(BinaryAdd, view, other);

  std::vector<float> output(64);
  runtime_->copyFromTensor(result, output.data(), 64 * sizeof(float));

  // Result should not crash; values depend on uninitialized parent data + 1.0
  // Just verify we got valid output without crashing
  EXPECT_EQ(output.size(), 64u);

  // Drop view — parent ref count drops, both destroyed
  view.reset();
}

// ============================================================================
// Runtime Lifecycle Tests
// ============================================================================

class RuntimeLifecycleTest : public ::testing::Test {};

TEST_F(RuntimeLifecycleTest, DefaultConstruction) {
  Runtime runtime;
  ASSERT_EQ(runtime.currentBackend(), BackendType::Vulkan);
}

TEST_F(RuntimeLifecycleTest, VulkanInitialization) {
  Runtime runtime;
  if (runtime.isVulkanAvailable()) {
    ASSERT_NO_THROW(runtime.init(BackendType::Vulkan));
    ASSERT_EQ(runtime.currentBackend(), BackendType::Vulkan);
    runtime.shutdown();
  } else {
    ASSERT_THROW(runtime.init(BackendType::Vulkan), std::runtime_error);
  }
}

TEST_F(RuntimeLifecycleTest, MultipleInitShutdown) {
  Runtime runtime;

  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }

  // First cycle
  runtime.init(BackendType::Vulkan);
  {
    auto buf1 = runtime.createTensorEmpty({16}, DataType::Float32);
    ASSERT_TRUE(buf1);
  } // buf1 goes out of scope here
  runtime.shutdown();

  // Second cycle
  runtime.init(BackendType::Vulkan);
  {
    auto buf2 = runtime.createTensorEmpty({16}, DataType::Float32);
    ASSERT_TRUE(buf2);
  } // buf2 goes out of scope here
  runtime.shutdown();
}

// ============================================================================
// Argmax/Argmin Tests
// ============================================================================

class ArgmaxArgminTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(ArgmaxArgminTest, GlobalArgmax_Float32) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());

  auto outTensor = runtime_->ops().reduce(ReduceArgmax, bufferIn);
  float output = 0.0f;
  runtime_->copyFromTensor(outTensor, &output, sizeof(float));

  ASSERT_EQ(static_cast<int>(output), 3)
      << "Argmax should be index 3 (value 9.0)";
}

TEST_F(ArgmaxArgminTest, GlobalArgmin_Float32) {
  const DataType dtype = DataType::Float32;
  std::vector<float> data = {5.0f, 3.0f, 1.0f, 9.0f, 2.0f, 7.0f, 4.0f, 6.0f};
  const uint32_t elements = static_cast<uint32_t>(data.size());

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());

  auto outTensor = runtime_->ops().reduce(ReduceArgmin, bufferIn);
  float output = 0.0f;
  runtime_->copyFromTensor(outTensor, &output, sizeof(float));

  ASSERT_EQ(static_cast<int>(output), 2)
      << "Argmin should be index 2 (value 1.0)";
}

TEST_F(ArgmaxArgminTest, GlobalArgmax_LargeTensor) {
  const DataType dtype = DataType::Float32;
  const uint32_t elements = 1024;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  std::vector<float> data(elements);
  for (auto &v : data)
    v = dist(gen);

  // Find expected argmax
  int expectedIdx = 0;
  for (uint32_t i = 1; i < elements; ++i) {
    if (data[i] > data[expectedIdx])
      expectedIdx = static_cast<int>(i);
  }

  auto bufferIn = runtime_->createTensor({elements}, dtype, data.data());

  auto outTensor = runtime_->ops().reduce(ReduceArgmax, bufferIn);
  float output = 0.0f;
  runtime_->copyFromTensor(outTensor, &output, sizeof(float));

  ASSERT_EQ(static_cast<int>(output), expectedIdx);
}

TEST_F(ArgmaxArgminTest, DimArgmax_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // 3x4 matrix, find argmax along dim 0 (across rows)
  std::vector<float> data = {
      1.0f, 9.0f, 3.0f, 2.0f, // row 0
      7.0f, 4.0f, 8.0f, 5.0f, // row 1
      6.0f, 2.0f, 1.0f, 10.0f // row 2
  };
  std::vector<uint32_t> shape = {3, 4};

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());

  auto bufferOut = runtime_->ops().reduce(ReduceArgmax, bufferIn, 0);

  std::vector<float> output(4);
  runtime_->copyFromTensor(bufferOut, output.data(), 4 * sizeof(float));

  // Expected: col 0->row 1 (7), col 1->row 0 (9), col 2->row 1 (8), col 3->row
  // 2 (10)
  ASSERT_EQ(static_cast<int>(output[0]), 1);
  ASSERT_EQ(static_cast<int>(output[1]), 0);
  ASSERT_EQ(static_cast<int>(output[2]), 1);
  ASSERT_EQ(static_cast<int>(output[3]), 2);
}

TEST_F(ArgmaxArgminTest, DimArgmin_2D_Dim0) {
  const DataType dtype = DataType::Float32;
  // Same data as above
  std::vector<float> data = {1.0f, 9.0f, 3.0f, 2.0f, 7.0f, 4.0f,
                             8.0f, 5.0f, 6.0f, 2.0f, 1.0f, 10.0f};
  std::vector<uint32_t> shape = {3, 4};

  auto bufferIn = runtime_->createTensor(shape, dtype, data.data());

  auto bufferOut = runtime_->ops().reduce(ReduceArgmin, bufferIn, 0);

  std::vector<float> output(4);
  runtime_->copyFromTensor(bufferOut, output.data(), 4 * sizeof(float));

  // Expected: col 0->row 0 (1), col 1->row 2 (2), col 2->row 2 (1), col 3->row
  // 0 (2)
  ASSERT_EQ(static_cast<int>(output[0]), 0);
  ASSERT_EQ(static_cast<int>(output[1]), 2);
  ASSERT_EQ(static_cast<int>(output[2]), 2);
  ASSERT_EQ(static_cast<int>(output[3]), 0);
}

// ============================================================================
// Cumulative Scan Tests
// ============================================================================

class CumsumCumprodTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(CumsumCumprodTest, CumSum_1D) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_1d")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumProd_1D) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumprod_1d")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumSum_2D_Dim0) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_2d_dim0")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumProd_2D_Dim0) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumprod_2d_dim0")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumSum_2D_Dim1) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_2d_dim1")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumProd_2D_Dim1) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumprod_2d_dim1")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumSum_Large_1D_MultiPass) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_large_multipass")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumSum_Large_1D_ManyWorkgroups) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_large_manyworkgroups")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumSum_3D_AllDims) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumsum_3d_alldims")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(CumsumCumprodTest, CumProd_Large_1D_MultiPass) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "cumulative/cumprod_large_multipass")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Extended Activation & Math Shader Compilation Tests
// ============================================================================

class NewOpsShaderCompileTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(NewOpsShaderCompileTest, AllNewUnaryActivations_Compile) {
  const DataType dtype = DataType::Float32;
  constexpr std::array<OperatorEnum, 11> kNewUnaryActivations = {
      UnaryRelu6,    UnaryElu,        UnarySelu,        UnaryCelu,
      UnaryMish,     UnaryHardswish,  UnaryHardsigmoid, UnaryHardtanh,
      UnarySoftsign, UnaryLogSigmoid, UnaryTanhshrink};

  std::vector<float> dataIn = {-2.0f, -1.0f, 0.0f, 1.0f,
                               2.0f,  3.0f,  5.0f, 7.0f};
  const uint32_t elements = static_cast<uint32_t>(dataIn.size());
  const size_t bufferSize = elements * sizeof(float);

  auto bufferIn = runtime_->createTensor({elements}, dtype, dataIn.data());
  ComputeHandle bufferOut;

  for (OperatorEnum op : kNewUnaryActivations) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    ASSERT_NO_THROW({ bufferOut = runtime_->ops().unaryOp(op, bufferIn); });

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = unaryRef(op, dataIn[i]);
      if (std::isfinite(expected)) {
        ASSERT_NEAR(output[i], expected, 1e-3f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

TEST_F(NewOpsShaderCompileTest, AllNewUnaryMath_Compile) {
  const DataType dtype = DataType::Float32;
  constexpr std::array<OperatorEnum, 7> kNewUnaryMath = {
      UnaryRsqrt, UnaryTrunc, UnaryFrac,    UnaryAsinh,
      UnaryAcosh, UnaryAtanh, UnaryIsFinite};

  // Values suitable for all these functions (acosh requires >= 1, atanh
  // requires |x| < 1)
  std::vector<float> dataIn = {0.1f, 0.5f, 0.9f, 1.5f, 2.0f, 3.0f, 5.0f, 10.0f};
  const uint32_t elements = static_cast<uint32_t>(dataIn.size());
  const size_t bufferSize = elements * sizeof(float);

  auto bufferIn = runtime_->createTensor({elements}, dtype, dataIn.data());
  ComputeHandle bufferOut;

  for (OperatorEnum op : kNewUnaryMath) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    ASSERT_NO_THROW({ bufferOut = runtime_->ops().unaryOp(op, bufferIn); });

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = unaryRef(op, dataIn[i]);
      if (std::isfinite(expected)) {
        ASSERT_NEAR(output[i], expected, 1e-4f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

TEST_F(NewOpsShaderCompileTest, NewBinaryVecVec_Logaddexp) {
  const DataType dtype = DataType::Float32;
  std::vector<float> dataA = {1.0f,  2.0f, 3.0f, -1.0f,
                              -2.0f, 0.0f, 5.0f, 10.0f};
  std::vector<float> dataB = {2.0f, 1.0f, 3.0f, 0.0f, -3.0f, 0.0f, 4.0f, 9.0f};
  const uint32_t elements = static_cast<uint32_t>(dataA.size());
  const size_t bufferSize = elements * sizeof(float);

  auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
  auto bufB = runtime_->createTensor({elements}, dtype, dataB.data());
  ComputeHandle bufOut;

  for (OperatorEnum op : {BinaryLogaddexp, BinaryLogaddexp2}) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    bufOut = runtime_->ops().binaryOp(op, bufA, bufB);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = binaryVecVecRef(op, dataA[i], dataB[i]);
      ASSERT_NEAR(output[i], expected, 1e-4f)
          << "Mismatch at index " << i << " for " << operatorName(op);
    }
  }
}

TEST_F(NewOpsShaderCompileTest, NewBinaryVecScalar_ParameterizedActivations) {
  const DataType dtype = DataType::Float32;
  std::vector<float> dataA = {-3.0f, -1.0f, -0.5f, 0.0f,
                              0.5f,  1.0f,  2.0f,  3.0f};
  const uint32_t elements = static_cast<uint32_t>(dataA.size());
  const size_t bufferSize = elements * sizeof(float);
  const float scalar = 0.5f;

  auto bufA = runtime_->createTensor({elements}, dtype, dataA.data());
  ComputeHandle bufOut;

  for (OperatorEnum op : {BinaryPrelu, BinaryHardshrink, BinarySoftshrink,
                          BinaryLogaddexp, BinaryLogaddexp2}) {
    SCOPED_TRACE(std::string("Op: ") + operatorName(op));

    bufOut = runtime_->ops().binaryOp(op, bufA, scalar);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = binaryVecScalarRef(op, dataA[i], scalar);
      if (std::isfinite(expected)) {
        ASSERT_NEAR(output[i], expected, 1e-4f)
            << "Mismatch at index " << i << " for " << operatorName(op);
      }
    }
  }
}

// ============================================================================
// Multi-Workgroup Reduce Tests
// ============================================================================

class MultiWorkgroupReduceTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(MultiWorkgroupReduceTest, ReduceSum_LargeArray) {
  // Tests that trigger multi-workgroup reduce (>65536 elements)
  for (uint32_t elements : {257u, 1000u, 4096u, 65537u, 100000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());

    auto outTensor = runtime_->ops().reduce(ReduceSum, bufIn);
    float output = 0.0f;
    runtime_->copyFromTensor(outTensor, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceSum, data);
    ASSERT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-3f)
        << "ReduceSum mismatch for " << elements << " elements";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceMean_LargeArray) {
  for (uint32_t elements : {1000u, 65537u, 100000u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());

    auto outTensor = runtime_->ops().reduce(ReduceMean, bufIn);
    float output = 0.0f;
    runtime_->copyFromTensor(outTensor, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMean, data);
    ASSERT_NEAR(output, expected, std::abs(expected) * 1e-3f + 1e-3f)
        << "ReduceMean mismatch for " << elements << " elements";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceMinMax_LargeArray) {
  uint32_t elements = 100000u;
  auto data = generateTestData<float>(elements, 42);
  auto bufIn =
      runtime_->createTensor({elements}, DataType::Float32, data.data());

  // Test ReduceMin
  {
    auto outTensor = runtime_->ops().reduce(ReduceMin, bufIn);
    float output = 0.0f;
    runtime_->copyFromTensor(outTensor, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMin, data);
    ASSERT_NEAR(output, expected, 1e-5f) << "ReduceMin mismatch";
  }

  // Test ReduceMax
  {
    auto outTensor = runtime_->ops().reduce(ReduceMax, bufIn);
    float output = 0.0f;
    runtime_->copyFromTensor(outTensor, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceMax, data);
    ASSERT_NEAR(output, expected, 1e-5f) << "ReduceMax mismatch";
  }
}

TEST_F(MultiWorkgroupReduceTest, ReduceProd_LargeArray) {
  // Use smaller values to avoid overflow
  uint32_t elements = 1000u;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.99f, 1.01f);
  std::vector<float> data(elements);
  for (auto &v : data)
    v = dist(gen);

  auto bufIn =
      runtime_->createTensor({elements}, DataType::Float32, data.data());

  auto outTensor = runtime_->ops().reduce(ReduceProd, bufIn);
  float output = 0.0f;
  runtime_->copyFromTensor(outTensor, &output, sizeof(float));

  float expected = reduceRef<float>(ReduceProd, data);
  ASSERT_NEAR(output, expected, std::abs(expected) * 1e-2f + 1e-5f)
      << "ReduceProd mismatch";
}

TEST_F(MultiWorkgroupReduceTest, SmallArrayStillWorks) {
  // Verify small arrays (<=256) still use single-workgroup path
  for (uint32_t elements : {1u, 4u, 100u, 256u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto data = generateTestData<float>(elements, 42);
    auto bufIn =
        runtime_->createTensor({elements}, DataType::Float32, data.data());

    auto outTensor = runtime_->ops().reduce(ReduceSum, bufIn);
    float output = 0.0f;
    runtime_->copyFromTensor(outTensor, &output, sizeof(float));

    float expected = reduceRef<float>(ReduceSum, data);
    ASSERT_NEAR(output, expected, std::abs(expected) * 1e-4f + 1e-5f);
  }
}

// ============================================================================
// Prefix Scan Tests
// ============================================================================

class PrefixScanTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(PrefixScanTest, ExclusiveSum_Small) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "prefixscan/exclusive_small")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PrefixScanTest, ExclusiveSum_Large) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "prefixscan/exclusive_large")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PrefixScanTest, InclusiveSum_Small) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "prefixscan/inclusive_small")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PrefixScanTest, InclusiveSum_Large) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "prefixscan/inclusive_large")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Bitonic Sort Tests (Float32)
// ============================================================================

class BitonicSortTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(BitonicSortTest, Sort_SmallArrays) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/bitonic_small")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BitonicSortTest, Sort_LargeArray) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/bitonic_large")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BitonicSortTest, Sort_AlreadySorted) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/bitonic_alreadysorted")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BitonicSortTest, Sort_ReverseSorted) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/bitonic_reversesorted")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BitonicSortTest, Sort_AllSameValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/bitonic_allsame")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Radix Sort Tests (UInt32)
// ============================================================================

class RadixSortTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(RadixSortTest, Sort_SmallArrays_UInt32) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/radix_small")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RadixSortTest, Sort_LargeArray_UInt32) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/radix_large")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RadixSortTest, Sort_AlreadySorted_UInt32) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/radix_alreadysorted")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RadixSortTest, Sort_ReverseSorted_UInt32) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/radix_reversesorted")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RadixSortTest, Sort_AllSameValues_UInt32) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "sort/radix_allsame")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Binary Vec-Scalar Tests - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, BinaryVecScalarOperators_Int32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "binary_vecscalar" ||
        c.name.find("/i32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST_F(VulkanBackendTest, BinaryVecScalarOperators_UInt32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "binary_vecscalar" ||
        c.name.find("/u32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Unary Operators - UInt32
// ============================================================================

TEST_F(VulkanBackendTest, UnaryOperators_UInt32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "unary" || c.name.find("/u32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Ternary Operators - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, TernaryClamp_Int32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("clamp/i32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST_F(VulkanBackendTest, TernaryClamp_UInt32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("clamp/u32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST_F(VulkanBackendTest, TernarySelect_Int32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("select/i32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST_F(VulkanBackendTest, TernarySelect_UInt32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "ternary" || c.name.find("select/u32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Reduction Operators - Int32 and UInt32
// ============================================================================

TEST_F(VulkanBackendTest, ReductionOperators_Int32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "reduce" || c.name.find("/i32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST_F(VulkanBackendTest, ReductionOperators_UInt32) {
  int ran = 0;
  for (const auto &c : opregistry::allOpCases()) {
    if (c.family != "reduce" || c.name.find("/u32") == std::string::npos)
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

class MatrixOpsTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Registry-driven: verifies the "matmul/square" case (default matmul, 4x4 identity).
TEST_F(MatrixOpsTest, MatMul_Square) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/square")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/rectangular" case.
TEST_F(MatrixOpsTest, MatMul_Rectangular) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/rectangular")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/larger" case (several sizes).
TEST_F(MatrixOpsTest, MatMul_LargerMatrices) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Compare BatchedRoPE GPU output against per-token GPU RoPE output —
// same input, same tables, but applied two different ways.
// Test BatchedRoPE with the EXACT shape/strides prefillBatched uses:
// input is [N, qdim+2*kvdim] (fused QKV output), apply RoPE to Q slice
// (cols 0..qdim-1) and K slice (cols qdim..qdim+kvdim-1).
TEST_F(MatrixOpsTest, BatchedRoPE_FusedQKVLayout) {
  const uint32_t N = 15;
  const uint32_t qdim = 576;
  const uint32_t kvdim = 192;
  const uint32_t head_dim = 64;
  const uint32_t halfDim = head_dim / 2;
  const uint32_t total = qdim + 2 * kvdim;  // 960
  const uint32_t maxSeq = 64;

  std::vector<float> input(N * total);
  std::mt19937 gen(13);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : input) x = dist(gen);

  std::vector<float> cosTbl(maxSeq * halfDim), sinTbl(maxSeq * halfDim);
  for (uint32_t p = 0; p < maxSeq; ++p) {
    for (uint32_t i = 0; i < halfDim; ++i) {
      float theta = float(p) / std::pow(10000.0f, 2.0f * i / head_dim);
      cosTbl[p * halfDim + i] = std::cos(theta);
      sinTbl[p * halfDim + i] = std::sin(theta);
    }
  }
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  std::vector<uint32_t> positions(N);
  for (uint32_t i = 0; i < N; ++i) positions[i] = i;
  auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());

  auto bufIn = runtime_->createTensor({N, total}, DataType::Float32, input.data());

  // Apply batched RoPE on Q slice
  auto bufQRoped = runtime_->ops().applyBatchedRoPE(
      bufIn, bufCos, bufSin, bufPos, N, qdim, total, 0, head_dim);
  std::vector<float> qRoped(N * qdim);
  runtime_->copyFromTensor(bufQRoped, qRoped.data(), qRoped.size() * sizeof(float));

  // CPU reference for Q rotation
  for (uint32_t row = 0; row < N; ++row) {
    uint32_t pos = row;
    uint32_t n_heads = qdim / head_dim;  // 9
    for (uint32_t head = 0; head < n_heads; ++head) {
      for (uint32_t i = 0; i < halfDim; ++i) {
        uint32_t idxLo = head * head_dim + i;
        uint32_t idxHi = idxLo + halfDim;
        float c = cosTbl[pos * halfDim + i];
        float s = sinTbl[pos * halfDim + i];
        // Read Q from input at column 0..qdim-1 of row
        float x0 = input[row * total + idxLo];
        float x1 = input[row * total + idxHi];
        float eLo = x0 * c - x1 * s;
        float eHi = x0 * s + x1 * c;
        ASSERT_NEAR(qRoped[row * qdim + idxLo], eLo,
                    std::abs(eLo) * 1e-4f + 1e-5f)
          << "row=" << row << " idxLo=" << idxLo;
        ASSERT_NEAR(qRoped[row * qdim + idxHi], eHi,
                    std::abs(eHi) * 1e-4f + 1e-5f)
          << "row=" << row << " idxHi=" << idxHi;
      }
    }
  }
}

// End-to-end test of the two-dispatch split:
// 1. BatchedKVCacheWrite writes K (with RoPE) and V to cache for N tokens.
// 2. BatchedAttentionReadCache reads the cache and computes attention.
// Compare against per-token AttentionOp + per-token cacheWrite + per-token
// applyRoPE (the proven path) — outputs should match within FP tolerance.
TEST_F(MatrixOpsTest, BatchedKVCacheWrite_plus_AttentionReadCache_MatchesPerToken) {
  const uint32_t N = 5;
  const uint32_t n_heads = 2;
  const uint32_t n_kv_heads = 1;  // GQA
  const uint32_t head_dim = 32;
  const uint32_t qdim = n_heads * head_dim;
  const uint32_t kvdim = n_kv_heads * head_dim;
  const uint32_t halfDim = head_dim / 2;
  const uint32_t maxSeq = 16;

  std::vector<float> qInput(N * qdim);
  std::vector<float> kInput(N * kvdim);
  std::vector<float> vInput(N * kvdim);
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : qInput) x = dist(gen);
  for (auto &x : kInput) x = dist(gen);
  for (auto &x : vInput) x = dist(gen);

  std::vector<float> cosTbl(maxSeq * halfDim), sinTbl(maxSeq * halfDim);
  for (uint32_t p = 0; p < maxSeq; ++p) {
    for (uint32_t i = 0; i < halfDim; ++i) {
      float theta = float(p) / std::pow(10000.0f, 2.0f * i / head_dim);
      cosTbl[p * halfDim + i] = std::cos(theta);
      sinTbl[p * halfDim + i] = std::sin(theta);
    }
  }
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  // --- Reference path: per-token applyRoPE + cacheWrite + attention ---
  std::vector<float> refAttn(N * qdim);
  {
    auto kCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float32);
    auto vCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float32);
    std::vector<std::vector<uint32_t>> paramsHolder(N);
    std::vector<std::vector<float>> qRowHolder(N), kRowHolder(N), vRowHolder(N);
    for (uint32_t i = 0; i < N; ++i) {
      paramsHolder[i] = {i, i + 1};
      qRowHolder[i].assign(qInput.begin() + i * qdim, qInput.begin() + (i + 1) * qdim);
      kRowHolder[i].assign(kInput.begin() + i * kvdim, kInput.begin() + (i + 1) * kvdim);
      vRowHolder[i].assign(vInput.begin() + i * kvdim, vInput.begin() + (i + 1) * kvdim);

      auto qBuf = runtime_->createTensor({qdim}, DataType::Float32, qRowHolder[i].data());
      auto kBuf = runtime_->createTensor({kvdim}, DataType::Float32, kRowHolder[i].data());
      auto vBuf = runtime_->createTensor({kvdim}, DataType::Float32, vRowHolder[i].data());
      auto bufParams = runtime_->createTensor({2}, DataType::UInt32, paramsHolder[i].data());

      auto qRoped = runtime_->ops().applyRoPE(qBuf, bufCos, bufSin, bufParams, head_dim);
      auto kRoped = runtime_->ops().applyRoPE(kBuf, bufCos, bufSin, bufParams, head_dim);
      runtime_->ops().cacheWrite(kCache, kRoped, bufParams);
      runtime_->ops().cacheWrite(vCache, vBuf, bufParams);
      auto attn = runtime_->ops().attention(qRoped, kCache, vCache, bufParams,
                                             n_heads, n_kv_heads, head_dim);
      runtime_->copyFromTensor(attn, refAttn.data() + i * qdim, qdim * sizeof(float));
    }
  }

  // --- Batched path: BatchedKVCacheWrite + BatchedAttentionReadCache ---
  std::vector<float> batchedAttn(N * qdim);
  {
    auto kCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float32);
    auto vCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float32);
    std::vector<uint32_t> positions(N);
    for (uint32_t i = 0; i < N; ++i) positions[i] = i;
    auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());

    auto qBuf = runtime_->createTensor({N, qdim}, DataType::Float32, qInput.data());
    auto kBuf = runtime_->createTensor({N, kvdim}, DataType::Float32, kInput.data());
    auto vBuf = runtime_->createTensor({N, kvdim}, DataType::Float32, vInput.data());

    runtime_->ops().batchedKVCacheWrite(kBuf, vBuf, kCache, vCache, bufPos,
                                         bufCos, bufSin,
                                         N, n_kv_heads, head_dim,
                                         /*kStride=*/kvdim, /*vStride=*/kvdim,
                                         /*kOffset=*/0, /*vOffset=*/0);
    auto attn = runtime_->ops().batchedAttentionReadCache(
        qBuf, kCache, vCache, bufPos, bufCos, bufSin,
        N, n_heads, n_kv_heads, head_dim,
        /*qStride=*/qdim, /*qOffset=*/0);
    runtime_->copyFromTensor(attn, batchedAttn.data(),
                              batchedAttn.size() * sizeof(float));
  }

  // Compare row by row.
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < qdim; ++j) {
      float r = refAttn[i * qdim + j];
      float b = batchedAttn[i * qdim + j];
      ASSERT_NEAR(b, r, std::abs(r) * 1e-3f + 1e-4f)
          << "row=" << i << " j=" << j;
    }
  }
}

// Minimal repro of the runtime issue: when applyBatchedRoPE runs and
// then per-token applyRoPE runs (consuming a view into a buffer that was
// the source of the batched op), the per-token output is all zeros.
TEST_F(MatrixOpsTest, PerTokenRoPE_AfterBatchedRoPE_Repro) {
  const uint32_t N = 4;
  const uint32_t head_dim = 32;
  const uint32_t dim = head_dim;
  const uint32_t halfDim = head_dim / 2;
  const uint32_t maxSeq = 8;

  std::vector<float> input(N * dim);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : input) x = dist(gen);

  // pos=0 → cos=1, sin=0 → RoPE is identity → output should equal input.
  std::vector<float> cosTbl(maxSeq * halfDim, 1.0f);
  std::vector<float> sinTbl(maxSeq * halfDim, 0.0f);
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  auto bufIn = runtime_->createTensor({N, dim}, DataType::Float32, input.data());

  // 1. Run batched RoPE (all positions = 0 for simplicity → identity).
  std::vector<uint32_t> positions(N, 0);
  auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());
  auto qRoped = runtime_->ops().applyBatchedRoPE(
      bufIn, bufCos, bufSin, bufPos, N, dim, dim, 0, head_dim);
  std::vector<float> batched(N * dim);
  runtime_->copyFromTensor(qRoped, batched.data(), batched.size() * sizeof(float));
  // Sanity: batched output should equal input (identity at pos=0).
  for (uint32_t i = 0; i < std::min<uint32_t>(N * dim, 8); ++i) {
    ASSERT_FLOAT_EQ(batched[i], input[i]) << "batched i=" << i;
  }

  // 2. Now apply per-token RoPE on a view of bufIn (row 0, identity at pos=0).
  uint32_t p[2] = {0, 1};
  auto bufParams = runtime_->createTensor({2}, DataType::UInt32, p);
  auto rowView = runtime_->store().createTensorView(bufIn, 0, {dim},
                                                     DataType::Float32);
  auto perTokOut = runtime_->ops().applyRoPE(rowView, bufCos, bufSin, bufParams, head_dim);
  std::vector<float> perTok(dim);
  runtime_->copyFromTensor(perTokOut, perTok.data(), dim * sizeof(float));

  // perTok should also equal input[0..dim-1] (identity at pos=0).
  for (uint32_t i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(perTok[i], input[i])
        << "per-token i=" << i << " (batched got " << batched[i] << ")";
  }
}

// BatchedRoPE: applies rotary embedding to N tokens in one dispatch.
// Verify by comparing to the per-token applyRoPE op called N times.
// Test with varying positions using real cos/sin tables.
TEST_F(MatrixOpsTest, BatchedRoPE_VaryingPos) {
  const uint32_t N = 2;
  const uint32_t head_dim = 32;
  const uint32_t dim = head_dim;
  const uint32_t halfDim = head_dim / 2;
  const uint32_t maxSeq = 8;

  std::vector<float> input(N * dim);
  for (uint32_t i = 0; i < input.size(); ++i) input[i] = float(i + 1) * 0.1f;

  std::vector<float> cosTbl(maxSeq * halfDim), sinTbl(maxSeq * halfDim);
  for (uint32_t p = 0; p < maxSeq; ++p) {
    for (uint32_t i = 0; i < halfDim; ++i) {
      float theta = float(p) / std::pow(10000.0f, 2.0f * i / head_dim);
      cosTbl[p * halfDim + i] = std::cos(theta);
      sinTbl[p * halfDim + i] = std::sin(theta);
    }
  }
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  std::vector<uint32_t> positions = {0, 1};  // varying positions
  auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());
  auto bufIn = runtime_->createTensor({N, dim}, DataType::Float32, input.data());
  auto bufOut = runtime_->ops().applyBatchedRoPE(bufIn, bufCos, bufSin, bufPos, N, dim, dim, 0, head_dim);

  std::vector<float> got(N * dim);
  runtime_->copyFromTensor(bufOut, got.data(), got.size() * sizeof(float));

  // Only check row 0 (pos=0 → identity). Row 1 (pos=1) is non-identity.
  for (uint32_t i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(got[0 * dim + i], input[0 * dim + i])
      << "row=0 i=" << i;
  }
  // Row 1: should differ from input (RoPE rotation applied)
  bool anyDiff = false;
  for (uint32_t i = 0; i < dim; ++i) {
    if (std::abs(got[1 * dim + i] - input[1 * dim + i]) > 1e-4f) { anyDiff = true; break; }
  }
  EXPECT_TRUE(anyDiff) << "row 1 should be non-identity for pos=1";
}

// Simpler test: pos=0 for all tokens → RoPE is identity, so output == input.
TEST_F(MatrixOpsTest, BatchedRoPE_IdentityAllPosZero) {
  const uint32_t N = 4;
  const uint32_t head_dim = 32;
  const uint32_t dim = head_dim;  // 1 head
  const uint32_t halfDim = head_dim / 2;
  const uint32_t maxSeq = 8;

  std::vector<float> input(N * dim);
  for (uint32_t i = 0; i < input.size(); ++i) input[i] = float(i) + 1.0f;

  // cos(0)=1, sin(0)=0 for pos 0. Fill table with valid values.
  std::vector<float> cosTbl(maxSeq * halfDim, 1.0f);
  std::vector<float> sinTbl(maxSeq * halfDim, 0.0f);
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  std::vector<uint32_t> positions(N, 0);  // ALL zeros
  auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());
  auto bufIn = runtime_->createTensor({N, dim}, DataType::Float32, input.data());
  auto bufOut = runtime_->ops().applyBatchedRoPE(bufIn, bufCos, bufSin, bufPos, N, dim, dim, 0, head_dim);

  std::vector<float> got(N * dim);
  runtime_->copyFromTensor(bufOut, got.data(), got.size() * sizeof(float));

  // With cos=1, sin=0: output = input (identity)
  for (uint32_t row = 0; row < N; ++row) {
    for (uint32_t i = 0; i < dim; ++i) {
      float e = input[row * dim + i];
      float g = got[row * dim + i];
      EXPECT_FLOAT_EQ(g, e) << "row=" << row << " i=" << i;
    }
  }
}

TEST_F(MatrixOpsTest, BatchedRoPE_MatchesPerTokenRoPE) {
  const uint32_t N = 15;
  const uint32_t n_heads = 9;
  const uint32_t head_dim = 64;
  const uint32_t dim = n_heads * head_dim;
  const uint32_t maxSeq = 64;
  const uint32_t halfDim = head_dim / 2;

  // Random input
  std::vector<float> input(N * dim);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : input) x = dist(gen);

  // Precompute cos/sin tables [maxSeq * halfDim] (theta_i = 1/10000^(2i/head_dim))
  std::vector<float> cosTbl(maxSeq * halfDim), sinTbl(maxSeq * halfDim);
  for (uint32_t pos = 0; pos < maxSeq; ++pos) {
    for (uint32_t i = 0; i < halfDim; ++i) {
      float theta = float(pos) / std::pow(10000.0f, 2.0f * i / head_dim);
      cosTbl[pos * halfDim + i] = std::cos(theta);
      sinTbl[pos * halfDim + i] = std::sin(theta);
    }
  }

  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  // --- CPU reference: apply RoPE row-by-row ---
  std::vector<float> expectedAll(N * dim);
  for (uint32_t row = 0; row < N; ++row) {
    uint32_t pos = row;
    for (uint32_t head = 0; head < n_heads; ++head) {
      for (uint32_t i = 0; i < halfDim; ++i) {
        uint32_t idxLo = head * head_dim + i;
        uint32_t idxHi = idxLo + halfDim;
        float c = cosTbl[pos * halfDim + i];
        float s = sinTbl[pos * halfDim + i];
        float x0 = input[row * dim + idxLo];
        float x1 = input[row * dim + idxHi];
        expectedAll[row * dim + idxLo] = x0 * c - x1 * s;
        expectedAll[row * dim + idxHi] = x0 * s + x1 * c;
      }
    }
  }

  // --- Batched ---
  std::vector<uint32_t> positions(N);
  for (uint32_t i = 0; i < N; ++i) positions[i] = i;
  auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());
  auto bufIn = runtime_->createTensor({N, dim}, DataType::Float32, input.data());
  auto bufBatched = runtime_->ops().applyBatchedRoPE(bufIn, bufCos, bufSin, bufPos, N, dim, dim, 0, head_dim);
  std::vector<float> gotAll(N * dim);
  runtime_->copyFromTensor(bufBatched, gotAll.data(), gotAll.size() * sizeof(float));

  for (uint32_t row = 0; row < N; ++row) {
    for (uint32_t i = 0; i < dim; ++i) {
      float e = expectedAll[row * dim + i];
      float g = gotAll[row * dim + i];
      ASSERT_NEAR(g, e, std::abs(e) * 1e-4f + 1e-5f)
        << "row=" << row << " i=" << i;
    }
  }
}

// Test batched rmsNorm + matmul chain on a row of a [N, dim] buffer.
// Reproduces the QKV path in prefillBatched: embedding → rmsNorm → matmul.
TEST_F(MatrixOpsTest, RMSNorm_BatchedFollowedByMatMul) {
  const uint32_t N = 15, dim = 576;
  std::vector<float> hidden(N * dim);
  std::vector<float> weight(dim);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : hidden) x = dist(gen);
  for (auto &x : weight) x = dist(gen) * 0.1f + 1.0f;

  auto bufH = runtime_->createTensor({N, dim}, DataType::Float32, hidden.data());
  auto bufW = runtime_->createTensor({dim}, DataType::Float32, weight.data());
  auto bufNormed = runtime_->ops().rmsNorm(bufH, bufW, 1e-5f);

  std::vector<float> normedGpu(N * dim);
  runtime_->copyFromTensor(bufNormed, normedGpu.data(), normedGpu.size() * sizeof(float));

  // CPU reference: rmsNorm per row
  for (uint32_t i = 0; i < N; ++i) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < dim; ++j) ss += hidden[i*dim+j] * hidden[i*dim+j];
    float scale = 1.0f / std::sqrt(ss / dim + 1e-5f);
    for (uint32_t j = 0; j < dim; ++j) {
      float expected = hidden[i*dim+j] * scale * weight[j];
      float got = normedGpu[i*dim+j];
      ASSERT_NEAR(got, expected, std::abs(expected) * 1e-3f + 1e-4f)
        << "row=" << i << " col=" << j;
    }
  }
}

// Test all the matmul shapes prefillBatched uses, all in one parameterized
// test. Each shape is verified against a CPU reference.
TEST_F(MatrixOpsTest, MatMul_LlamaPrefillShapes_F32xF16) {
  struct Shape { uint32_t M, K, N; const char* name; };
  std::array<Shape, 5> shapes = {{
      {15, 576, 960,  "QKV"},
      {15, 576, 576,  "OutputProj"},
      {15, 576, 1536, "Gate/Up"},
      {15, 1536, 576, "Down"},
      {15, 576, 49152,"LMHead-likeButTooLarge"},  // would be slow; exercises bigger N
  }};

  std::mt19937 gen0(42);
  for (const auto &sh : shapes) {
    if (sh.M * sh.N > 500000) continue;  // skip very large to keep test fast
    SCOPED_TRACE(sh.name);
    uint32_t M = sh.M, K = sh.K, N = sh.N;

    std::vector<float> dataA(M * K);
    std::vector<uint16_t> dataB16(K * N);
    std::vector<float> dataB32(K * N);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : dataA) x = dist(gen0);
    for (size_t i = 0; i < dataB32.size(); ++i) {
      dataB32[i] = dist(gen0);
      uint32_t f = *reinterpret_cast<uint32_t*>(&dataB32[i]);
      uint32_t sign = (f >> 31) & 1;
      int32_t exp = ((f >> 23) & 0xff) - 127 + 15;
      uint32_t mant = (f >> 13) & 0x3ff;
      if (exp < 0)       dataB16[i] = static_cast<uint16_t>(sign << 15);
      else if (exp > 30) dataB16[i] = static_cast<uint16_t>((sign << 15) | (0x1f << 10));
      else               dataB16[i] = static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
      uint16_t h = dataB16[i];
      sign = (h >> 15) & 1;
      uint32_t hexp = (h >> 10) & 0x1f, hmant = h & 0x3ff;
      if (hexp == 0) dataB32[i] = 0.0f;
      else if (hexp == 0x1f) dataB32[i] = sign ? -1e30f : 1e30f;
      else {
        uint32_t fexp = hexp + 127 - 15;
        uint32_t fbits = (sign << 31) | (fexp << 23) | (hmant << 13);
        dataB32[i] = *reinterpret_cast<float*>(&fbits);
      }
    }

    auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
    auto bufB = runtime_->createTensor({K, N}, DataType::Float16, dataB16.data());
    auto bufC = runtime_->ops().matmul(bufA, bufB);

    std::vector<float> output(M * N);
    runtime_->copyFromTensor(bufC, output.data(), output.size() * sizeof(float));

    int mismatches = 0;
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < N; ++j) {
        float expected = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
          expected += dataA[i * K + k] * dataB32[k * N + j];
        }
        float got = output[i * N + j];
        float tol = std::abs(expected) * 1e-2f + 1e-3f;
        if (std::abs(got - expected) > tol) {
          if (mismatches < 3)
            std::cerr << sh.name << " mismatch at [" << i << "," << j << "]: got=" << got
                      << " expected=" << expected << "\n";
          mismatches++;
        }
      }
    }
    EXPECT_EQ(mismatches, 0) << sh.name << " mismatches: " << mismatches;
  }
}

TEST_F(MatrixOpsTest, MatMul_LlamaPrefillQKV_F32xF16) {
  const uint32_t M = 15, K = 576, N = 960;

  std::vector<float> dataA(M * K);
  std::vector<uint16_t> dataB16(K * N);
  std::vector<float> dataB32(K * N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : dataA) x = dist(gen);
  for (size_t i = 0; i < dataB32.size(); ++i) {
    dataB32[i] = dist(gen);
    // Quick f32->f16 conversion via bit manipulation (good enough for test data)
    uint32_t f = *reinterpret_cast<uint32_t*>(&dataB32[i]);
    uint32_t sign = (f >> 31) & 1;
    int32_t exp = ((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3ff;
    if (exp < 0) { dataB16[i] = static_cast<uint16_t>(sign << 15); }
    else if (exp > 30) { dataB16[i] = static_cast<uint16_t>((sign << 15) | (0x1f << 10)); }
    else { dataB16[i] = static_cast<uint16_t>((sign << 15) | (exp << 10) | mant); }
    // Reconstruct f16 value back to f32 for reference
    uint16_t h = dataB16[i];
    sign = (h >> 15) & 1;
    uint32_t hexp = (h >> 10) & 0x1f;
    uint32_t hmant = h & 0x3ff;
    if (hexp == 0) dataB32[i] = 0.0f;
    else if (hexp == 0x1f) dataB32[i] = sign ? -1e30f : 1e30f;
    else {
      uint32_t fexp = hexp + 127 - 15;
      uint32_t fbits = (sign << 31) | (fexp << 23) | (hmant << 13);
      dataB32[i] = *reinterpret_cast<float*>(&fbits);
    }
  }

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Float16, dataB16.data());
  auto bufC = runtime_->ops().matmul(bufA, bufB);

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), output.size() * sizeof(float));

  // CPU reference using the f16-roundtripped values
  int mismatches = 0;
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        expected += dataA[i * K + k] * dataB32[k * N + j];
      }
      float got = output[i * N + j];
      float tol = std::abs(expected) * 1e-2f + 1e-3f;
      if (std::abs(got - expected) > tol) {
        if (mismatches < 5)
          std::cerr << "Mismatch at [" << i << "," << j << "]: got=" << got
                    << " expected=" << expected << " diff=" << (got - expected) << "\n";
        mismatches++;
      }
    }
  }
  EXPECT_EQ(mismatches, 0) << "Total mismatches: " << mismatches << " / " << (M * N);
}

// ============================================================================
// MatMul Variant Tests
// ============================================================================

// Matmul variant count and names come from MatMulVariants.generated.h
// via impl/matmul/MatMul.h (kMatMulVariantCount, getMatMulVariantName)
// The variant sweep now lives in the op-case registry (family "matmul");
// see tests/harness/OpRegistry.h shouldSkipMatMulVariant / matmulVariantsSweep.

// Registry-driven: verifies the "matmul/variants_square" case (variant sweep).
TEST_F(MatrixOpsTest, MatMulVariants_Square) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/variants_square")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/variants_rectangular" case.
TEST_F(MatrixOpsTest, MatMulVariants_Rectangular) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/variants_rectangular")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/variants_nonmultiple" case.
TEST_F(MatrixOpsTest, MatMulVariants_NonMultipleOfTileSize) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/variants_nonmultiple")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/variants_larger" case (tol = K*5e-5).
TEST_F(MatrixOpsTest, MatMulVariants_LargerMatrices) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/variants_larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "matmul/variants_identity" case (skips SiLU).
TEST_F(MatrixOpsTest, MatMulVariants_Identity) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "matmul/variants_identity")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// MatMulQ8 Tests
// ============================================================================

// Helper: convert float to float16 bits (IEEE 754 half-precision)

TEST_F(MatrixOpsTest, MatMulQ8_Simple) {
  // Test matmulQ8 with known data: A[M,K] * B[K,N] = C[M,N]
  // B stored as Int8 [K,N], scales as Float16 [K/32,N]
  // With scale=1.0, dequantized B values equal the int8 values.
  const uint32_t M = 1, K = 32, N = 2;
  const uint32_t blocksK = K / 32; // = 1

  // Activation A[1, 32]: all ones
  std::vector<float> dataA(M * K, 1.0f);

  // Weight B[32, 2] as Int8: col 0 = all 1s, col 1 = all 2s
  std::vector<int8_t> dataB(K * N);
  for (uint32_t k = 0; k < K; ++k) {
    dataB[k * N + 0] = 1; // col 0
    dataB[k * N + 1] = 2; // col 1
  }

  // Scales [1, 2] as Float16: all 1.0
  uint16_t f16_one = f32_to_f16(1.0f);
  std::vector<uint16_t> scales(blocksK * N, f16_one);

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS =
      runtime_->createTensor({blocksK, N}, DataType::Float16, scales.data());

  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // Expected: C[0][0] = sum(1.0 * 1 * 1.0) = 32.0
  //           C[0][1] = sum(1.0 * 2 * 1.0) = 64.0
  ASSERT_NEAR(output[0], 32.0f, 1e-3f) << "C[0][0] mismatch";
  ASSERT_NEAR(output[1], 64.0f, 1e-3f) << "C[0][1] mismatch";
}

TEST_F(MatrixOpsTest, MatMulQ8_WithScales) {
  // Test that scales are applied correctly
  const uint32_t M = 1, K = 64, N = 2;
  const uint32_t blocksK = K / 32; // = 2

  // A[1, 64]: all ones
  std::vector<float> dataA(M * K, 1.0f);

  // B[64, 2] as Int8: all 4s
  std::vector<int8_t> dataB(K * N, 4);

  // Scales [2, 2] as [K/32, N]: transposed from original [N, K/32]
  // Original: n=0 → {0.5, 1.0}, n=1 → {2.0, 0.25}
  // Transposed [K/32=2, N=2]: row0={0.5, 2.0}, row1={1.0, 0.25}
  std::vector<uint16_t> scales = {
      f32_to_f16(0.5f),
      f32_to_f16(2.0f), // block 0 scales for n=0, n=1
      f32_to_f16(1.0f),
      f32_to_f16(0.25f) // block 1 scales for n=0, n=1
  };

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS =
      runtime_->createTensor({blocksK, N}, DataType::Float16, scales.data());

  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // C[0][0] = sum_k(1.0 * 4 * scale[k/32][0])
  //         = 32 * (4 * 0.5) + 32 * (4 * 1.0) = 64 + 128 = 192
  // C[0][1] = 32 * (4 * 2.0) + 32 * (4 * 0.25) = 256 + 32 = 288
  ASSERT_NEAR(output[0], 192.0f, 1.0f) << "C[0][0] mismatch";
  ASSERT_NEAR(output[1], 288.0f, 1.0f) << "C[0][1] mismatch";
}

TEST_F(MatrixOpsTest, MatMulQ8_VsRegularMatMul) {
  // Compare matmulQ8 output against regular matmul with manually
  // dequantized weights to verify correctness end-to-end.
  const uint32_t M = 2, K = 64, N = 4;
  const uint32_t blocksPerRow = K / 32;

  auto dataA = generateTestData<float>(M * K, 42);

  // Generate int8 weight values in range [-10, 10] in [K, N] layout
  // (transposed at load time, matching regular matmul convention)
  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i) {
    dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  }

  // Generate scale values in [blocksPerRow, N] layout: small positive floats
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  }
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  }

  // Run matmulQ8 on GPU — B is [K, N], scales are [blocksPerRow, N]
  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16,
                                     scaleF16.data());
  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> gpuOutput(M * N);
  runtime_->copyFromTensor(bufC, gpuOutput.data(), M * N * sizeof(float));

  // CPU reference: dequantize B then matmul
  // B is [K, N], scales are [K/32, N]
  // C[m][n] = sum_k(A[m][k] * B[k][n] * scale[k/32][n])
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        float a = dataA[m * K + k];
        float b = static_cast<float>(dataB[k * N + n]);
        float s = scaleFloats[(k / 32) * N + n];
        expected += a * b * s;
      }
      ASSERT_NEAR(gpuOutput[m * N + n], expected,
                  std::abs(expected) * 0.01f + 0.1f)
          << "Mismatch at C[" << m << "][" << n << "]";
    }
  }
}

// Mistral-Small-24B geometry: exercise the M>1 Q8 matmul variant (VecT16R4x4,
// auto-selected for M=2..31) at K=5120 / N=4096 — the wq projection shape —
// against the proven M=1 GEMV decode path and a CPU reference.
// Repro for the batched-prefill corruption (docs/multi-device-followup.md P1).
TEST_F(MatrixOpsTest, MatMulQ8_MistralGeometry_BatchedVsPerRow) {
  const uint32_t M = 4, K = 5120, N = 4096;
  const uint32_t blocksPerRow = K / 32; // 160

  std::vector<float> dataA(M * K);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : dataA) x = dist(gen);

  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i) {
    dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  }

  // Scales exactly representable in f16 so the CPU reference is exact.
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleFloats[i] = (1.0f + (float)(i % 7)) * 0.03125f;
  }
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  }

  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16,
                                     scaleF16.data());

  // --- Batched path (M=4, tiled M>1 variant) ---
  std::vector<float> batchedOut(M * N);
  {
    auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
    auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);
    runtime_->copyFromTensor(bufC, batchedOut.data(), M * N * sizeof(float));
  }

  // --- Per-row path (M=1 GEMV, the proven decode path) ---
  std::vector<float> perRowOut(M * N);
  for (uint32_t m = 0; m < M; ++m) {
    auto bufRow =
        runtime_->createTensor({K}, DataType::Float32, dataA.data() + m * K);
    auto bufOut = runtime_->ops().matmul(bufRow, bufB, bufS);
    runtime_->copyFromTensor(bufOut, perRowOut.data() + m * N,
                             N * sizeof(float));
  }

  // --- CPU reference (double accumulator) ---
  std::vector<float> ref(M * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      double sum = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        double a = dataA[m * K + k];
        double b = static_cast<double>(dataB[k * N + n]);
        double s = scaleFloats[(k / 32) * N + n];
        sum += a * b * s;
      }
      ref[m * N + n] = static_cast<float>(sum);
    }
  }

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float r = ref[m * N + n];
      float b = batchedOut[m * N + n];
      ASSERT_NEAR(b, r, std::abs(r) * 2e-3f + 0.05f)
          << "batched vs CPU m=" << m << " n=" << n;
    }
  }
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float b = batchedOut[m * N + n];
      float p = perRowOut[m * N + n];
      ASSERT_NEAR(b, p, std::abs(p) * 2e-3f + 0.05f)
          << "batched vs GEMV m=" << m << " n=" << n;
    }
  }
}

// Mistral-Small-24B geometry batched attention with Float16 KV caches (what
// the real pipeline allocates): head_dim=128, n_rep=4, kvdim=1024 (> WG_SIZE
// 256, exercises the strided cache-write loop). Compares K cache contents
// first (isolates a write bug from a read bug), then attention outputs.
// Repro for the batched-prefill corruption (docs/multi-device-followup.md P1).
TEST_F(MatrixOpsTest, BatchedAttention_MistralGeometry_F16Cache_MatchesPerToken) {
  const uint32_t N = 4;
  const uint32_t n_heads = 32;
  const uint32_t n_kv_heads = 8; // GQA, n_rep = 4
  const uint32_t head_dim = 128;
  const uint32_t qdim = n_heads * head_dim;     // 4096
  const uint32_t kvdim = n_kv_heads * head_dim; // 1024
  const uint32_t halfDim = head_dim / 2;
  const uint32_t maxSeq = 16;

  std::vector<float> qInput(N * qdim);
  std::vector<float> kInput(N * kvdim);
  std::vector<float> vInput(N * kvdim);
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : qInput) x = dist(gen);
  for (auto &x : kInput) x = dist(gen);
  for (auto &x : vInput) x = dist(gen);

  std::vector<float> cosTbl(maxSeq * halfDim), sinTbl(maxSeq * halfDim);
  for (uint32_t p = 0; p < maxSeq; ++p) {
    for (uint32_t i = 0; i < halfDim; ++i) {
      float theta = float(p) / std::pow(10000.0f, 2.0f * i / head_dim);
      cosTbl[p * halfDim + i] = std::cos(theta);
      sinTbl[p * halfDim + i] = std::sin(theta);
    }
  }
  auto bufCos = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({maxSeq * halfDim}, DataType::Float32, sinTbl.data());

  // --- Reference path: per-token applyRoPE + cacheWrite + attention ---
  std::vector<float> refAttn(N * qdim);
  std::vector<uint16_t> refKCache(N * kvdim);
  {
    auto kCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float16);
    auto vCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float16);
    std::vector<std::vector<uint32_t>> paramsHolder(N);
    std::vector<std::vector<float>> qRowHolder(N), kRowHolder(N), vRowHolder(N);
    for (uint32_t i = 0; i < N; ++i) {
      paramsHolder[i] = {i, i + 1};
      qRowHolder[i].assign(qInput.begin() + i * qdim, qInput.begin() + (i + 1) * qdim);
      kRowHolder[i].assign(kInput.begin() + i * kvdim, kInput.begin() + (i + 1) * kvdim);
      vRowHolder[i].assign(vInput.begin() + i * kvdim, vInput.begin() + (i + 1) * kvdim);

      auto qBuf = runtime_->createTensor({qdim}, DataType::Float32, qRowHolder[i].data());
      auto kBuf = runtime_->createTensor({kvdim}, DataType::Float32, kRowHolder[i].data());
      auto vBuf = runtime_->createTensor({kvdim}, DataType::Float32, vRowHolder[i].data());
      auto bufParams = runtime_->createTensor({2}, DataType::UInt32, paramsHolder[i].data());

      auto qRoped = runtime_->ops().applyRoPE(qBuf, bufCos, bufSin, bufParams, head_dim);
      auto kRoped = runtime_->ops().applyRoPE(kBuf, bufCos, bufSin, bufParams, head_dim);
      runtime_->ops().cacheWrite(kCache, kRoped, bufParams);
      runtime_->ops().cacheWrite(vCache, vBuf, bufParams);
      auto attn = runtime_->ops().attention(qRoped, kCache, vCache, bufParams,
                                            n_heads, n_kv_heads, head_dim);
      runtime_->copyFromTensor(attn, refAttn.data() + i * qdim, qdim * sizeof(float));
    }
    runtime_->copyFromTensor(kCache, refKCache.data(), N * kvdim * sizeof(uint16_t));
  }

  // --- Batched path: BatchedKVCacheWrite + BatchedAttentionReadCache ---
  std::vector<float> batchedAttn(N * qdim);
  std::vector<uint16_t> batKCache(N * kvdim);
  {
    auto kCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float16);
    auto vCache = runtime_->createTensorEmpty({maxSeq, kvdim}, DataType::Float16);
    std::vector<uint32_t> positions(N);
    for (uint32_t i = 0; i < N; ++i) positions[i] = i;
    auto bufPos = runtime_->createTensor({N}, DataType::UInt32, positions.data());

    auto qBuf = runtime_->createTensor({N, qdim}, DataType::Float32, qInput.data());
    auto kBuf = runtime_->createTensor({N, kvdim}, DataType::Float32, kInput.data());
    auto vBuf = runtime_->createTensor({N, kvdim}, DataType::Float32, vInput.data());

    runtime_->ops().batchedKVCacheWrite(kBuf, vBuf, kCache, vCache, bufPos,
                                        bufCos, bufSin,
                                        N, n_kv_heads, head_dim,
                                        /*kStride=*/kvdim, /*vStride=*/kvdim,
                                        /*kOffset=*/0, /*vOffset=*/0);
    auto attn = runtime_->ops().batchedAttentionReadCache(
        qBuf, kCache, vCache, bufPos, bufCos, bufSin,
        N, n_heads, n_kv_heads, head_dim,
        /*qStride=*/qdim, /*qOffset=*/0);
    runtime_->copyFromTensor(attn, batchedAttn.data(),
                             batchedAttn.size() * sizeof(float));
    runtime_->copyFromTensor(kCache, batKCache.data(), N * kvdim * sizeof(uint16_t));
  }

  // Compare K cache contents (isolates BatchedKVCacheWrite from the reader).
  for (uint32_t i = 0; i < N * kvdim; ++i) {
    float r = halfToFloat(refKCache[i]);
    float b = halfToFloat(batKCache[i]);
    ASSERT_NEAR(b, r, std::abs(r) * 1e-3f + 1e-3f)
        << "K cache row=" << (i / kvdim) << " col=" << (i % kvdim);
  }

  // Compare attention outputs.
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < qdim; ++j) {
      float r = refAttn[i * qdim + j];
      float b = batchedAttn[i * qdim + j];
      ASSERT_NEAR(b, r, std::abs(r) * 2e-3f + 2e-3f)
          << "row=" << i << " j=" << j;
    }
  }
}

// Sweep the remaining Q8 matmul shapes of the Mistral-Small prefill
// (wo, gate+SiLU, up, down) — batched M=13 vs the proven per-row GEMV path.
TEST_F(MatrixOpsTest, MatMulQ8_MistralFFNShapes_BatchedVsPerRow) {
  struct ShapeCase { uint32_t M, K, N; bool silu; const char *name; };
  const ShapeCase cases[] = {
      {13, 4096, 5120, false, "wo"},
      {13, 5120, 32768, true,  "gate_silu"},
      {13, 5120, 32768, false, "up"},
      {13, 32768, 5120, false, "down"},
      {33, 5120, 32768, true, "gate_silu_m33"},
  };

  for (const auto &c : cases) {
    SCOPED_TRACE(c.name);
    const uint32_t blocksPerRow = c.K / 32;

    std::vector<float> dataA(c.M * c.K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : dataA) x = dist(gen);

    std::vector<int8_t> dataB(c.K * c.N);
    for (size_t i = 0; i < dataB.size(); ++i) {
      dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
    }

    std::vector<float> scaleFloats(blocksPerRow * c.N);
    for (size_t i = 0; i < scaleFloats.size(); ++i) {
      scaleFloats[i] = (1.0f + (float)(i % 7)) * 0.03125f;
    }
    std::vector<uint16_t> scaleF16(scaleFloats.size());
    for (size_t i = 0; i < scaleFloats.size(); ++i) {
      scaleF16[i] = f32_to_f16(scaleFloats[i]);
    }

    auto bufB = runtime_->createTensor({c.K, c.N}, DataType::Int8, dataB.data());
    auto bufS = runtime_->createTensor({blocksPerRow, c.N}, DataType::Float16,
                                       scaleF16.data());

    std::vector<float> batchedOut(c.M * c.N);
    {
      auto bufA = runtime_->createTensor({c.M, c.K}, DataType::Float32, dataA.data());
      auto bufC = c.silu
          ? runtime_->ops().matmulUnary(cut::UnarySilu, bufA, bufB, bufS)
          : runtime_->ops().matmul(bufA, bufB, bufS);
      runtime_->copyFromTensor(bufC, batchedOut.data(), c.M * c.N * sizeof(float));
    }

    std::vector<float> perRowOut(c.M * c.N);
    for (uint32_t m = 0; m < c.M; ++m) {
      auto bufRow = runtime_->createTensor({c.K}, DataType::Float32, dataA.data() + m * c.K);
      auto bufOut = c.silu
          ? runtime_->ops().matmulUnary(cut::UnarySilu, bufRow, bufB, bufS)
          : runtime_->ops().matmul(bufRow, bufB, bufS);
      runtime_->copyFromTensor(bufOut, perRowOut.data() + m * c.N, c.N * sizeof(float));
    }

    for (uint32_t m = 0; m < c.M; ++m) {
      for (uint32_t n = 0; n < c.N; ++n) {
        float b = batchedOut[m * c.N + n];
        float p = perRowOut[m * c.N + n];
        ASSERT_NEAR(b, p, std::abs(p) * 2e-3f + 0.05f)
            << c.name << " m=" << m << " n=" << n;
      }
    }
  }
}

// Probe the TiledDot Q8 variant (spec 6) WITHOUT fusion, forced via explicit
// spec. TiledDot had no prior test coverage and FAILS this test (garbage
// output at M=33, K=5120, N=4096 on RTX 3090 / RX 9070), so it is excluded
// from auto-selection in MatMulOp.cpp. DISABLED until the shader is fixed;
// run manually with --gtest_also_run_disabled_tests.
TEST_F(MatrixOpsTest, DISABLED_MatMulQ8_TiledDotVariant_Unfused_M33) {
  if (!runtime_->store().caps().integerDotProduct) {
    GTEST_SKIP() << "no integer dot product";
  }
  const uint32_t M = 33, K = 5120, N = 4096;
  const uint32_t blocksPerRow = K / 32;

  std::vector<float> dataA(M * K);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : dataA) x = dist(gen);

  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i) {
    dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  }

  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleFloats[i] = (1.0f + (float)(i % 7)) * 0.03125f;
  }
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  }

  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16,
                                     scaleF16.data());

  std::vector<float> batchedOut(M * N);
  {
    auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
    auto bufC = runtime_->ops().matmul(bufA, bufB, bufS, /*spec=*/6u);
    runtime_->copyFromTensor(bufC, batchedOut.data(), M * N * sizeof(float));
  }

  std::vector<float> perRowOut(M * N);
  for (uint32_t m = 0; m < M; ++m) {
    auto bufRow =
        runtime_->createTensor({K}, DataType::Float32, dataA.data() + m * K);
    auto bufOut = runtime_->ops().matmul(bufRow, bufB, bufS);
    runtime_->copyFromTensor(bufOut, perRowOut.data() + m * N,
                             N * sizeof(float));
  }

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float b = batchedOut[m * N + n];
      float p = perRowOut[m * N + n];
      ASSERT_NEAR(b, p, std::abs(p) * 2e-3f + 0.05f)
          << "m=" << m << " n=" << n;
    }
  }
}

// Batched rmsNorm at Mistral dim (5120) against a CPU reference.
TEST_F(MatrixOpsTest, RMSNorm_Batched_MistralDim) {
  const uint32_t N = 13, dim = 5120;

  std::vector<float> hidden(N * dim);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : hidden) x = dist(gen);

  std::vector<float> weight(dim);
  for (auto &w : weight) w = dist(gen) * 0.1f + 1.0f;

  auto bufH = runtime_->createTensor({N, dim}, DataType::Float32, hidden.data());
  auto bufW = runtime_->createTensor({dim}, DataType::Float32, weight.data());
  auto bufNormed = runtime_->ops().rmsNorm(bufH, bufW, 1e-5f);
  std::vector<float> got(N * dim);
  runtime_->copyFromTensor(bufNormed, got.data(), N * dim * sizeof(float));

  for (uint32_t i = 0; i < N; ++i) {
    double ss = 0.0;
    for (uint32_t j = 0; j < dim; ++j) {
      ss += static_cast<double>(hidden[i * dim + j]) * hidden[i * dim + j];
    }
    double scale = 1.0 / std::sqrt(ss / dim + 1e-5);
    for (uint32_t j = 0; j < dim; ++j) {
      double expected = static_cast<double>(hidden[i * dim + j]) * scale * weight[j];
      ASSERT_NEAR(got[i * dim + j], static_cast<float>(expected),
                  std::abs(expected) * 1e-3f + 1e-4f)
          << "row=" << i << " col=" << j;
    }
  }
}

// Phase-2 LTX groundwork: interleaved RoPE op + composed bidirectional
// attention recipe (transpose + views + matmul + softmax + matmulBinary).
TEST_F(MatrixOpsTest, RoPEInterleaved_MatchesCPU) {
  const uint32_t S = 3, D = 16;
  std::vector<float> x(S * D);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : x) val = dist(gen);

  std::vector<float> cosTbl(S * D), sinTbl(S * D);
  for (uint32_t s = 0; s < S; ++s) {
    for (uint32_t k = 0; k < D / 2; ++k) {
      float theta = 0.37f * (s + 1) * (k + 1);
      float c = std::cos(theta), si = std::sin(theta);
      cosTbl[s * D + 2 * k] = cosTbl[s * D + 2 * k + 1] = c;
      sinTbl[s * D + 2 * k] = sinTbl[s * D + 2 * k + 1] = si;
    }
  }

  auto bufX = runtime_->createTensor({S, D}, DataType::Float32, x.data());
  auto bufCos = runtime_->createTensor({S, D}, DataType::Float32, cosTbl.data());
  auto bufSin = runtime_->createTensor({S, D}, DataType::Float32, sinTbl.data());
  auto bufOut = runtime_->ops().applyRoPEInterleaved(bufX, bufCos, bufSin);
  std::vector<float> got(S * D);
  runtime_->copyFromTensor(bufOut, got.data(), got.size() * sizeof(float));

  std::vector<float> expected(S * D);
  for (uint32_t s = 0; s < S; ++s) {
    for (uint32_t k = 0; k < D / 2; ++k) {
      float c = cosTbl[s * D + 2 * k], si = sinTbl[s * D + 2 * k];
      float x0 = x[s * D + 2 * k], x1 = x[s * D + 2 * k + 1];
      expected[s * D + 2 * k] = x0 * c - x1 * si;
      expected[s * D + 2 * k + 1] = x1 * c + x0 * si;
    }
  }

  for (uint32_t s = 0; s < S; ++s) {
    for (uint32_t k = 0; k < D; ++k) {
      ASSERT_NEAR(got[s * D + k], expected[s * D + k], 1e-5f)
          << "s=" << s << " k=" << k;
    }
  }
}

TEST_F(MatrixOpsTest, ComposedBidirectionalCrossAttention_MatchesCPU) {
  const uint32_t H = 4, Dh = 8, D = H * Dh, Sq = 6, Skv = 5;
  std::vector<float> q(Sq * D), k(Skv * D), v(Skv * D), wOut(D * D);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : q) val = dist(gen);
  for (auto &val : k) val = dist(gen);
  for (auto &val : v) val = dist(gen);
  for (auto &val : wOut) val = dist(gen);

  // Scale q by 1/sqrt(Dh)
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
  std::vector<float> scaledQ(Sq * D);
  for (size_t i = 0; i < scaledQ.size(); ++i) scaledQ[i] = q[i] * scale;

  auto bufQ = runtime_->createTensor({Sq, D}, DataType::Float32, scaledQ.data());
  auto bufK = runtime_->createTensor({Skv, D}, DataType::Float32, k.data());
  auto bufV = runtime_->createTensor({Skv, D}, DataType::Float32, v.data());
  auto qT = runtime_->ops().transpose(bufQ);
  auto kT = runtime_->ops().transpose(bufK);
  auto vT = runtime_->ops().transpose(bufV);
  runtime_->ops().barrier();

  uint32_t alignedSq = (Sq + 3) & ~3u, alignedSkv = (Skv + 3) & ~3u;
  cut::Tensor acc;
  for (uint32_t h = 0; h < H; ++h) {
    runtime_->ops().barrier();
    auto qhT = runtime_->store().createTensorView(qT, static_cast<size_t>(h) * Dh * alignedSq * sizeof(float),
                                                  {Dh, Sq}, DataType::Float32);
    auto khT = runtime_->store().createTensorView(kT, static_cast<size_t>(h) * Dh * alignedSkv * sizeof(float),
                                                  {Dh, Skv}, DataType::Float32);
    auto vhT = runtime_->store().createTensorView(vT, static_cast<size_t>(h) * Dh * alignedSkv * sizeof(float),
                                                  {Dh, Skv}, DataType::Float32);
    auto Qh = runtime_->ops().transpose(qhT);
    auto scores = runtime_->ops().matmul(Qh, khT);
    auto probs = runtime_->ops().softmaxFused(scores, 1);
    auto Vh = runtime_->ops().transpose(vhT);
    auto outH = runtime_->ops().matmul(probs, Vh);
    auto bufWh = runtime_->createTensor({Dh, D}, DataType::Float32,
                                        wOut.data() + static_cast<size_t>(h) * Dh * D);
    if (h == 0) {
      acc = runtime_->ops().matmul(outH, bufWh);
    } else {
      acc = runtime_->ops().matmulBinary(cut::BinaryAdd, outH, bufWh, acc);
    }
  }

  std::vector<float> gpu(Sq * D);
  runtime_->copyFromTensor(acc, gpu.data(), gpu.size() * sizeof(float));

  // CPU reference
  std::vector<float> ref(Sq * D, 0.0f);
  for (uint32_t h = 0; h < H; ++h) {
    std::vector<double> scoresRef(Sq * Skv, 0.0);
    for (uint32_t i = 0; i < Sq; ++i) {
      for (uint32_t j = 0; j < Skv; ++j) {
        for (uint32_t d = 0; d < Dh; ++d) {
          scoresRef[i * Skv + j] += scaledQ[i * D + h * Dh + d] * k[j * D + h * Dh + d];
        }
      }
    }
    // Softmax over j
    std::vector<double> probs(Sq * Skv);
    for (uint32_t i = 0; i < Sq; ++i) {
      double maxScore = *std::max_element(scoresRef.begin() + i * Skv, scoresRef.begin() + (i + 1) * Skv);
      double sumExp = 0.0;
      for (uint32_t j = 0; j < Skv; ++j) {
        probs[i * Skv + j] = std::exp(scoresRef[i * Skv + j] - maxScore);
        sumExp += probs[i * Skv + j];
      }
      for (uint32_t j = 0; j < Skv; ++j) {
        probs[i * Skv + j] /= sumExp;
      }
    }
    // Context
    std::vector<double> ctx(Sq * Dh, 0.0);
    for (uint32_t i = 0; i < Sq; ++i) {
      for (uint32_t d = 0; d < Dh; ++d) {
        for (uint32_t j = 0; j < Skv; ++j) {
          ctx[i * Dh + d] += probs[i * Skv + j] * v[j * D + h * Dh + d];
        }
      }
    }
    // Accumulate into ref
    for (uint32_t i = 0; i < Sq; ++i) {
      for (uint32_t n = 0; n < D; ++n) {
        for (uint32_t d = 0; d < Dh; ++d) {
          ref[i * D + n] += ctx[i * Dh + d] * wOut[(h * Dh + d) * D + n];
        }
      }
    }
  }

  for (uint32_t i = 0; i < Sq; ++i) {
    for (uint32_t n = 0; n < D; ++n) {
      float tol = std::abs(ref[i * D + n]) * 2e-3f + 1e-3f;
      ASSERT_NEAR(gpu[i * D + n], ref[i * D + n], tol) << "i=" << i << " n=" << n;
    }
  }
}

// Row-broadcast binary op: out[r,c] = op(a[r,c], b[c]). cols=6 deliberately
// not a multiple of 4 to exercise the padding lanes.
TEST_F(MatrixOpsTest, BinaryOpRowBcast_AddMul_MatchesCPU) {
  const uint32_t rows = 5, cols = 6;
  std::vector<float> aData(rows * cols), bData(cols);
  std::mt19937 gen(3);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : aData) v = dist(gen);
  for (auto &v : bData) v = dist(gen);

  auto bufA = runtime_->createTensor({rows, cols}, DataType::Float32, aData.data());
  auto bufB = runtime_->createTensor({cols}, DataType::Float32, bData.data());

  std::vector<float> got(rows * cols);
  auto outAdd = runtime_->ops().binaryOpRowBcast(cut::BinaryAdd, bufA, bufB);
  runtime_->copyFromTensor(outAdd, got.data(), got.size() * sizeof(float));
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_NEAR(got[r * cols + c], aData[r * cols + c] + bData[c], 1e-6f)
          << "add r=" << r << " c=" << c;
    }
  }

  auto outMul = runtime_->ops().binaryOpRowBcast(cut::BinaryMul, bufA, bufB);
  runtime_->copyFromTensor(outMul, got.data(), got.size() * sizeof(float));
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      ASSERT_NEAR(got[r * cols + c], aData[r * cols + c] * bData[c], 1e-6f)
          << "mul r=" << r << " c=" << c;
    }
  }
}

// ============================================================================
// MatMulQ4 Tests
// ============================================================================

// Helper: pack two nibble values (0-15) into one byte: low nibble + high nibble

TEST_F(MatrixOpsTest, MatMulQ4_Simple) {
  // Test matmulQ4 with known data: A[M,K] * dequant(B)[K,N] = C[M,N]
  // B stored as packed nibbles [K, N/2] Int8, scales [K/32, N] Float16
  // Dequant: value = (nibble - 8) * scale
  // With scale=1.0 and nibble=9, dequantized value = 1.0.
  const uint32_t M = 1, K = 32, N = 2;
  const uint32_t blocksK = K / 32; // = 1

  // Activation A[1, 32]: all ones
  std::vector<float> dataA(M * K, 1.0f);

  // PackedB [32, 1] (N/2=1): pack two N values into one byte.
  // N=0: nibble=9 (dequant: 9-8=1), N=1: nibble=10 (dequant: 10-8=2)
  // Byte = packNibbles(9, 10) = 0xA9
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k) {
    packedB[k * (N / 2)] = packNibbles(9, 10); // lo=n0(9), hi=n1(10)
  }

  // Scales [1, 2] as Float16: all 1.0
  uint16_t f16_one = f32_to_f16(1.0f);
  std::vector<uint16_t> scales(blocksK * N, f16_one);

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB =
      runtime_->createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS =
      runtime_->createTensor({blocksK, N}, DataType::Float16, scales.data());

  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // Expected: C[0][0] = sum(1.0 * (9-8) * 1.0) = 32 * 1 = 32.0
  //           C[0][1] = sum(1.0 * (10-8) * 1.0) = 32 * 2 = 64.0
  ASSERT_NEAR(output[0], 32.0f, 1e-3f) << "C[0][0] mismatch";
  ASSERT_NEAR(output[1], 64.0f, 1e-3f) << "C[0][1] mismatch";
}

TEST_F(MatrixOpsTest, MatMulQ4_WithScales) {
  // Test that scales are applied correctly with Q4 dequantization
  const uint32_t M = 1, K = 64, N = 2;
  const uint32_t blocksK = K / 32; // = 2

  // A[1, 64]: all ones
  std::vector<float> dataA(M * K, 1.0f);

  // PackedB [64, 1]: nibble=12 for both N values -> dequant = 12-8 = 4
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k) {
    packedB[k * (N / 2)] = packNibbles(12, 12); // both nibbles = 12
  }

  // Scales [2, 2] as [K/32, N]: same pattern as Q8 test
  // Row 0: {0.5, 2.0}, Row 1: {1.0, 0.25}
  std::vector<uint16_t> scales = {
      f32_to_f16(0.5f),
      f32_to_f16(2.0f), // block 0 scales for n=0, n=1
      f32_to_f16(1.0f),
      f32_to_f16(0.25f) // block 1 scales for n=0, n=1
  };

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB =
      runtime_->createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS =
      runtime_->createTensor({blocksK, N}, DataType::Float16, scales.data());

  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // Dequant value = (12-8) * scale = 4 * scale
  // C[0][0] = 32 * (4 * 0.5) + 32 * (4 * 1.0) = 64 + 128 = 192
  // C[0][1] = 32 * (4 * 2.0) + 32 * (4 * 0.25) = 256 + 32 = 288
  ASSERT_NEAR(output[0], 192.0f, 1.0f) << "C[0][0] mismatch";
  ASSERT_NEAR(output[1], 288.0f, 1.0f) << "C[0][1] mismatch";
}

TEST_F(MatrixOpsTest, MatMulQ4_VsReference) {
  // Compare matmulQ4 GPU output against CPU reference.
  // B is [K, N/2] packed nibbles, scales [K/32, N].
  const uint32_t M = 2, K = 64, N = 4;
  const uint32_t blocksPerRow = K / 32;

  auto dataA = generateTestData<float>(M * K, 42);

  // Generate nibble values in range [0, 15] and pack into [K, N/2] bytes
  // Also keep unpacked nibbles for CPU reference
  std::vector<uint8_t> nibbles(K * N);
  for (size_t i = 0; i < nibbles.size(); ++i) {
    nibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
  }

  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t n = 0; n < N; n += 2) {
      uint8_t lo = nibbles[k * N + n];
      uint8_t hi = nibbles[k * N + n + 1];
      packedB[k * (N / 2) + n / 2] = packNibbles(lo, hi);
    }
  }

  // Generate scales [blocksPerRow, N]: small positive floats
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  }
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i) {
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  }

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB =
      runtime_->createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16,
                                     scaleF16.data());
  auto bufC = runtime_->ops().matmul(bufA, bufB, bufS);

  std::vector<float> gpuOutput(M * N);
  runtime_->copyFromTensor(bufC, gpuOutput.data(), M * N * sizeof(float));

  // CPU reference: C[m][n] = sum_k(A[m][k] * (nibble[k][n] - 8) *
  // scale[k/32][n])
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        float a = dataA[m * K + k];
        float dequant =
            static_cast<float>(static_cast<int>(nibbles[k * N + n]) - 8);
        float s = scaleFloats[(k / 32) * N + n];
        expected += a * dequant * s;
      }
      ASSERT_NEAR(gpuOutput[m * N + n], expected,
                  std::abs(expected) * 0.01f + 0.1f)
          << "Mismatch at C[" << m << "][" << n << "]";
    }
  }
}

TEST_F(MatrixOpsTest, MatMulQ8_AllVariants_VsReference) {
  const uint32_t M = 1, K = 512, N = 16;
  const uint32_t blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<int8_t> dataB(K * N);
  for (size_t i = 0; i < dataB.size(); ++i)
    dataB[i] = static_cast<int8_t>((i * 7 + 3) % 21 - 10);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i)
    scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i)
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Int8, dataB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  // CPU reference
  std::vector<float> ref(M * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        ref[m * N + n] += dataA[m * K + k] * static_cast<float>(dataB[k * N + n]) *
                          scaleFloats[(k / 32) * N + n];
  int tested = 0;
  for (int vi = 0; vi < kMatMulQ8VariantCount; ++vi) {
    if (!getCompiledMatMulQ8(vi, DataType::Float32, DataType::Float16,
                             DataType::Float32)
             .has_value())
      continue;
    std::string name = getMatMulQ8VariantName(vi);
    // Skip variants that need a specialized packed-weight layout not produced
    // by this plain [K,N] int8 + f16-scale reference: integer-dot ("Dot") and
    // cooperative-matrix ("CoopMat") kernels expect pre-packed operands and are
    // exercised via their own dedicated paths.
    if (name.find("Dot") != std::string::npos ||
        name.find("CoopMat") != std::string::npos)
      continue;
    SCOPED_TRACE(std::string("Q8 variant ") + std::to_string(vi) + " " + name);
    auto bufC = runtime_->ops().matmul(bufA, bufB, bufS, vi);
    std::vector<float> got(M * N);
    runtime_->copyFromTensor(bufC, got.data(), M * N * sizeof(float));
    for (uint32_t i = 0; i < M * N; ++i)
      ASSERT_NEAR(got[i], ref[i], std::abs(ref[i]) * 0.02f + 0.1f) << "i=" << i;
    ++tested;
  }
  ASSERT_GT(tested, 1) << "expected multiple compiled Q8 variants";
}

TEST_F(MatrixOpsTest, MatMulQ4_AllVariants_VsReference) {
  const uint32_t M = 1, K = 64, N = 4;
  const uint32_t blocksPerRow = K / 32;
  auto dataA = generateTestData<float>(M * K, 42);
  std::vector<uint8_t> nibbles(K * N);
  for (size_t i = 0; i < nibbles.size(); ++i)
    nibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
  std::vector<uint8_t> packedB(K * (N / 2));
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; n += 2)
      packedB[k * (N / 2) + n / 2] = packNibbles(nibbles[k * N + n], nibbles[k * N + n + 1]);
  std::vector<float> scaleFloats(blocksPerRow * N);
  for (size_t i = 0; i < scaleFloats.size(); ++i)
    scaleFloats[i] = 0.1f + 0.05f * static_cast<float>(i);
  std::vector<uint16_t> scaleF16(scaleFloats.size());
  for (size_t i = 0; i < scaleFloats.size(); ++i)
    scaleF16[i] = f32_to_f16(scaleFloats[i]);
  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N / 2}, DataType::Int8, packedB.data());
  auto bufS = runtime_->createTensor({blocksPerRow, N}, DataType::Float16, scaleF16.data());
  std::vector<float> ref(M * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        ref[m * N + n] += dataA[m * K + k] *
                          static_cast<float>(static_cast<int>(nibbles[k * N + n]) - 8) *
                          scaleFloats[(k / 32) * N + n];
  int tested = 0;
  for (int vi = 0; vi < kMatMulQ4VariantCount; ++vi) {
    if (!getCompiledMatMulQ4(vi, DataType::Float32, DataType::Float16,
                             DataType::Float32)
             .has_value())
      continue;
    std::string name = getMatMulQ4VariantName(vi);
    // Skip variants that need a specialized packed-weight layout not produced
    // by this plain [K,N] int8 + f16-scale reference: integer-dot ("Dot") and
    // cooperative-matrix ("CoopMat") kernels expect pre-packed operands and are
    // exercised via their own dedicated paths.
    if (name.find("Dot") != std::string::npos ||
        name.find("CoopMat") != std::string::npos)
      continue;
    SCOPED_TRACE(std::string("Q4 variant ") + std::to_string(vi) + " " + name);
    auto bufC = runtime_->ops().matmul(bufA, bufB, bufS, vi);
    std::vector<float> got(M * N);
    runtime_->copyFromTensor(bufC, got.data(), M * N * sizeof(float));
    for (uint32_t i = 0; i < M * N; ++i)
      ASSERT_NEAR(got[i], ref[i], std::abs(ref[i]) * 0.02f + 0.1f) << "i=" << i;
    ++tested;
  }
  ASSERT_GT(tested, 0) << "expected at least one compiled Q4 variant";
}

// =========================================================================
// Dequantization tests
// =========================================================================

TEST_F(MatrixOpsTest, Dequant_BF16) {
  // BF16 is upper 16 bits of F32. Test round-trip: F32 → BF16 bits → GPU
  // dequant → F32.
  const uint32_t rows = 2, cols = 4;
  const uint32_t n_elements = rows * cols;

  // Source F32 values
  std::vector<float> srcValues = {1.0f,  -2.0f,   0.5f,   0.0f,
                                  3.14f, -0.125f, 100.0f, 42.0f};

  // Convert to BF16 (truncate lower 16 bits of F32)
  std::vector<uint16_t> bf16Data(n_elements);
  std::vector<float> expectedF32(n_elements);
  for (uint32_t i = 0; i < n_elements; ++i) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &srcValues[i], sizeof(float));
    bf16Data[i] = static_cast<uint16_t>(f32_bits >> 16);
    // Expected: BF16 truncation loses lower mantissa bits
    uint32_t reconstructed = static_cast<uint32_t>(bf16Data[i]) << 16;
    std::memcpy(&expectedF32[i], &reconstructed, sizeof(float));
  }

  // Upload raw BF16 bytes as Int8 tensor
  auto rawTensor = runtime_->createTensor(
      {static_cast<uint32_t>(n_elements * 2)}, DataType::Int8, bf16Data.data());

  auto result = runtime_->ops().dequantize(
      rawTensor, static_cast<uint32_t>(cut::DequantFormat::BF16), rows, cols);

  std::vector<float> gpuOutput(n_elements);
  runtime_->copyFromTensor(result, gpuOutput.data(),
                           n_elements * sizeof(float));

  for (uint32_t i = 0; i < n_elements; ++i) {
    ASSERT_NEAR(gpuOutput[i], expectedF32[i], 1e-6f)
        << "BF16 dequant mismatch at index " << i;
  }
}

// Helper: convert F32 to F16 bits (for dequant test scale encoding)
static float f16_bits_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;
  if (exponent == 0) {
    if (mantissa == 0) {
      float result;
      std::memcpy(&result, &sign, sizeof(float));
      return result;
    }
    float result = static_cast<float>(mantissa) * 5.960464477539063e-08f;
    return (h & 0x8000) ? -result : result;
  }
  if (exponent == 31) {
    uint32_t f32_bits = sign | 0x7F800000u | (mantissa << 13);
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
  }
  uint32_t f32_bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  float result;
  std::memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

TEST_F(MatrixOpsTest, Dequant_Q4K) {
  // Q4_K: 256 elements per super-block, 144 bytes per block.
  // Layout: [d:f16][dmin:f16][scales:12B][qs:128B]
  // Test with 1 row, 256 cols (one super-block).
  const uint32_t rows = 1, cols = 256;
  const size_t blockBytes = 144;

  std::vector<uint8_t> rawBlock(blockBytes, 0);

  // Set d = 1.0 (f16), dmin = 0.0 (f16)
  uint16_t d_f16 = f32_to_f16(1.0f);
  uint16_t dmin_f16 = f32_to_f16(0.0f);
  std::memcpy(rawBlock.data(), &d_f16, 2);
  std::memcpy(rawBlock.data() + 2, &dmin_f16, 2);

  // Set all sub-block scales to 1, mins to 0 (in packed 12-byte format)
  // For j < 4: scales[j] & 63 = 1, mins[j+4] & 63 = 0
  for (int j = 0; j < 4; ++j) {
    rawBlock[4 + j] = 1;     // scale = 1 (lower 6 bits)
    rawBlock[4 + j + 4] = 0; // min = 0
  }
  // For j >= 4: packed into bytes 8-11
  for (int j = 4; j < 8; ++j) {
    rawBlock[4 + j + 4] = 1; // lower nibble = scale, upper nibble = min
  }

  // Set nibbles: value i%16 for element i
  // qs is 128 bytes starting at offset 16
  uint8_t *qs = rawBlock.data() + 16;
  for (int i = 0; i < 128; ++i) {
    qs[i] = static_cast<uint8_t>(((i % 8) & 0xF) | (((i % 8 + 1) & 0xF) << 4));
  }

  auto rawTensor =
      runtime_->createTensor({static_cast<uint32_t>(rawBlock.size())},
                             DataType::Int8, rawBlock.data());

  auto result = runtime_->ops().dequantize(
      rawTensor, static_cast<uint32_t>(cut::DequantFormat::Q4_K), rows, cols);

  std::vector<float> gpuOutput(rows * cols);
  runtime_->copyFromTensor(result, gpuOutput.data(),
                           rows * cols * sizeof(float));

  // CPU reference dequant
  float d = f16_bits_to_f32(d_f16);

  // Verify first few elements are non-NaN and finite
  for (uint32_t i = 0; i < cols; ++i) {
    ASSERT_FALSE(std::isnan(gpuOutput[i]))
        << "Q4_K dequant produced NaN at index " << i;
    ASSERT_FALSE(std::isinf(gpuOutput[i]))
        << "Q4_K dequant produced Inf at index " << i;
  }

  // Verify that d=1.0 with scale=1, min=0 gives value = 1.0 * 1 * nibble
  // First 32 elements use lower nibble of qs[0..31]
  for (uint32_t i = 0; i < 32; ++i) {
    float nibble = static_cast<float>(qs[i] & 0xF);
    float expected = d * 1.0f * nibble; // d * scale * nibble - dmin * min
    ASSERT_NEAR(gpuOutput[i], expected, 0.01f)
        << "Q4_K mismatch at index " << i;
  }
}

TEST_F(MatrixOpsTest, Dequant_Q6K) {
  // Q6_K: 256 elements per super-block, 210 bytes per block.
  // Layout: [ql:128B][qh:64B][scales:16B][d:f16]
  // Test with 1 row, 256 cols.
  const uint32_t rows = 1, cols = 256;
  const size_t blockBytes = 210;

  std::vector<uint8_t> rawBlock(blockBytes, 0);

  // Set d = 1.0 at offset 208
  uint16_t d_f16 = f32_to_f16(1.0f);
  std::memcpy(rawBlock.data() + 208, &d_f16, 2);

  // Set all int8 scales to 1 (at offset 192, 16 bytes)
  for (int i = 0; i < 16; ++i) {
    rawBlock[192 + i] = 1;
  }

  // Set ql (lower 4 bits) at offset 0: all values = 5 (lower nibble)
  for (int i = 0; i < 128; ++i) {
    rawBlock[i] = 0x55; // lower=5, upper=5
  }

  // Set qh (upper 2 bits) at offset 128: all zeros (so 6-bit value = lower 4
  // bits)
  // Already zero from initialization

  auto rawTensor =
      runtime_->createTensor({static_cast<uint32_t>(rawBlock.size())},
                             DataType::Int8, rawBlock.data());

  auto result = runtime_->ops().dequantize(
      rawTensor, static_cast<uint32_t>(cut::DequantFormat::Q6_K), rows, cols);

  std::vector<float> gpuOutput(rows * cols);
  runtime_->copyFromTensor(result, gpuOutput.data(),
                           rows * cols * sizeof(float));

  // With d=1.0, scale=1, all ql=5, qh=0:
  // 6-bit value = 5 | (0 << 4) = 5
  // dequant = d * scale * (6bit - 32) = 1.0 * 1 * (5 - 32) = -27.0
  for (uint32_t i = 0; i < 32; ++i) {
    ASSERT_FALSE(std::isnan(gpuOutput[i]))
        << "Q6_K dequant produced NaN at index " << i;
    ASSERT_NEAR(gpuOutput[i], -27.0f, 0.01f) << "Q6_K mismatch at index " << i;
  }
}

// Registry-driven: verifies the "transpose/square" case.
TEST_F(MatrixOpsTest, Transpose_Square) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "transpose/square")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "transpose/rectangular" case.
TEST_F(MatrixOpsTest, Transpose_Rectangular) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "transpose/rectangular")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "dot/basic" case.
TEST_F(MatrixOpsTest, Dot_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "dot/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "dot/larger" case.
TEST_F(MatrixOpsTest, Dot_LargerVectors) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "dot/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Global Norm Test
// ============================================================================

class NormTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(NormTest, Norm_KnownValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "norm/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(NormTest, Norm_VariousSizes) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "norm/varioussizes")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(NormTest, Norm_MultiDimensional) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "norm/multidim")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Convolution Operation Tests
// ============================================================================

class ConvolutionTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }

  // CPU reference for conv1d: input [N,C_in,L_in], weight [C_out,C_in,kL]
  std::vector<float> conv1dRef(const std::vector<float> &input,
                               const std::vector<float> &weight,
                               uint32_t N,
                               uint32_t C_in,
                               uint32_t L_in,
                               uint32_t C_out,
                               uint32_t kL,
                               uint32_t stride,
                               uint32_t padding) {
    uint32_t L_out = (L_in + 2 * padding - kL) / stride + 1;
    std::vector<float> output(N * C_out * L_out, 0.0f);

    for (uint32_t n = 0; n < N; n++) {
      for (uint32_t co = 0; co < C_out; co++) {
        for (uint32_t lo = 0; lo < L_out; lo++) {
          float sum = 0.0f;
          for (uint32_t ci = 0; ci < C_in; ci++) {
            for (uint32_t k = 0; k < kL; k++) {
              int li =
                  static_cast<int>(lo * stride + k) - static_cast<int>(padding);
              if (li < 0 || li >= static_cast<int>(L_in))
                continue;
              sum += input[n * C_in * L_in + ci * L_in + li] *
                     weight[co * C_in * kL + ci * kL + k];
            }
          }
          output[n * C_out * L_out + co * L_out + lo] = sum;
        }
      }
    }
    return output;
  }

  // CPU reference for conv2d: input [N,C_in,H_in,W_in], weight
  // [C_out,C_in,kH,kW]
  std::vector<float> conv2dRef(const std::vector<float> &input,
                               const std::vector<float> &weight,
                               uint32_t N,
                               uint32_t C_in,
                               uint32_t H_in,
                               uint32_t W_in,
                               uint32_t C_out,
                               uint32_t kH,
                               uint32_t kW,
                               uint32_t strideH,
                               uint32_t strideW,
                               uint32_t padH,
                               uint32_t padW) {
    uint32_t H_out = (H_in + 2 * padH - kH) / strideH + 1;
    uint32_t W_out = (W_in + 2 * padW - kW) / strideW + 1;
    std::vector<float> output(N * C_out * H_out * W_out, 0.0f);

    for (uint32_t n = 0; n < N; n++) {
      for (uint32_t co = 0; co < C_out; co++) {
        for (uint32_t ho = 0; ho < H_out; ho++) {
          for (uint32_t wo = 0; wo < W_out; wo++) {
            float sum = 0.0f;
            for (uint32_t ci = 0; ci < C_in; ci++) {
              for (uint32_t kh = 0; kh < kH; kh++) {
                for (uint32_t kw = 0; kw < kW; kw++) {
                  int hi = static_cast<int>(ho * strideH + kh) -
                           static_cast<int>(padH);
                  int wi = static_cast<int>(wo * strideW + kw) -
                           static_cast<int>(padW);
                  if (hi < 0 || hi >= static_cast<int>(H_in) || wi < 0 ||
                      wi >= static_cast<int>(W_in))
                    continue;
                  sum +=
                      input[n * C_in * H_in * W_in + ci * H_in * W_in +
                            hi * W_in + wi] *
                      weight[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
                }
              }
            }
            output[n * C_out * H_out * W_out + co * H_out * W_out + ho * W_out +
                   wo] = sum;
          }
        }
      }
    }
    return output;
  }
};

TEST_F(ConvolutionTest, Conv1D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv1D_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv1D_WithStride) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/stride")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv1D_MultiChannel) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/multichannel")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2D_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2D_WithStride) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/stride")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2D_MultiChannel) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/multichannel")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2D_StridePadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/stridepadding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Tensor Creation Operation Tests
// ============================================================================

class TensorCreationTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(TensorCreationTest, Zeros_Float32) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->ops().full({elements}, 0.0f, dtype);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      ASSERT_EQ(output[i], 0.0f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Ones_Float32) {
  const DataType dtype = DataType::Float32;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->ops().full({elements}, 1.0f, dtype);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      ASSERT_EQ(output[i], 1.0f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Full_Float32) {
  const DataType dtype = DataType::Float32;
  const float fillValue = 3.14f;

  for (uint32_t elements : {4u, 8u, 16u, 100u}) {
    SCOPED_TRACE("elements=" + std::to_string(elements));

    auto bufferOut = runtime_->ops().full({elements}, fillValue, dtype);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(),
                             elements * sizeof(float));

    for (uint32_t i = 0; i < elements; ++i) {
      ASSERT_NEAR(output[i], fillValue, 1e-5f) << "Mismatch at index " << i;
    }
  }
}

TEST_F(TensorCreationTest, Arange_Float32) {
  const DataType dtype = DataType::Float32;

  // arange(0, 8, 1) -> [0, 1, 2, 3, 4, 5, 6, 7]
  const uint32_t elements = 8;
  const float start = 0.0f;
  const float step = 1.0f;
  const float end = start + static_cast<float>(elements) * step;

  auto bufferOut = runtime_->ops().arange(start, end, step, dtype);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Arange_WithStep) {
  const DataType dtype = DataType::Float32;

  // arange(1, ?, 0.5) -> [1.0, 1.5, 2.0, 2.5, ...]
  const uint32_t elements = 8;
  const float start = 1.0f;
  const float step = 0.5f;
  const float end = start + static_cast<float>(elements) * step;

  auto bufferOut = runtime_->ops().arange(start, end, step, dtype);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Linspace_Float32) {
  const DataType dtype = DataType::Float32;

  // linspace(0, 1, 8) -> [0.0, 0.143, ..., 1.0]
  const uint32_t elements = 8;
  const float start = 0.0f;
  const float end = 1.0f;

  auto bufferOut =
      runtime_->ops().linspace(start, end, static_cast<int>(elements), dtype);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  const float step = (end - start) / static_cast<float>(elements - 1);
  for (uint32_t i = 0; i < elements; ++i) {
    float expected = start + static_cast<float>(i) * step;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Zeros_MultiDimensional) {
  const DataType dtype = DataType::Float32;

  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = totalElements(shape);

  auto bufferOut = runtime_->ops().full(shape, 0.0f, dtype);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    ASSERT_EQ(output[i], 0.0f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Ones_MultiDimensional) {
  const DataType dtype = DataType::Float32;

  std::vector<uint32_t> shape = {3, 4};
  const uint32_t elements = totalElements(shape);

  auto bufferOut = runtime_->ops().full(shape, 1.0f, dtype);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), elements * sizeof(float));

  for (uint32_t i = 0; i < elements; ++i) {
    ASSERT_EQ(output[i], 1.0f) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Zeros_Int32) {
  const DataType dtype = DataType::Int32;

  const uint32_t elements = 16;
  int32_t fillVal = 0;
  auto bufferOut =
      runtime_->ops().full({elements}, DataReference(fillVal), dtype);

  std::vector<int32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(int32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    ASSERT_EQ(output[i], 0) << "Mismatch at index " << i;
  }
}

TEST_F(TensorCreationTest, Ones_UInt32) {
  const DataType dtype = DataType::UInt32;

  const uint32_t elements = 16;
  uint32_t fillVal = 1;
  auto bufferOut =
      runtime_->ops().full({elements}, DataReference(fillVal), dtype);

  std::vector<uint32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    ASSERT_EQ(output[i], 1u) << "Mismatch at index " << i;
  }
}

// ============================================================================
// Temporary Tensor Deallocation Tests
// ============================================================================

// Verify that temporary tensors created by ops are deallocated after use.
// Operations like reduce, dot, softmax, etc. create intermediate
// GPU buffers internally. These must be freed when their handles go out of
// scope so that GPU memory doesn't leak.
//
// Note: GPU operations are batched in a command buffer which holds references
// to bound buffers. flush() must be called to submit and release the command
// buffer before checking that buffers are freed.

TEST_F(VulkanBackendTest, TemporaryTensors_BinaryOp) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().binaryOp(BinaryAdd, a, b);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  // After scope, the result handle is destroyed and buffer should be freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_UnaryOp) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().unaryOp(UnaryNeg, a);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Reduce) {
  // reduce returns a 1-element output tensor. The buffer is freed when
  // the returned handle goes out of scope.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().reduce(ReduceSum, a);
    float sum = 0.0f;
    runtime_->copyFromTensor(result, &sum, sizeof(float));
    ASSERT_FLOAT_EQ(sum, 10.0f);
  }
  // The output buffer should have been freed after result goes out of scope
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Dot) {
  // dot creates a temporary 1-element output buffer internally
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().dot(a, b);
    runtime_->flush();
    // dot now returns a {1} tensor; partials tensor should be freed
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  // After scope, the returned output is also freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_VarianceScalar) {
  // variance calls reduce internally (temporary buffer)
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto var = runtime_->ops().variance(a, 0);
    runtime_->flush();
    // variance now returns a {1} tensor
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  // After scope, the returned output is also freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_VarianceDim) {
  // variance with dim creates intermediate tensors via GPU ops.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().variance(a, 0, 1);
    runtime_->flush();
    // Only the returned output should exist (intermediates freed)
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  // After scope, the returned output is also freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Softmax) {
  // softmax creates a maxHandle intermediate tensor via reduce
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().softmax(a, 1);
    runtime_->flush();
    // Only the returned output should exist (maxHandle intermediate freed)
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  // After scope, all buffers freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_LogSoftmax) {
  // logSoftmax creates a maxHandle intermediate tensor via reduce
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().logSoftmax(a, 1);
    runtime_->flush();
    // Only the returned output should exist (maxHandle intermediate freed)
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_ReduceWithDim) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().reduce(ReduceSum, a, 1);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Reshape) {
  // reshape creates a new output buffer and uses encodeCopy
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().reshape(a, {3, 2});
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_ChainedOps) {
  // Chained operations: each intermediate should be freed when no longer held
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto neg = runtime_->ops().unaryOp(UnaryNeg, a);
    auto added = runtime_->ops().binaryOp(BinaryAdd, neg, 10.0f);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 2);
  }
  // Both intermediate buffers should be freed
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Matmul) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({2, 2}, DataType::Float32, data.data());
  auto b = runtime_->createTensor({2, 2}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().matmul(a, b);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

TEST_F(VulkanBackendTest, TemporaryTensors_Transpose) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());
  runtime_->flush();
  size_t before = runtime_->bufferCount();

  {
    auto result = runtime_->ops().transpose(a);
    runtime_->flush();
    ASSERT_EQ(runtime_->bufferCount(), before + 1);
  }
  ASSERT_EQ(runtime_->bufferCount(), before);
}

// ============================================================================
// Pooling Operation Tests
// ============================================================================

class PoolingTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }

  // CPU reference for max_pool2d: input [N,C,H_in,W_in]
  std::vector<float> maxPool2dRef(const std::vector<float> &input,
                                  uint32_t N,
                                  uint32_t C,
                                  uint32_t H_in,
                                  uint32_t W_in,
                                  uint32_t kH,
                                  uint32_t kW,
                                  uint32_t sH,
                                  uint32_t sW,
                                  uint32_t pH,
                                  uint32_t pW) {
    uint32_t H_out = (H_in + 2 * pH - kH) / sH + 1;
    uint32_t W_out = (W_in + 2 * pW - kW) / sW + 1;
    std::vector<float> output(N * C * H_out * W_out);

    for (uint32_t n = 0; n < N; n++) {
      for (uint32_t c = 0; c < C; c++) {
        for (uint32_t ho = 0; ho < H_out; ho++) {
          for (uint32_t wo = 0; wo < W_out; wo++) {
            float maxVal = -std::numeric_limits<float>::infinity();
            for (uint32_t kh = 0; kh < kH; kh++) {
              for (uint32_t kw = 0; kw < kW; kw++) {
                int hi = static_cast<int>(ho * sH + kh) - static_cast<int>(pH);
                int wi = static_cast<int>(wo * sW + kw) - static_cast<int>(pW);
                if (hi >= 0 && hi < static_cast<int>(H_in) && wi >= 0 &&
                    wi < static_cast<int>(W_in)) {
                  float val = input[n * C * H_in * W_in + c * H_in * W_in +
                                    hi * W_in + wi];
                  maxVal = std::max(maxVal, val);
                }
              }
            }
            output[n * C * H_out * W_out + c * H_out * W_out + ho * W_out +
                   wo] = maxVal;
          }
        }
      }
    }
    return output;
  }

  // CPU reference for avg_pool2d: input [N,C,H_in,W_in]
  std::vector<float> avgPool2dRef(const std::vector<float> &input,
                                  uint32_t N,
                                  uint32_t C,
                                  uint32_t H_in,
                                  uint32_t W_in,
                                  uint32_t kH,
                                  uint32_t kW,
                                  uint32_t sH,
                                  uint32_t sW,
                                  uint32_t pH,
                                  uint32_t pW) {
    uint32_t H_out = (H_in + 2 * pH - kH) / sH + 1;
    uint32_t W_out = (W_in + 2 * pW - kW) / sW + 1;
    std::vector<float> output(N * C * H_out * W_out);

    for (uint32_t n = 0; n < N; n++) {
      for (uint32_t c = 0; c < C; c++) {
        for (uint32_t ho = 0; ho < H_out; ho++) {
          for (uint32_t wo = 0; wo < W_out; wo++) {
            float sum = 0.0f;
            uint32_t count = 0;
            for (uint32_t kh = 0; kh < kH; kh++) {
              for (uint32_t kw = 0; kw < kW; kw++) {
                int hi = static_cast<int>(ho * sH + kh) - static_cast<int>(pH);
                int wi = static_cast<int>(wo * sW + kw) - static_cast<int>(pW);
                if (hi >= 0 && hi < static_cast<int>(H_in) && wi >= 0 &&
                    wi < static_cast<int>(W_in)) {
                  sum += input[n * C * H_in * W_in + c * H_in * W_in +
                               hi * W_in + wi];
                  count++;
                }
              }
            }
            output[n * C * H_out * W_out + c * H_out * W_out + ho * W_out +
                   wo] = count > 0 ? sum / count : 0.0f;
          }
        }
      }
    }
    return output;
  }
};

TEST_F(PoolingTest, MaxPool2D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "maxpool/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, MaxPool2D_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "maxpool/padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, MaxPool2D_MultiChannel) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "maxpool/multichannel")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, AvgPool2D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, AvgPool2D_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, AvgPool2D_MultiChannel) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/multichannel")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, AdaptiveAvgPool2D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/adaptive_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PoolingTest, AdaptiveAvgPool2D_GlobalPool) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/adaptive_global")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Layer Normalization Tests
// ============================================================================

class LayerNormTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }

  // CPU reference for layer norm
  std::vector<float> layerNormRef(const std::vector<float> &input,
                                  size_t outerSize,
                                  size_t normSize,
                                  const std::vector<float> *weight,
                                  const std::vector<float> *bias,
                                  float eps) {
    std::vector<float> result(input.size());
    for (size_t o = 0; o < outerSize; ++o) {
      size_t base = o * normSize;

      double sum = 0.0;
      for (size_t i = 0; i < normSize; ++i)
        sum += input[base + i];
      float mean = static_cast<float>(sum / normSize);

      double varSum = 0.0;
      for (size_t i = 0; i < normSize; ++i) {
        double diff = input[base + i] - mean;
        varSum += diff * diff;
      }
      float invStd =
          1.0f / std::sqrt(static_cast<float>(varSum / normSize) + eps);

      for (size_t i = 0; i < normSize; ++i) {
        float normalized = (input[base + i] - mean) * invStd;
        if (weight)
          normalized *= (*weight)[i];
        if (bias)
          normalized += (*bias)[i];
        result[base + i] = normalized;
      }
    }
    return result;
  }
};

TEST_F(LayerNormTest, Basic_NoWeightBias) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "layernorm/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(LayerNormTest, WithWeightAndBias) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "layernorm/weightbias")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(LayerNormTest, HigherDimensional) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "layernorm/higherdim")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Batch Normalization Tests
// ============================================================================

class BatchNormTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }

  // CPU reference for batch norm (inference mode)
  std::vector<float> batchNormRef(const std::vector<float> &input,
                                  const std::vector<float> &runningMean,
                                  const std::vector<float> &runningVar,
                                  const std::vector<float> *weight,
                                  const std::vector<float> *bias,
                                  uint32_t N,
                                  uint32_t C,
                                  size_t spatialSize,
                                  float eps) {
    std::vector<float> result(input.size());
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t c = 0; c < C; ++c) {
        float invStd = 1.0f / std::sqrt(runningVar[c] + eps);
        float scale = weight ? (*weight)[c] * invStd : invStd;
        float shift = bias ? (*bias)[c] - runningMean[c] * scale
                           : -runningMean[c] * scale;
        size_t base = (n * C + c) * spatialSize;
        for (size_t s = 0; s < spatialSize; ++s) {
          result[base + s] = input[base + s] * scale + shift;
        }
      }
    }
    return result;
  }
};

TEST_F(BatchNormTest, Basic_NoWeightBias) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "batchnorm/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BatchNormTest, WithWeightAndBias) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "batchnorm/weightbias")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(BatchNormTest, SingleSpatial) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "batchnorm/singlespatial")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Embedding Operation Tests
// ============================================================================

class EmbeddingTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(EmbeddingTest, Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "embedding/basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(EmbeddingTest, LargerTable) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "embedding/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(EmbeddingTest, RepeatedIndices) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "embedding/repeated")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Pad Operation Tests
// ============================================================================

class PadTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(PadTest, Pad1D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "pad/1d_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PadTest, Pad2D_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "pad/2d_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PadTest, Pad2D_MultipleDims) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "pad/2d_multidims")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PadTest, Pad4D_Image) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "pad/4d_image")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(PadTest, PadWithFillValue) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "pad/fillvalue")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Transpose Variant Tests
// ============================================================================

// Registry-driven: verifies the "transpose/variants_square" case.
TEST_F(MatrixOpsTest, TransposeVariants_Square) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "transpose/variants_square")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// Registry-driven: verifies the "transpose/variants_rectangular" case.
TEST_F(MatrixOpsTest, TransposeVariants_Rectangular) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "transpose/variants_rectangular")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Conv1D Variant Tests
// ============================================================================

TEST_F(ConvolutionTest, Conv1DVariants_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/variants_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv1DVariants_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv1d/variants_padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// Conv2D Variant Tests
// ============================================================================

TEST_F(ConvolutionTest, Conv2DVariants_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/variants_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(ConvolutionTest, Conv2DVariants_WithPadding) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "conv2d/variants_padding")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// MaxPool2D Variant Tests
// ============================================================================

TEST_F(PoolingTest, MaxPool2DVariants_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "maxpool/variants_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// AvgPool2D Variant Tests
// ============================================================================

TEST_F(PoolingTest, AvgPool2DVariants_Basic) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "avgpool/variants_basic")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ============================================================================
// ReduceDim Variant Tests
// ============================================================================

TEST_F(VulkanBackendTest, ReduceDimVariants_Dim0) {
  const DataType dtype = DataType::Float32;
  const uint32_t M = 16, N = 8;

  auto data = generateTestData<float>(M * N, 42);

  // CPU reference: sum along dim 0 → shape [N]
  std::vector<float> expected(N, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] += data[i * N + j];

  for (int vi = 0; vi < 2; ++vi) { // Only generic variants (Naive, Shared)
    SCOPED_TRACE(std::string("Variant: ") + getReduceDimVariantName(vi));
    auto buf = runtime_->createTensor({M, N}, dtype, data.data());
    auto bufOut = runtime_->ops().reduce(ReduceSum, buf, 0, vi);

    std::vector<float> output(N);
    runtime_->copyFromTensor(bufOut, output.data(), N * sizeof(float));

    for (uint32_t j = 0; j < N; ++j) {
      ASSERT_NEAR(output[j], expected[j], std::abs(expected[j]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << j;
    }
  }
}

TEST_F(VulkanBackendTest, ReduceDimVariants_Dim1) {
  const DataType dtype = DataType::Float32;
  const uint32_t M = 8, N = 16;

  auto data = generateTestData<float>(M * N, 42);

  // CPU reference: sum along dim 1 → shape [M]
  std::vector<float> expected(M, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      expected[i] += data[i * N + j];

  for (int vi = 0; vi < 2; ++vi) { // Only generic variants (Naive, Shared)
    SCOPED_TRACE(std::string("Variant: ") + getReduceDimVariantName(vi));
    auto buf = runtime_->createTensor({M, N}, dtype, data.data());
    auto bufOut = runtime_->ops().reduce(ReduceSum, buf, 1, vi);

    std::vector<float> output(M);
    runtime_->copyFromTensor(bufOut, output.data(), M * sizeof(float));

    for (uint32_t i = 0; i < M; ++i) {
      ASSERT_NEAR(output[i], expected[i], std::abs(expected[i]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << i;
    }
  }
}

TEST_F(VulkanBackendTest, ReduceDimVariants_Mean) {
  const DataType dtype = DataType::Float32;
  const uint32_t M = 16, N = 8;

  auto data = generateTestData<float>(M * N, 42);

  // CPU reference: mean along dim 0 → shape [N]
  std::vector<float> expected(N, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] += data[i * N + j];
  for (uint32_t j = 0; j < N; ++j)
    expected[j] /= static_cast<float>(M);

  for (int vi = 0; vi < 2; ++vi) { // Only generic variants (Naive, Shared)
    SCOPED_TRACE(std::string("Variant: ") + getReduceDimVariantName(vi));
    auto buf = runtime_->createTensor({M, N}, dtype, data.data());
    auto bufOut = runtime_->ops().reduce(ReduceMean, buf, 0, vi);

    std::vector<float> output(N);
    runtime_->copyFromTensor(bufOut, output.data(), N * sizeof(float));

    for (uint32_t j = 0; j < N; ++j) {
      ASSERT_NEAR(output[j], expected[j], std::abs(expected[j]) * 1e-4f + 1e-5f)
          << "Mismatch at index " << j;
    }
  }
}

TEST_F(VulkanBackendTest, ReduceDimVariants_Max) {
  const DataType dtype = DataType::Float32;
  const uint32_t M = 16, N = 8;

  auto data = generateTestData<float>(M * N, 42);

  // CPU reference: max along dim 0 → shape [N]
  std::vector<float> expected(N, -std::numeric_limits<float>::max());
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] = std::max(expected[j], data[i * N + j]);

  for (int vi = 0; vi < 2; ++vi) { // Only generic variants (Naive, Shared)
    SCOPED_TRACE(std::string("Variant: ") + getReduceDimVariantName(vi));
    auto buf = runtime_->createTensor({M, N}, dtype, data.data());
    auto bufOut = runtime_->ops().reduce(ReduceMax, buf, 0, vi);

    std::vector<float> output(N);
    runtime_->copyFromTensor(bufOut, output.data(), N * sizeof(float));

    for (uint32_t j = 0; j < N; ++j) {
      ASSERT_NEAR(output[j], expected[j], std::abs(expected[j]) * 1e-5f + 1e-5f)
          << "Mismatch at index " << j;
    }
  }
}

// ============================================================================
// BinaryVecScalar with Handle Tests
// ============================================================================

class BinaryVecScalarHandleTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

// Test binary vec-scalar operations where second operand is a handle of size 1
TEST_F(BinaryVecScalarHandleTest, BinaryAdd_Handle_1D) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryAdd;

  std::vector<uint32_t> shape = {16};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 2.5f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  // Use binaryOp with compute handle of size 1
  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = dataA[i] + scalarValue;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(BinaryVecScalarHandleTest, BinaryMul_Handle_2D) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  std::vector<uint32_t> shape = {4, 8};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 3.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = dataA[i] * scalarValue;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(BinaryVecScalarHandleTest, BinarySub_Handle_3D) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinarySub;

  std::vector<uint32_t> shape = {2, 3, 4};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 1.5f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = dataA[i] - scalarValue;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(BinaryVecScalarHandleTest, BinaryDiv_Handle_4D) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryDiv;

  std::vector<uint32_t> shape = {2, 2, 3, 4};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 2.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = dataA[i] / scalarValue;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

// Test with non-aligned innermost dimensions
TEST_F(BinaryVecScalarHandleTest, BinaryMax_Handle_NonAligned) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMax;

  // Test innermost dimensions that are not multiples of 4
  for (uint32_t innerDim : {1u, 3u, 5u, 7u, 11u, 13u}) {
    std::vector<uint32_t> shape = {3, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    float scalarValue = 5.0f;

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = std::max(dataA[i], scalarValue);
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(BinaryVecScalarHandleTest, BinaryMin_Handle_NonAligned) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMin;

  // Test innermost dimensions that are not multiples of 4
  for (uint32_t innerDim : {1u, 3u, 5u, 7u, 11u, 13u}) {
    std::vector<uint32_t> shape = {4, innerDim};
    const uint32_t elements = totalElements(shape);
    const size_t bufferSize = elements * sizeof(float);

    SCOPED_TRACE("Shape: " + shapeToString(shape));

    auto dataA = generateTestData<float>(elements, 42);
    float scalarValue = 5.0f;

    auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
    auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

    auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

    std::vector<float> output(elements);
    runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

    for (uint32_t i = 0; i < elements; ++i) {
      float expected = std::min(dataA[i], scalarValue);
      ASSERT_NEAR(output[i], expected, 1e-5f)
          << "Mismatch at index " << i << " for shape " << shapeToString(shape);
    }
  }
}

TEST_F(BinaryVecScalarHandleTest, BinaryPow_Handle) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryPow;

  std::vector<uint32_t> shape = {2, 8};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  // Use positive values for pow operation
  for (auto &val : dataA) {
    val = std::abs(val);
  }
  float scalarValue = 2.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = std::pow(dataA[i], scalarValue);
    ASSERT_NEAR(output[i], expected, std::abs(expected) * 1e-4f + 1e-5f)
        << "Mismatch at index " << i;
  }
}

// Test multiple operations in a sequence
TEST_F(BinaryVecScalarHandleTest, BinaryVecScalar_Handle_MultipleOps) {
  const DataType dtype = DataType::Float32;

  std::vector<uint32_t> shape = {3, 12};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float scalar1 = 2.0f;
  float scalar2 = 3.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer1 = runtime_->createTensor({1}, dtype, &scalar1);
  auto scalarBuffer2 = runtime_->createTensor({1}, dtype, &scalar2);

  // Perform: (A * 2.0) + 3.0
  auto bufferTemp = runtime_->ops().binaryOp(BinaryMul, bufferA, scalarBuffer1);
  auto bufferOut =
      runtime_->ops().binaryOp(BinaryAdd, bufferTemp, scalarBuffer2);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = (dataA[i] * scalar1) + scalar2;
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

// Test comparison operations with handle
TEST_F(BinaryVecScalarHandleTest, BinaryLess_Handle) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryLess;

  std::vector<uint32_t> shape = {4, 8};
  const uint32_t elements = totalElements(shape);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 5.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  // Comparison ops produce UInt32 output
  std::vector<uint32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t expected = (dataA[i] < scalarValue) ? 1u : 0u;
    ASSERT_EQ(output[i], expected) << "Mismatch at index " << i;
  }
}

TEST_F(BinaryVecScalarHandleTest, BinaryGreater_Handle) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryGreater;

  std::vector<uint32_t> shape = {2, 3, 8};
  const uint32_t elements = totalElements(shape);

  auto dataA = generateTestData<float>(elements, 42);
  float scalarValue = 5.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  // Comparison ops produce UInt32 output
  std::vector<uint32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(),
                           elements * sizeof(uint32_t));

  for (uint32_t i = 0; i < elements; ++i) {
    uint32_t expected = (dataA[i] > scalarValue) ? 1u : 0u;
    ASSERT_EQ(output[i], expected) << "Mismatch at index " << i;
  }
}

// Test with activation operations
TEST_F(BinaryVecScalarHandleTest, BinaryLeakyRelu_Handle) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryLeakyRelu;

  std::vector<uint32_t> shape = {4, 12};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  auto dataA = generateTestData<float>(elements, 42);
  float alpha = 0.2f; // LeakyReLU alpha parameter

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto alphaBuffer = runtime_->createTensor({1}, dtype, &alpha);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, alphaBuffer);

  std::vector<float> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    float expected = dataA[i] > 0.0f ? dataA[i] : alpha * dataA[i];
    ASSERT_NEAR(output[i], expected, 1e-5f) << "Mismatch at index " << i;
  }
}

// Test with integer types
TEST_F(BinaryVecScalarHandleTest, BinaryAdd_Handle_Int32) {
  const DataType dtype = DataType::Int32;
  const OperatorEnum op = BinaryAdd;

  std::vector<uint32_t> shape = {4, 8};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(int32_t);

  std::vector<int32_t> dataA(elements);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(-100, 100);
  for (auto &val : dataA) {
    val = dist(gen);
  }
  int32_t scalarValue = 10;

  auto bufferA = runtime_->createTensor(shape, dtype, dataA.data());
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  std::vector<int32_t> output(elements);
  runtime_->copyFromTensor(bufferOut, output.data(), bufferSize);

  for (uint32_t i = 0; i < elements; ++i) {
    int32_t expected = dataA[i] + scalarValue;
    ASSERT_EQ(output[i], expected) << "Mismatch at index " << i;
  }
}

// Test edge case: very small tensor
TEST_F(BinaryVecScalarHandleTest, BinaryMul_Handle_SingleElement) {
  const DataType dtype = DataType::Float32;
  const OperatorEnum op = BinaryMul;

  std::vector<uint32_t> shape = {1};
  const uint32_t elements = totalElements(shape);
  const size_t bufferSize = elements * sizeof(float);

  float dataA = 4.0f;
  float scalarValue = 3.0f;

  auto bufferA = runtime_->createTensor(shape, dtype, &dataA);
  auto scalarBuffer = runtime_->createTensor({1}, dtype, &scalarValue);

  auto bufferOut = runtime_->ops().binaryOp(op, bufferA, scalarBuffer);

  float output;
  runtime_->copyFromTensor(bufferOut, &output, bufferSize);

  float expected = dataA * scalarValue;
  ASSERT_NEAR(output, expected, 1e-5f);
}

// ============================================================================
// Single-Pass Statistical Reduction Tests
// ============================================================================

// --- Variance ---

TEST_F(VulkanBackendTest, VarianceFused_Global_MatchesComposite) {
  auto data = generateTestData<float>(256, 42);
  auto buf = runtime_->createTensor({256}, DataType::Float32, data.data());

  auto varComposite = runtime_->ops().variance(buf, 0);
  auto varFused = runtime_->ops().varianceFused(buf, 0);

  float composite, fused;
  runtime_->copyFromTensor(varComposite, &composite, sizeof(float));
  runtime_->copyFromTensor(varFused, &fused, sizeof(float));

  ASSERT_NEAR(fused, composite, std::abs(composite) * 1e-4f + 1e-5f)
      << "Fused variance " << fused << " != composite " << composite;
}

TEST_F(VulkanBackendTest, VarianceFused_Global_BesselCorrection) {
  auto data = generateTestData<float>(100, 42);
  auto buf = runtime_->createTensor({100}, DataType::Float32, data.data());

  auto varComposite = runtime_->ops().variance(buf, 1);
  auto varFused = runtime_->ops().varianceFused(buf, 1);

  float composite, fused;
  runtime_->copyFromTensor(varComposite, &composite, sizeof(float));
  runtime_->copyFromTensor(varFused, &fused, sizeof(float));

  ASSERT_NEAR(fused, composite, std::abs(composite) * 1e-4f + 1e-5f)
      << "Fused variance (Bessel) " << fused << " != composite " << composite;
}

TEST_F(VulkanBackendTest, VarianceFused_Dim_MatchesComposite) {
  const uint32_t M = 16, N = 32;
  auto data = generateTestData<float>(M * N, 42);
  auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());

  auto varComposite = runtime_->ops().variance(buf, 0, 1);
  auto varFused = runtime_->ops().varianceFused(buf, 0, 1);

  std::vector<float> compositeOut(M), fusedOut(M);
  runtime_->copyFromTensor(varComposite, compositeOut.data(),
                           M * sizeof(float));
  runtime_->copyFromTensor(varFused, fusedOut.data(), M * sizeof(float));

  for (uint32_t i = 0; i < M; ++i) {
    ASSERT_NEAR(fusedOut[i], compositeOut[i],
                std::abs(compositeOut[i]) * 1e-4f + 1e-5f)
        << "Dim variance mismatch at index " << i;
  }
}

TEST_F(VulkanBackendTest, VarianceFused_Global_KnownValues) {
  // var([1,2,3,4,5]) = 2.0 (population variance)
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto buf = runtime_->createTensor({5}, DataType::Float32, data.data());
  auto result = runtime_->ops().varianceFused(buf, 0);

  float val;
  runtime_->copyFromTensor(result, &val, sizeof(float));
  ASSERT_NEAR(val, 2.0f, 1e-5f);
}

// --- RMS ---

TEST_F(VulkanBackendTest, RMS_Global_KnownValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "rms/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(VulkanBackendTest, RMS_Global_LargerArray) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "rms/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(VulkanBackendTest, RMS_Dim) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "rms/dim")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// --- LogSumExp ---

TEST_F(VulkanBackendTest, LogSumExp_Global_KnownValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsumexp/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(VulkanBackendTest, LogSumExp_Global_LargerArray) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsumexp/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(VulkanBackendTest, LogSumExp_Dim) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsumexp/dim")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// --- Timing comparison tests ---

TEST_F(VulkanBackendTest, SinglePassReductions_Timing) {
  constexpr int kWarmupIters = 3;
  constexpr int kTimingIters = 10;
  const uint32_t N = 65536;

  auto data = generateTestData<float>(N, 42);

  // Print header
  std::cout << "\n=== Single-Pass vs Composite Reduction Timing ===\n";
  std::cout << "Array size: " << N << " elements, " << kTimingIters
            << " iterations\n\n";

  // --- Variance ---
  {
    // Warmup
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().variance(buf, 0);
      runtime_->flush();
    }

    // Time composite variance
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().variance(buf, 0);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    // Warmup fused
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().varianceFused(buf, 0);
      runtime_->flush();
    }

    // Time single-pass variance
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().varianceFused(buf, 0);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "Variance:   composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  // --- RMS (compare against composite: sqrt(mean(x*x))) ---
  {
    // Warmup composite
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto sq = runtime_->ops().unaryOp(UnarySquare, buf);
      auto meanSq = runtime_->ops().reduce(ReduceMean, sq);
      auto rms = runtime_->ops().unaryOp(UnarySqrt, meanSq);
      runtime_->flush();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto sq = runtime_->ops().unaryOp(UnarySquare, buf);
      auto meanSq = runtime_->ops().reduce(ReduceMean, sq);
      auto rms = runtime_->ops().unaryOp(UnarySqrt, meanSq);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    // Warmup fused
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().rms(buf);
      runtime_->flush();
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().rms(buf);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "RMS:        composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  // --- LogSumExp (compare against composite from softmax pattern) ---
  {
    // Warmup composite
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto maxV = runtime_->ops().reduce(ReduceMax, buf);
      auto shifted = runtime_->ops().binaryOp(BinarySub, buf, maxV);
      auto exps = runtime_->ops().unaryOp(UnaryExp, shifted);
      auto sumE = runtime_->ops().reduce(ReduceSum, exps);
      auto logS = runtime_->ops().unaryOp(UnaryLog, sumE);
      auto lse = runtime_->ops().binaryOp(BinaryAdd, maxV, logS);
      runtime_->flush();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto maxV = runtime_->ops().reduce(ReduceMax, buf);
      auto shifted = runtime_->ops().binaryOp(BinarySub, buf, maxV);
      auto exps = runtime_->ops().unaryOp(UnaryExp, shifted);
      auto sumE = runtime_->ops().reduce(ReduceSum, exps);
      auto logS = runtime_->ops().unaryOp(UnaryLog, sumE);
      auto lse = runtime_->ops().binaryOp(BinaryAdd, maxV, logS);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    // Warmup fused
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSumExp(buf);
      runtime_->flush();
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSumExp(buf);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "LogSumExp:  composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  std::cout << std::endl;
}

// ===========================================================================
// Fused Softmax / LogSoftmax Tests
// ===========================================================================

TEST_F(RuntimeOperatorTest, SoftmaxFused_MatchesComposite_Dim1) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "softmax/composite_dim1")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, SoftmaxFused_MatchesComposite_Dim0) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "softmax/composite_dim0")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, SoftmaxFused_KnownValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "softmax/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, SoftmaxFused_LargerArray) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "softmax/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, SoftmaxFused_3D) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "softmax/3d")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, LogSoftmaxFused_MatchesComposite) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsoftmax/composite")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, LogSoftmaxFused_KnownValues) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsoftmax/known")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

TEST_F(RuntimeOperatorTest, LogSoftmaxFused_LargerArray) {
  for (const auto &c : opregistry::allOpCases()) {
    if (c.name != "logsoftmax/larger")
      continue;
    SCOPED_TRACE(c.name);
    Tensor out = c.run(*runtime_, -1);
    opregistry::VerifyResult vr = c.verify(*runtime_, out);
    EXPECT_TRUE(vr.ok) << c.name << ": " << vr.detail;
  }
}

// ===========================================================================
// Timing: Fused vs Composite Softmax + RMSNorm
// ===========================================================================

TEST_F(RuntimeOperatorTest, FusedSoftmax_Timing) {
  const uint32_t M = 128, N = 512;
  std::vector<float> data(M * N);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<float>(i % 100) * 0.01f;

  const int kWarmupIters = 5;
  const int kTimingIters = 20;

  std::cout << "\n=== Fused Softmax/LogSoftmax Timing ===\n";
  std::cout << "Shape: [" << M << ", " << N << "], reduce dim=1\n\n";

  // --- Softmax ---
  {
    // Warmup composite
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().softmax(buf, 1);
      runtime_->flush();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().softmax(buf, 1);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    // Warmup fused
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().softmaxFused(buf, 1);
      runtime_->flush();
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().softmaxFused(buf, 1);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "Softmax:     composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  // --- LogSoftmax ---
  {
    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSoftmax(buf, 1);
      runtime_->flush();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSoftmax(buf, 1);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    for (int i = 0; i < kWarmupIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSoftmaxFused(buf, 1);
      runtime_->flush();
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto buf = runtime_->createTensor({M, N}, DataType::Float32, data.data());
      auto r = runtime_->ops().logSoftmaxFused(buf, 1);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "LogSoftmax:  composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  // --- RMSNorm: fused (existing) vs composite ---
  {
    const uint32_t dim = 512;
    std::vector<float> xData(dim), wData(dim);
    for (uint32_t i = 0; i < dim; ++i) {
      xData[i] = static_cast<float>(i) * 0.01f - 2.5f;
      wData[i] = 1.0f;
    }

    // Warmup fused RMSNorm
    for (int i = 0; i < kWarmupIters; ++i) {
      auto x = runtime_->createTensor({dim}, DataType::Float32, xData.data());
      auto w = runtime_->createTensor({dim}, DataType::Float32, wData.data());
      auto r = runtime_->ops().rmsNorm(x, w, 1e-5f);
      runtime_->flush();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto x = runtime_->createTensor({dim}, DataType::Float32, xData.data());
      auto w = runtime_->createTensor({dim}, DataType::Float32, wData.data());
      auto r = runtime_->ops().rmsNorm(x, w, 1e-5f);
      runtime_->flush();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fusedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    // Composite RMSNorm: rms(x) → divide → multiply weight
    for (int i = 0; i < kWarmupIters; ++i) {
      auto x = runtime_->createTensor({dim}, DataType::Float32, xData.data());
      auto w = runtime_->createTensor({dim}, DataType::Float32, wData.data());
      auto rmsVal = runtime_->ops().rms(x);
      auto rmsScalar =
          runtime_->ops().binaryOp(OperatorEnum::BinaryAdd, rmsVal, 1e-5f);
      auto invRms =
          runtime_->ops().unaryOp(OperatorEnum::UnaryReciprocal, rmsScalar);
      auto normalized =
          runtime_->ops().binaryOp(OperatorEnum::BinaryMul, x, invRms);
      auto result =
          runtime_->ops().binaryOp(OperatorEnum::BinaryMul, normalized, w);
      runtime_->flush();
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimingIters; ++i) {
      auto x = runtime_->createTensor({dim}, DataType::Float32, xData.data());
      auto w = runtime_->createTensor({dim}, DataType::Float32, wData.data());
      auto rmsVal = runtime_->ops().rms(x);
      auto rmsScalar =
          runtime_->ops().binaryOp(OperatorEnum::BinaryAdd, rmsVal, 1e-5f);
      auto invRms =
          runtime_->ops().unaryOp(OperatorEnum::UnaryReciprocal, rmsScalar);
      auto normalized =
          runtime_->ops().binaryOp(OperatorEnum::BinaryMul, x, invRms);
      auto result =
          runtime_->ops().binaryOp(OperatorEnum::BinaryMul, normalized, w);
      runtime_->flush();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double compositeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        kTimingIters;

    std::cout << "RMSNorm:     composite=" << compositeMs
              << "ms  fused=" << fusedMs
              << "ms  speedup=" << compositeMs / fusedMs << "x\n";
  }

  std::cout << std::endl;
}

// ============================================================================
// Mapped Tensor Tests
// ============================================================================

class MappedTensorTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(MappedTensorTest, CreateAndRead) {
  // Create mapped tensor with initial data
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto buf = runtime_->createTensorMapped({4}, DataType::Float32, data.data());

  // Read back — should match
  std::vector<float> output(4);
  runtime_->copyFromTensor(buf, output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], data[i]) << "Mismatch at " << i;
  }
}

TEST_F(MappedTensorTest, UpdateViaCopy) {
  // Create mapped tensor
  std::vector<uint32_t> init = {0, 0};
  auto buf = runtime_->createTensorMapped({2}, DataType::UInt32, init.data());

  // Update via copyToTensor (should be direct memcpy, no staging)
  uint32_t newVals[2] = {42, 99};
  runtime_->copyToTensor(buf, newVals, sizeof(newVals));

  // Read back
  uint32_t output[2] = {0, 0};
  runtime_->copyFromTensor(buf, output, sizeof(output));
  EXPECT_EQ(output[0], 42u);
  EXPECT_EQ(output[1], 99u);
}

TEST_F(MappedTensorTest, GpuCanRead) {
  // Create a mapped tensor and verify the GPU can read from it
  // by using it in a binary op
  std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dataB = {10.0f, 20.0f, 30.0f, 40.0f};

  auto bufA =
      runtime_->createTensorMapped({4}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({4}, DataType::Float32, dataB.data());

  auto result = runtime_->ops().binaryOp(OperatorEnum::BinaryAdd, bufA, bufB);
  runtime_->flush();

  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], dataA[i] + dataB[i], 1e-5f);
  }
}

TEST_F(MappedTensorTest, UpdateThenGpuRead) {
  // Create mapped tensor, update it via CPU memcpy, then have GPU read it
  std::vector<float> init = {0.0f, 0.0f, 0.0f, 0.0f};
  auto mapped =
      runtime_->createTensorMapped({4}, DataType::Float32, init.data());
  auto ones = runtime_->createTensor({4}, DataType::Float32);
  {
    std::vector<float> onesData = {1.0f, 1.0f, 1.0f, 1.0f};
    runtime_->copyToTensor(ones, onesData.data(), 4 * sizeof(float));
    runtime_->flush();
  }

  // CPU-side update of mapped buffer
  std::vector<float> newData = {5.0f, 6.0f, 7.0f, 8.0f};
  runtime_->copyToTensor(mapped, newData.data(), 4 * sizeof(float));

  // GPU add: mapped + ones
  auto result = runtime_->ops().binaryOp(OperatorEnum::BinaryAdd, mapped, ones);
  runtime_->flush();

  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], newData[i] + 1.0f, 1e-5f);
  }
}

// ============================================================================
// CacheWrite Tests
// ============================================================================

class CacheWriteTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(CacheWriteTest, WriteSingleRow) {
  const uint32_t maxSeq = 8, dim = 4;

  // Cache: [maxSeq, dim] initialized to zero
  std::vector<float> cacheData(maxSeq * dim, 0.0f);
  auto cache = runtime_->createTensor({maxSeq, dim}, DataType::Float32,
                                      cacheData.data());

  // New data to write at position 3
  std::vector<float> newRow = {1.0f, 2.0f, 3.0f, 4.0f};
  auto newData =
      runtime_->createTensor({dim}, DataType::Float32, newRow.data());

  // Runtime params: {pos=3, seqLen=4}
  uint32_t params[2] = {3, 4};
  auto runtimeParams = runtime_->createTensor({2}, DataType::UInt32, params);

  runtime_->ops().cacheWrite(cache, newData, runtimeParams);
  runtime_->flush();

  // Read back the cache
  std::vector<float> output(maxSeq * dim, -1.0f);
  runtime_->copyFromTensor(cache, output.data(), maxSeq * dim * sizeof(float));

  // Row 3 should be our data, others should be 0
  for (uint32_t row = 0; row < maxSeq; ++row) {
    for (uint32_t col = 0; col < dim; ++col) {
      float expected = (row == 3) ? newRow[col] : 0.0f;
      EXPECT_NEAR(output[row * dim + col], expected, 1e-5f)
          << "Mismatch at [" << row << ", " << col << "]";
    }
  }
}

TEST_F(CacheWriteTest, WriteMultipleRows) {
  const uint32_t maxSeq = 8, dim = 4;

  std::vector<float> cacheData(maxSeq * dim, 0.0f);
  auto cache = runtime_->createTensor({maxSeq, dim}, DataType::Float32,
                                      cacheData.data());

  // Write rows 0, 1, 2
  for (uint32_t pos = 0; pos < 3; ++pos) {
    std::vector<float> row(dim);
    for (uint32_t d = 0; d < dim; ++d)
      row[d] = static_cast<float>(pos * dim + d + 1);
    auto newData = runtime_->createTensor({dim}, DataType::Float32, row.data());

    uint32_t params[2] = {pos, pos + 1};
    auto runtimeParams = runtime_->createTensor({2}, DataType::UInt32, params);

    runtime_->ops().cacheWrite(cache, newData, runtimeParams);
    runtime_->flush();
  }

  std::vector<float> output(maxSeq * dim, -1.0f);
  runtime_->copyFromTensor(cache, output.data(), maxSeq * dim * sizeof(float));

  // Rows 0-2 should have our data
  for (uint32_t pos = 0; pos < 3; ++pos) {
    for (uint32_t d = 0; d < dim; ++d) {
      float expected = static_cast<float>(pos * dim + d + 1);
      EXPECT_NEAR(output[pos * dim + d], expected, 1e-5f)
          << "Mismatch at [" << pos << ", " << d << "]";
    }
  }
  // Row 3+ should still be zero
  for (uint32_t pos = 3; pos < maxSeq; ++pos) {
    for (uint32_t d = 0; d < dim; ++d) {
      EXPECT_FLOAT_EQ(output[pos * dim + d], 0.0f);
    }
  }
}

// ============================================================================
// Attention Tests
// ============================================================================

class AttentionTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }

  // CPU reference: single-head scaled dot-product attention
  // q: [dim], kCache: [seqLen, dim], vCache: [seqLen, dim]
  std::vector<float> cpuAttention(const std::vector<float> &q,
                                  const std::vector<float> &kCache,
                                  const std::vector<float> &vCache,
                                  uint32_t seqLen,
                                  uint32_t headDim,
                                  uint32_t nHeads,
                                  uint32_t nKvHeads) {
    uint32_t dim = nHeads * headDim;
    uint32_t kvDim = nKvHeads * headDim;
    uint32_t nRep = nHeads / nKvHeads;
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    std::vector<float> output(dim, 0.0f);

    for (uint32_t h = 0; h < nHeads; ++h) {
      uint32_t kvH = h / nRep; // GQA mapping

      // Compute scores: q[h] dot k[t] for each cached position t
      std::vector<float> scores(seqLen);
      for (uint32_t t = 0; t < seqLen; ++t) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < headDim; ++d) {
          dot += q[h * headDim + d] * kCache[t * kvDim + kvH * headDim + d];
        }
        scores[t] = dot * scale;
      }

      // Softmax
      float maxScore = *std::max_element(scores.begin(), scores.end());
      float sumExp = 0.0f;
      for (auto &s : scores) {
        s = std::exp(s - maxScore);
        sumExp += s;
      }
      for (auto &s : scores)
        s /= sumExp;

      // Weighted sum of values
      for (uint32_t d = 0; d < headDim; ++d) {
        float val = 0.0f;
        for (uint32_t t = 0; t < seqLen; ++t) {
          val += scores[t] * vCache[t * kvDim + kvH * headDim + d];
        }
        output[h * headDim + d] = val;
      }
    }
    return output;
  }
};

TEST_F(AttentionTest, SingleHeadBasic) {
  const uint32_t nHeads = 1, nKvHeads = 1, headDim = 4;
  const uint32_t dim = nHeads * headDim;
  const uint32_t seqLen = 3;
  const uint32_t maxSeq = 8;

  // Q: single query vector
  auto qData = generateTestData<float>(dim, 42);
  auto q = runtime_->createTensor({dim}, DataType::Float32, qData.data());

  // KV caches: [maxSeq, dim] — fill first seqLen rows
  auto kData = generateTestData<float>(maxSeq * dim, 100);
  auto vData = generateTestData<float>(maxSeq * dim, 200);
  auto kCache =
      runtime_->createTensor({maxSeq, dim}, DataType::Float32, kData.data());
  auto vCache =
      runtime_->createTensor({maxSeq, dim}, DataType::Float32, vData.data());

  // runtimeParams: {pos=seqLen-1, seqLen}
  uint32_t params[2] = {seqLen - 1, seqLen};
  auto runtimeParams = runtime_->createTensor({2}, DataType::UInt32, params);

  auto result = runtime_->ops().attention(q, kCache, vCache, runtimeParams,
                                          nHeads, nKvHeads, headDim);
  runtime_->flush();

  std::vector<float> output(dim);
  runtime_->copyFromTensor(result, output.data(), dim * sizeof(float));

  auto expected =
      cpuAttention(qData, kData, vData, seqLen, headDim, nHeads, nKvHeads);

  for (uint32_t i = 0; i < dim; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-3f) << "Mismatch at index " << i;
  }
}

TEST_F(AttentionTest, MultiHeadGQA) {
  // 4 query heads, 2 KV heads (GQA with repeat=2)
  const uint32_t nHeads = 4, nKvHeads = 2, headDim = 8;
  const uint32_t dim = nHeads * headDim;     // 32
  const uint32_t kvDim = nKvHeads * headDim; // 16
  const uint32_t seqLen = 4;
  const uint32_t maxSeq = 16;

  auto qData = generateTestData<float>(dim, 42);
  auto q = runtime_->createTensor({dim}, DataType::Float32, qData.data());

  auto kData = generateTestData<float>(maxSeq * kvDim, 100);
  auto vData = generateTestData<float>(maxSeq * kvDim, 200);
  auto kCache =
      runtime_->createTensor({maxSeq, kvDim}, DataType::Float32, kData.data());
  auto vCache =
      runtime_->createTensor({maxSeq, kvDim}, DataType::Float32, vData.data());

  uint32_t params[2] = {seqLen - 1, seqLen};
  auto runtimeParams = runtime_->createTensor({2}, DataType::UInt32, params);

  auto result = runtime_->ops().attention(q, kCache, vCache, runtimeParams,
                                          nHeads, nKvHeads, headDim);
  runtime_->flush();

  std::vector<float> output(dim);
  runtime_->copyFromTensor(result, output.data(), dim * sizeof(float));

  auto expected =
      cpuAttention(qData, kData, vData, seqLen, headDim, nHeads, nKvHeads);

  for (uint32_t i = 0; i < dim; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-2f)
        << "Mismatch at head " << (i / headDim) << " dim " << (i % headDim);
  }
}

// ============================================================================
// GEMV-specific Tests (K-parallel subgroup reduction)
// ============================================================================

class GemvTest : public RuntimeOperatorTest {
protected:
  void SetUp() override {
    RuntimeOperatorTest::SetUp();
    initBackend(BackendType::Vulkan);
  }
};

TEST_F(GemvTest, SmallN) {
  // N < 4 (tail handling in COLS_PER_WG=4)
  const uint32_t M = 1, K = 32, N = 3;
  auto dataA = generateTestData<float>(M * K, 42);
  auto dataB = generateTestData<float>(K * N, 99);

  std::vector<float> expected(M * N, 0.0f);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] += dataA[k] * dataB[k * N + j];

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Float32, dataB.data());
  auto bufC = runtime_->ops().matmul(bufA, bufB);
  runtime_->flush();

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  for (uint32_t j = 0; j < N; ++j) {
    EXPECT_NEAR(output[j], expected[j], K * 1e-5f)
        << "Mismatch at column " << j;
  }
}

TEST_F(GemvTest, LargeK) {
  // Large K to exercise multiple iterations per thread (K/32 > 1)
  const uint32_t M = 1, K = 1536, N = 576;
  auto dataA = generateTestData<float>(M * K, 1);
  auto dataB = generateTestData<float>(K * N, 2);

  std::vector<float> expected(M * N, 0.0f);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] += dataA[k] * dataB[k * N + j];

  auto bufA = runtime_->createTensor({M, K}, DataType::Float32, dataA.data());
  auto bufB = runtime_->createTensor({K, N}, DataType::Float32, dataB.data());
  auto bufC = runtime_->ops().matmul(bufA, bufB);
  runtime_->flush();

  std::vector<float> output(M * N);
  runtime_->copyFromTensor(bufC, output.data(), M * N * sizeof(float));

  // K-parallel reduction accumulates in different order than sequential,
  // so use relative tolerance based on expected magnitude.
  for (uint32_t j = 0; j < N; ++j) {
    float tol = std::max(std::abs(expected[j]) * 1e-5f, 1e-3f);
    EXPECT_NEAR(output[j], expected[j], tol) << "Mismatch at column " << j;
  }
}

TEST_F(GemvTest, TransformerDimensions) {
  // Test dimensions matching SmolLM2-135M: dim=576, ffn=1536
  struct TestCase {
    uint32_t K, N;
  };
  std::array<TestCase, 4> cases = {
      {{576, 960}, {576, 576}, {576, 1536}, {1536, 576}}};

  for (const auto &tc : cases) {
    SCOPED_TRACE("K=" + std::to_string(tc.K) + " N=" + std::to_string(tc.N));

    auto dataA = generateTestData<float>(tc.K, 42);
    auto dataB = generateTestData<float>(tc.K * tc.N, 123);

    std::vector<float> expected(tc.N, 0.0f);
    for (uint32_t k = 0; k < tc.K; ++k)
      for (uint32_t j = 0; j < tc.N; ++j)
        expected[j] += dataA[k] * dataB[k * tc.N + j];

    auto bufA =
        runtime_->createTensor({1, tc.K}, DataType::Float32, dataA.data());
    auto bufB =
        runtime_->createTensor({tc.K, tc.N}, DataType::Float32, dataB.data());
    auto bufC = runtime_->ops().matmul(bufA, bufB);
    runtime_->flush();

    std::vector<float> output(tc.N);
    runtime_->copyFromTensor(bufC, output.data(), tc.N * sizeof(float));

    for (uint32_t j = 0; j < tc.N; ++j) {
      float tol = std::max(std::abs(expected[j]) * 1e-5f, 1e-3f);
      ASSERT_NEAR(output[j], expected[j], tol) << "Mismatch at column " << j;
    }
  }
}

TEST_F(GemvTest, MixedPrecisionF32xF16) {
  // F32 activation × F16 weights → F32 output (the common LLM inference path)
  const uint32_t M = 1, K = 128, N = 64;

  std::vector<float> dataA_f32(K), dataB_f32(K * N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : dataA_f32)
    v = dist(gen);
  for (auto &v : dataB_f32)
    v = dist(gen);

  std::vector<float> expected(N, 0.0f);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j)
      expected[j] += dataA_f32[k] * dataB_f32[k * N + j];

  // A stays F32, B is cast to F16 (mimics weight storage)
  auto bufA =
      runtime_->createTensor({M, K}, DataType::Float32, dataA_f32.data());
  auto bufB32 =
      runtime_->createTensor({K, N}, DataType::Float32, dataB_f32.data());
  auto bufB = runtime_->ops().cast(bufB32, DataType::Float16);
  runtime_->flush();
  auto bufC = runtime_->ops().matmul(bufA, bufB);
  runtime_->flush();

  std::vector<float> output(N);
  runtime_->copyFromTensor(bufC, output.data(), N * sizeof(float));

  // F32 accumulation with F16 weights — moderate precision loss
  for (uint32_t j = 0; j < N; ++j) {
    float tol = std::max(std::abs(expected[j]) * 5e-3f, 1e-2f);
    EXPECT_NEAR(output[j], expected[j], tol) << "Mismatch at column " << j;
  }
}

// =========================================================================
// Multi-device Runtime
// =========================================================================

/// Device index for multi-device tests, overridable via env var so the
/// same tests can exercise two distinct physical devices (default: -1,
/// i.e. two contexts on the backend's default device).
static int testDeviceIndex(const char *envVar) {
  if (const char *v = std::getenv(envVar)) {
    return std::atoi(v);
  }
  return -1;
}

class MultiDeviceTest : public ::testing::Test {};

TEST_F(MultiDeviceTest, TwoContextsOnDefaultDevice) {
  Runtime runtime;
  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }
  const int devA = testDeviceIndex("CUT_TEST_DEVICE_A");
  const int devB = testDeviceIndex("CUT_TEST_DEVICE_B");
  runtime.init({{BackendType::Vulkan, devA}, {BackendType::Vulkan, devB}});
  ASSERT_EQ(runtime.deviceCount(), 2u);

  for (size_t d = 0; d < 2; ++d) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto in =
        runtime.createTensor({4}, DataType::Float32, data.data(), false, d);
    auto out = runtime.ops(d).binaryOp(BinaryAdd, in, in);
    runtime.flush(d);

    std::vector<float> host(4);
    runtime.copyFromTensor(out, host.data(), 4 * sizeof(float), 0, 0, d);
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(host[i], data[i] * 2.0f) << "device " << d;
    }
  }
}

TEST_F(MultiDeviceTest, TransferTensorAcrossDevices) {
  Runtime runtime;
  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }
  const int devA = testDeviceIndex("CUT_TEST_DEVICE_A");
  const int devB = testDeviceIndex("CUT_TEST_DEVICE_B");
  runtime.init({{BackendType::Vulkan, devA}, {BackendType::Vulkan, devB}});

  std::vector<float> data = {1.5f, -2.0f, 3.25f, 0.0f};
  auto a =
      runtime.createTensor({2, 2}, DataType::Float32, data.data(), false, 0);

  // Transfer to a freshly created tensor on device 1.
  auto b = runtime.transferTensor(a, 0, 1);
  std::vector<float> host1(4);
  runtime.copyFromTensor(b, host1.data(), 4 * sizeof(float), 0, 0, 1);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(host1[i], data[i]);
  }

  // Transfer into a pre-created tensor on device 1.
  auto c = runtime.createTensorEmpty({2, 2}, DataType::Float32, false, 1);
  runtime.transferTensor(a, 0, c, 1);
  std::vector<float> host2(4);
  runtime.copyFromTensor(c, host2.data(), 4 * sizeof(float), 0, 0, 1);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(host2[i], data[i]);
  }
}

TEST_F(MultiDeviceTest, TransferShapeMismatchThrows) {
  Runtime runtime;
  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }
  runtime.init({{BackendType::Vulkan, -1}, {BackendType::Vulkan, -1}});

  auto a = runtime.createTensorEmpty({4}, DataType::Float32, false, 0);
  auto b = runtime.createTensorEmpty({8}, DataType::Float32, false, 1);
  EXPECT_THROW(runtime.transferTensor(a, 0, b, 1), std::runtime_error);
}

TEST_F(MultiDeviceTest, SingleDeviceCompatibility) {
  Runtime runtime;
  if (!runtime.isVulkanAvailable()) {
    GTEST_SKIP() << "Vulkan not available";
  }
  runtime.init(BackendType::Vulkan);
  ASSERT_EQ(runtime.deviceCount(), 1u);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto in = runtime.createTensor({4}, DataType::Float32, data.data());
  auto out = runtime.ops().binaryOp(BinaryAdd, in, in);
  runtime.flush();

  std::vector<float> host(4);
  runtime.copyFromTensor(out, host.data(), 4 * sizeof(float));
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(host[i], data[i] * 2.0f);
  }
}

} // namespace
} // namespace cut
