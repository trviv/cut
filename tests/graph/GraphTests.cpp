#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <OpNode.h>
#include <Operations.h>
#include <Runtime.h>
#include <SharedRuntime.h>
#include <graph/Graph.h>
#include <graph/GraphBuilder.h>
#include <graph/GraphExecutor.h>
#include <graph/GraphOptimizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cut {
namespace graph {
namespace {

// ============================================================================
// Test Fixture — initializes Vulkan Runtime
// ============================================================================

class GraphTest : public ::testing::Test {
protected:
  void SetUp() override {
    runtime_ = test::sharedRuntime();
    if (!runtime_) {
      GTEST_SKIP() << "Vulkan not available";
    }
  }

  void TearDown() override { runtime_->flush(); }

  Runtime *runtime_ = nullptr;
};

// ============================================================================
// Graph Construction Tests
// ============================================================================

TEST_F(GraphTest, EmptyGraph) {
  Graph graph;
  EXPECT_EQ(graph.size(), 0u);
  EXPECT_TRUE(graph.outputs().empty());
  auto order = graph.topologicalOrder();
  EXPECT_TRUE(order.empty());
}

TEST_F(GraphTest, SingleInputNode) {
  auto tensor = runtime_->createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto x = builder.input(tensor);
  builder.markOutput(x);
  auto graph = builder.build();

  EXPECT_EQ(graph->size(), 1u);
  EXPECT_EQ(graph->outputs().size(), 1u);
  EXPECT_TRUE(graph->node(x).isInput);
  EXPECT_EQ(graph->node(x).op->outputShape(), (std::vector<uint32_t>{4, 8}));
  EXPECT_EQ(graph->node(x).op->outputDtype(), DataType::Float32);
}

TEST_F(GraphTest, LinearChain) {
  auto a = runtime_->createTensor({8}, DataType::Float32);
  auto b = runtime_->createTensor({8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().unaryOp(UnarySqrt, sum);
  builder.markOutput(result);
  auto graph = builder.build();

  EXPECT_EQ(graph->size(), 4u);
  EXPECT_EQ(graph->node(result).op->outputShape(), (std::vector<uint32_t>{8}));

  auto order = graph->topologicalOrder();
  EXPECT_EQ(order.size(), 4u);
  // inputs come first, then sum, then result
  uint32_t sumId = graph->nodeId(sum);
  uint32_t resultId = graph->nodeId(result);
  EXPECT_TRUE(std::find(order.begin(), order.end(), sumId) != order.end());
  auto sumPos = std::find(order.begin(), order.end(), sumId);
  auto resultPos = std::find(order.begin(), order.end(), resultId);
  EXPECT_TRUE(sumPos < resultPos);
}

TEST_F(GraphTest, DiamondDAG) {
  // Test fan-out + fan-in: x → a, x → b, (a, b) → c
  auto t = runtime_->createTensor({16}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto x = builder.input(t);
  auto a = builder.ops().unaryOp(UnarySin, x);
  auto b = builder.ops().unaryOp(UnaryCos, x);
  auto c = builder.ops().binaryOp(BinaryAdd, a, b);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph->size(), 4u);
  // x has refCount 2 (used by both a and b)
  graph->recomputeRefCounts();
  EXPECT_EQ(graph->node(x).refCount, 2u);
}

TEST_F(GraphTest, MatMulShapeInference) {
  auto a = runtime_->createTensor({3, 4}, DataType::Float32);
  auto b = runtime_->createTensor({4, 5}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto c = builder.ops().matmul(va, vb);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(c).op->outputShape(), (std::vector<uint32_t>{3, 5}));
  EXPECT_EQ(graph->node(c).op->outputDtype(), DataType::Float32);
}

TEST_F(GraphTest, TransposeShapeInference) {
  auto a = runtime_->createTensor({3, 7}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto t = builder.ops().transpose(va);
  builder.markOutput(t);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(t).op->outputShape(), (std::vector<uint32_t>{7, 3}));
}

TEST_F(GraphTest, ReshapeShapeInference) {
  auto a = runtime_->createTensor({12}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {3, 4});
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(r).op->outputShape(), (std::vector<uint32_t>{3, 4}));
}

TEST_F(GraphTest, ReshapeWithNegativeOne) {
  auto a = runtime_->createTensor({12}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {3, -1});
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(r).op->outputShape(), (std::vector<uint32_t>{3, 4}));
}

TEST_F(GraphTest, ReduceGlobalShapeInference) {
  auto a = runtime_->createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reduce(ReduceSum, va);
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(r).op->outputShape(), (std::vector<uint32_t>{1}));
}

TEST_F(GraphTest, ReduceDimShapeInference) {
  auto a = runtime_->createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reduce(ReduceSum, va, 0);
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph->node(r).op->outputShape(), (std::vector<uint32_t>{8}));
}

// ============================================================================
// Optimization Pass Tests
// ============================================================================

TEST_F(GraphTest, IdentityReshapeElimination) {
  // reshape(x, same_shape) → x
  auto a = runtime_->createTensor({8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {8}); // Same shape → identity
  builder.markOutput(r);
  auto graph = builder.build();

  uint32_t vaId = graph->nodeId(va);

  EXPECT_EQ(graph->size(), 2u);

  IdentityReshapePass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // Output should now point to the input node
  EXPECT_EQ(graph->outputs()[0], vaId);
}

TEST_F(GraphTest, ReshapeChainElimination) {
  // reshape(reshape(x, {1, 8}), {8}) → reshape(x, {8})
  auto a = runtime_->createTensor({8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r1 = builder.ops().reshape(va, {1, 8}); // 8 → 1x8
  auto r2 = builder.ops().reshape(r1, {8});    // 1x8 → 8
  builder.markOutput(r2);
  auto graph = builder.build();

  uint32_t vaId = graph->nodeId(va);
  uint32_t r2Id = graph->nodeId(r2);

  EXPECT_EQ(graph->size(), 3u);

  ReshapeChainPass chainPass;
  chainPass.run(*graph, runtime_->store());

  // r2 should now point directly to va's output (skip r1)
  EXPECT_EQ(graph->node(r2Id).inputIds[0], vaId);
}

TEST_F(GraphTest, CrossDimensionalityReshapeElimination) {
  // [576] → [1, 576]: dimensionality changes but inner dim is the same,
  // so memory layout is identical — should be eliminated by NoOpReshapePass.
  auto a = runtime_->createTensor({576}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {1, 576}); // 1D → 2D, same inner
  builder.markOutput(r);
  auto graph = builder.build();

  uint32_t vaId = graph->nodeId(va);

  NoOpReshapePass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // Output should now point to the input node (reshape eliminated)
  EXPECT_EQ(graph->outputs()[0], vaId);
}

TEST_F(GraphTest, CrossDimensionalityReshapeElimination_2Dto1D) {
  // [1, 576] → [576]: 2D→1D, same inner dim — should be eliminated.
  auto a = runtime_->createTensor({1, 576}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {576});
  builder.markOutput(r);
  auto graph = builder.build();

  uint32_t vaId = graph->nodeId(va);

  NoOpReshapePass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  EXPECT_EQ(graph->outputs()[0], vaId);
}

TEST_F(GraphTest, TransposeCancelElimination) {
  auto a = runtime_->createTensor({3, 5}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto t1 = builder.ops().transpose(va);
  auto t2 = builder.ops().transpose(t1);
  builder.markOutput(t2);
  auto graph = builder.build();

  uint32_t vaId = graph->nodeId(va);

  TransposeCancelPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // Output should now point to the original input
  EXPECT_EQ(graph->outputs()[0], vaId);
}

TEST_F(GraphTest, DeadCodeElimination) {
  auto a = runtime_->createTensor({8}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto dead = builder.ops().unaryOp(UnarySin, va); // Not used by output
  auto live = builder.ops().unaryOp(UnaryCos, va);
  builder.markOutput(live);
  auto graph = builder.build();

  EXPECT_EQ(graph->size(), 3u);

  DeadCodePass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // The dead node should be marked as removed
  EXPECT_TRUE(graph->node(dead).isRemoved);
  EXPECT_FALSE(graph->node(live).isRemoved);
}

TEST_F(GraphTest, FullOptimizationPipeline) {
  // Simulates FFN-like pattern: reshape → matmul → reshape back
  // The outer reshapes may be eliminable
  auto x = runtime_->createTensor({8}, DataType::Float32);
  auto w = runtime_->createTensor({8, 4}, DataType::Float32);

  GraphBuilder builder(*runtime_);
  auto vx = builder.input(x);
  auto vw = builder.input(w, true);

  auto reshaped = builder.ops().reshape(vx, {1, 8});
  auto mm = builder.ops().matmul(reshaped, vw);          // [1, 4]
  auto activated = builder.ops().unaryOp(UnarySilu, mm); // [1, 4]
  auto out = builder.ops().reshape(activated, {4});      // [4]
  builder.markOutput(out);

  auto graph = builder.build();
  size_t origSize = graph->size();

  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());

  // The topological order should only contain non-removed nodes
  auto order = graph->topologicalOrder();
  EXPECT_LE(order.size(), origSize);
}

// ============================================================================
// Executor Round-Trip Tests
// ============================================================================

TEST_F(GraphTest, ExecutorBinaryOp) {
  // Create data
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Build graph
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  builder.markOutput(sum);
  auto graph = builder.build();

  // Execute
  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  // Read back and verify
  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], aData[i] + bData[i]);
  }
}

TEST_F(GraphTest, ExecutorUnaryOp) {
  std::vector<float> aData = {1.0f, 4.0f, 9.0f, 16.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto sq = builder.ops().unaryOp(UnarySqrt, va);
  builder.markOutput(sq);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(aData[i]), 1e-5f);
  }
}

TEST_F(GraphTest, ExecutorMatMul) {
  // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
  // Result = [[19, 22], [43, 50]]
  std::vector<float> aData = {1, 2, 3, 4};
  std::vector<float> bData = {5, 6, 7, 8};
  auto a = runtime_->createTensor({2, 2}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({2, 2}, DataType::Float32, bData.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto mm = builder.ops().matmul(va, vb);
  builder.markOutput(mm);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  EXPECT_NEAR(output[0], 19.0f, 1e-4f);
  EXPECT_NEAR(output[1], 22.0f, 1e-4f);
  EXPECT_NEAR(output[2], 43.0f, 1e-4f);
  EXPECT_NEAR(output[3], 50.0f, 1e-4f);
}

TEST_F(GraphTest, ExecutorReshape) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_->createTensor({6}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {2, 3});
  builder.markOutput(r);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  // Verify shape via reading back all elements
  std::vector<float> output(6);
  runtime_->copyFromTensor(results[0], output.data(), 6 * sizeof(float));

  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(output[i], data[i]);
  }
}

TEST_F(GraphTest, ExecutorVecScalarOp) {
  std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto scaled = builder.ops().binaryOp(BinaryMul, va, 0.5f);
  builder.markOutput(scaled);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], data[i] * 0.5f);
  }
}

TEST_F(GraphTest, ExecutorTranspose) {
  // 2x3 matrix → 3x2
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto t = builder.ops().transpose(va);
  builder.markOutput(t);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  std::vector<float> output(6);
  runtime_->copyFromTensor(results[0], output.data(), 6 * sizeof(float));

  // Expected: [[1, 4], [2, 5], [3, 6]]
  EXPECT_NEAR(output[0], 1.0f, 1e-5f);
  EXPECT_NEAR(output[1], 4.0f, 1e-5f);
  EXPECT_NEAR(output[2], 2.0f, 1e-5f);
  EXPECT_NEAR(output[3], 5.0f, 1e-5f);
  EXPECT_NEAR(output[4], 3.0f, 1e-5f);
  EXPECT_NEAR(output[5], 6.0f, 1e-5f);
}

TEST_F(GraphTest, ExecutorReduce) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto sum = builder.ops().reduce(ReduceSum, va);
  builder.markOutput(sum);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);

  float output = 0.0f;
  runtime_->copyFromTensor(results[0], &output, sizeof(float));

  EXPECT_NEAR(output, 10.0f, 1e-5f);
}

TEST_F(GraphTest, OptimizedExecutionMatchesEager) {
  // Build an FFN-like graph, optimize, execute, and compare against eager
  std::vector<float> xData(8);
  std::vector<float> wData(8 * 4);
  for (int i = 0; i < 8; ++i)
    xData[i] = static_cast<float>(i + 1) * 0.1f;
  for (int i = 0; i < 32; ++i)
    wData[i] = static_cast<float>(i + 1) * 0.01f;

  auto x = runtime_->createTensor({8}, DataType::Float32, xData.data());
  auto w = runtime_->createTensor({8, 4}, DataType::Float32, wData.data());

  // Eager execution
  auto &ops = runtime_->ops();
  auto eager_x2d = ops.reshape(x, {1, 8});
  auto eager_mm = ops.matmul(eager_x2d, w);
  auto eager_act = ops.unaryOp(UnarySilu, eager_mm);
  auto eager_out = ops.reshape(eager_act, {4});

  std::vector<float> eagerResult(4);
  runtime_->copyFromTensor(eager_out, eagerResult.data(), 4 * sizeof(float));

  // Graph execution (with optimization)
  GraphBuilder builder(*runtime_);
  auto vx = builder.input(x);
  auto vw = builder.input(w, true);
  auto vx2d = builder.ops().reshape(vx, {1, 8});
  auto vmm = builder.ops().matmul(vx2d, vw);
  auto vact = builder.ops().unaryOp(UnarySilu, vmm);
  auto vout = builder.ops().reshape(vact, {4});
  builder.markOutput(vout);

  auto graph = builder.build();
  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);

  std::vector<float> graphResult(4);
  runtime_->copyFromTensor(results[0], graphResult.data(), 4 * sizeof(float));

  // Should match exactly
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphResult[i], eagerResult[i], 1e-5f);
  }
}

TEST_F(GraphTest, MultiOutputGraph) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto s = builder.ops().unaryOp(UnarySin, va);
  auto c = builder.ops().unaryOp(UnaryCos, va);
  builder.markOutput(s);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph->outputs().size(), 2u);

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 2u);

  std::vector<float> sinOut(4), cosOut(4);
  runtime_->copyFromTensor(results[0], sinOut.data(), 4 * sizeof(float));
  runtime_->copyFromTensor(results[1], cosOut.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(sinOut[i], std::sin(data[i]), 1e-5f);
    EXPECT_NEAR(cosOut[i], std::cos(data[i]), 1e-5f);
  }
}

// ============================================================================
// Auto-flush Tests (operations record to graph, copyFromTensor triggers flush)
// ============================================================================

TEST_F(GraphTest, AutoFlushBinaryOp) {
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto result = runtime_->ops().binaryOp(BinaryAdd, a, b);
  // copyFromTensor triggers flush automatically
  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], aData[i] + bData[i]);
  }
}

TEST_F(GraphTest, AutoFlushUnaryOp) {
  std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  auto result = runtime_->ops().unaryOp(UnarySqrt, a);
  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(data[i]), 1e-5f);
  }
}

TEST_F(GraphTest, AutoFlushMatMul) {
  std::vector<float> aData = {1, 2, 3, 4};
  std::vector<float> bData = {5, 6, 7, 8};
  auto a = runtime_->createTensor({2, 2}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({2, 2}, DataType::Float32, bData.data());

  auto result = runtime_->ops().matmul(a, b);
  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));

  EXPECT_NEAR(output[0], 19.0f, 1e-4f);
  EXPECT_NEAR(output[1], 22.0f, 1e-4f);
  EXPECT_NEAR(output[2], 43.0f, 1e-4f);
  EXPECT_NEAR(output[3], 50.0f, 1e-4f);
}

TEST_F(GraphTest, AutoFlushChain) {
  // Test multi-op chain: (a + b) * 2.0
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto sum = ops.binaryOp(BinaryAdd, a, b);
  float two = 2.0f;
  auto scaled =
      ops.binaryOp(BinaryMul, sum, DataReference(&two, sizeof(float)));
  std::vector<float> output(4);
  runtime_->copyFromTensor(scaled, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], (aData[i] + bData[i]) * 2.0f);
  }
}

TEST_F(GraphTest, AutoFlushFFN) {
  // FFN-like: reshape → matmul → SiLU → reshape
  std::vector<float> xData(8);
  std::vector<float> wData(8 * 4);
  for (int i = 0; i < 8; ++i)
    xData[i] = static_cast<float>(i + 1) * 0.1f;
  for (int i = 0; i < 32; ++i)
    wData[i] = static_cast<float>(i + 1) * 0.01f;

  auto x = runtime_->createTensor({8}, DataType::Float32, xData.data());
  auto w = runtime_->createTensor({8, 4}, DataType::Float32, wData.data());

  auto &ops = runtime_->ops();
  auto x2d = ops.reshape(x, {1, 8});
  auto mm = ops.matmul(x2d, w);
  auto act = ops.unaryOp(UnarySilu, mm);
  auto out = ops.reshape(act, {4});
  std::vector<float> result(4);
  runtime_->copyFromTensor(out, result.data(), 4 * sizeof(float));

  // Verify output is non-zero (SiLU of matmul result)
  for (int i = 0; i < 4; ++i) {
    EXPECT_NE(result[i], 0.0f);
  }
}

TEST_F(GraphTest, AutoFlushExplicit) {
  // Test explicit flush() call (not triggered by copyFromTensor)
  std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  auto result = runtime_->ops().unaryOp(UnarySqrt, a);
  runtime_->flush();

  // Now read the resolved result
  std::vector<float> output(4);
  runtime_->copyFromTensor(result, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(data[i]), 1e-5f);
  }
}

TEST_F(GraphTest, AutoFlushReduce) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  auto result = runtime_->ops().reduce(ReduceSum, a);
  float output = 0.0f;
  runtime_->copyFromTensor(result, &output, sizeof(float));

  EXPECT_NEAR(output, 10.0f, 1e-5f);
}

TEST_F(GraphTest, AutoFlushTranspose) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_->createTensor({2, 3}, DataType::Float32, data.data());

  auto result = runtime_->ops().transpose(a);
  std::vector<float> output(6);
  runtime_->copyFromTensor(result, output.data(), 6 * sizeof(float));

  EXPECT_NEAR(output[0], 1.0f, 1e-5f);
  EXPECT_NEAR(output[1], 4.0f, 1e-5f);
  EXPECT_NEAR(output[2], 2.0f, 1e-5f);
  EXPECT_NEAR(output[3], 5.0f, 1e-5f);
  EXPECT_NEAR(output[4], 3.0f, 1e-5f);
  EXPECT_NEAR(output[5], 6.0f, 1e-5f);
}

// ============================================================================
// MemoryPlanner Tests
// ============================================================================

TEST_F(GraphTest, MemoryPlannerNoTransients) {
  // All nodes are inputs or outputs — planner (run by executor) does nothing
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  builder.markOutput(sum);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], 2.0f * data[i]);
  }
}

TEST_F(GraphTest, MemoryPlannerLinearChain) {
  // input → add → sin → neg → cos → output
  // 3 transient nodes (add, sin, neg). In a linear chain, a producer overlaps
  // with its immediate consumer but NOT with nodes 2+ steps away.
  // So add and neg can share memory via arena allocation.
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {0.5f, 0.5f, 0.5f, 0.5f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto s = builder.ops().unaryOp(UnarySin, sum);
  auto n = builder.ops().unaryOp(UnaryNeg, s);
  auto result = builder.ops().unaryOp(UnaryCos, n);
  builder.markOutput(result);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    float expected = std::cos(-std::sin(aData[i] + bData[i]));
    EXPECT_NEAR(output[i], expected, 1e-5f);
  }
}

TEST_F(GraphTest, MemoryPlannerDiamond) {
  // input → sin, input → cos, (sin, cos) → add → output
  // sin and cos are both transient but overlap in lifetime
  std::vector<float> data = {0.5f, 1.0f, 1.5f, 2.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto s = builder.ops().unaryOp(UnarySin, va);
  auto c = builder.ops().unaryOp(UnaryCos, va);
  auto sum = builder.ops().binaryOp(BinaryAdd, s, c);
  builder.markOutput(sum);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sin(data[i]) + std::cos(data[i]), 1e-5f);
  }
}

TEST_F(GraphTest, MemoryPlannerSequentialReuse) {
  // input → A → B → C → output
  // A and C have non-overlapping lifetimes, so they can share memory
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto opA = builder.ops().unaryOp(UnarySin, va);   // transient
  auto opB = builder.ops().unaryOp(UnaryCos, opA);  // transient
  auto opC = builder.ops().unaryOp(UnarySqrt, opB); // transient
  auto opD = builder.ops().unaryOp(UnaryNeg, opC);  // output
  builder.markOutput(opD);
  auto graph = builder.build();

  GraphExecutor executor(runtime_->ops(), runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> output(4);
  runtime_->copyFromTensor(results[0], output.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    float expected = -std::sqrt(std::cos(std::sin(data[i])));
    EXPECT_NEAR(output[i], expected, 1e-4f);
  }
}

TEST_F(GraphTest, MemoryPlannerWithOptimizer) {
  // Full pipeline: optimize → execute (which plans memory internally)
  std::vector<float> xData(8);
  std::vector<float> wData(8 * 4);
  for (int i = 0; i < 8; ++i)
    xData[i] = static_cast<float>(i + 1) * 0.1f;
  for (int i = 0; i < 32; ++i)
    wData[i] = static_cast<float>(i + 1) * 0.01f;

  auto x = runtime_->createTensor({8}, DataType::Float32, xData.data());
  auto w = runtime_->createTensor({8, 4}, DataType::Float32, wData.data());

  // Eager reference
  auto &ops = runtime_->ops();
  auto eager_x2d = ops.reshape(x, {1, 8});
  auto eager_mm = ops.matmul(eager_x2d, w);
  auto eager_act = ops.unaryOp(UnarySilu, eager_mm);
  auto eager_out = ops.reshape(eager_act, {4});
  std::vector<float> eagerResult(4);
  runtime_->copyFromTensor(eager_out, eagerResult.data(), 4 * sizeof(float));

  // Graph with optimize + execute (planner runs inside executor)
  GraphBuilder builder(*runtime_);
  auto vx = builder.input(x);
  auto vw = builder.input(w, true);
  auto vx2d = builder.ops().reshape(vx, {1, 8});
  auto vmm = builder.ops().matmul(vx2d, vw);
  auto vact = builder.ops().unaryOp(UnarySilu, vmm);
  auto vout = builder.ops().reshape(vact, {4});
  builder.markOutput(vout);

  auto graph = builder.build();
  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphResult(4);
  runtime_->copyFromTensor(results[0], graphResult.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphResult[i], eagerResult[i], 1e-5f);
  }
}

// ============================================================================
// Fused Binary Pass Tests
// ============================================================================

TEST_F(GraphTest, FusedBinaryVecVecVecScalar) {
  // Pattern: (a + b) * 2.0 → fused into single dispatch
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerSum, 2.0f);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimization
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto scaled = builder.ops().binaryOp(BinaryMul, sum, 2.0f);
  builder.markOutput(scaled);
  auto graph = builder.build();

  // Verify fusion happens
  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // Execute and compare
  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecScalarVecVec) {
  // Pattern: (a * 2.0) + b → fused into single dispatch
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {10.0f, 20.0f, 30.0f, 40.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerScaled = ops.binaryOp(BinaryMul, a, 2.0f);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerScaled, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimization
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto scaled = builder.ops().binaryOp(BinaryMul, va, 2.0f);
  auto sum = builder.ops().binaryOp(BinaryAdd, scaled, vb);
  builder.markOutput(sum);
  auto graph = builder.build();

  // Verify fusion happens
  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  // Execute and compare
  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryNoFusionMultiConsumer) {
  // When intermediate has multiple consumers, fusion should NOT happen
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto scaled = builder.ops().binaryOp(BinaryMul, sum, 2.0f);
  // Mark both sum and scaled as outputs — sum has 2 consumers (scaled + output)
  builder.markOutput(sum);
  builder.markOutput(scaled);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_FALSE(changed); // Should not fuse since sum has refCount > 1
}

TEST_F(GraphTest, FusedBinaryFullPipeline) {
  // End-to-end: build graph, optimize with full pipeline, execute
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Eager reference: (a + b) * 3.0
  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerSum, 3.0f);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with full optimizer
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto scaled = builder.ops().binaryOp(BinaryMul, sum, 3.0f);
  builder.markOutput(scaled);
  auto graph = builder.build();

  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecVecScalarBuf) {
  // Pattern: (a + b) * scalarBuf → fused VecVecVecScalarBuf
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  float scalarVal = 2.0f;
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());
  auto s = runtime_->createTensor({1}, DataType::Float32, &scalarVal);

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerSum, s);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimizer
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto vs = builder.input(s);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto scaled = builder.ops().binaryOp(BinaryMul, sum, vs);
  builder.markOutput(scaled);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecScalarBufVecVec) {
  // Pattern: (a * scalarBuf) + b → fused VecScalarBufVecVec
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  float scalarVal = 3.0f;
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());
  auto s = runtime_->createTensor({1}, DataType::Float32, &scalarVal);

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerScaled = ops.binaryOp(BinaryMul, a, s);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerScaled, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimizer
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto vs = builder.input(s);
  auto scaled = builder.ops().binaryOp(BinaryMul, va, vs);
  auto sum = builder.ops().binaryOp(BinaryAdd, scaled, vb);
  builder.markOutput(sum);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryUnaryVecVec) {
  // Pattern: silu(a) * b → fused UnaryVecVec
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerAct = ops.unaryOp(UnarySilu, a);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerAct, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimizer
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto act = builder.ops().unaryOp(UnarySilu, va);
  auto result = builder.ops().binaryOp(BinaryMul, act, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecUnary) {
  // Pattern: silu(a + b) → fused VecVecUnary
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  // Eager reference
  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.unaryOp(UnarySilu, eagerSum);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph with optimizer
  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().unaryOp(UnarySilu, sum);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_TRUE(changed);

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

// ============================================================================
// Fused Binary Pass — Extended Tests
// ============================================================================

TEST_F(GraphTest, FusedBinaryVecVecVecScalar_SubDiv) {
  // Pattern: (a - b) / 2.0 → fused VecVecVecScalar with Sub+Div
  std::vector<float> aData = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> bData = {1.0f, 5.0f, 10.0f, 15.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerDiff = ops.binaryOp(BinarySub, a, b);
  auto eagerResult = ops.binaryOp(BinaryDiv, eagerDiff, 2.0f);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto diff = builder.ops().binaryOp(BinarySub, va, vb);
  auto result = builder.ops().binaryOp(BinaryDiv, diff, 2.0f);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecScalarVecVec_SubAdd) {
  // Pattern: (a - 1.0) + b → fused VecScalarVecVec with Sub+Add
  std::vector<float> aData = {5.0f, 10.0f, 15.0f, 20.0f};
  std::vector<float> bData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerShifted = ops.binaryOp(BinarySub, a, 1.0f);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerShifted, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto shifted = builder.ops().binaryOp(BinarySub, va, 1.0f);
  auto result = builder.ops().binaryOp(BinaryAdd, shifted, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecVecScalar_MinMax) {
  // Pattern: min(a, b) * 0.5 → fused VecVecVecScalar with Min+Mul
  std::vector<float> aData = {3.0f, 1.0f, 7.0f, 2.0f};
  std::vector<float> bData = {5.0f, 0.5f, 4.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerMin = ops.binaryOp(BinaryMin, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerMin, 0.5f);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto m = builder.ops().binaryOp(BinaryMin, va, vb);
  auto result = builder.ops().binaryOp(BinaryMul, m, 0.5f);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryUnaryVecVec_Relu) {
  // Pattern: relu(a) * b → fused UnaryVecVec with Relu
  std::vector<float> aData = {-2.0f, 3.0f, -1.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerAct = ops.unaryOp(UnaryRelu, a);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerAct, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto act = builder.ops().unaryOp(UnaryRelu, va);
  auto result = builder.ops().binaryOp(BinaryMul, act, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryUnaryVecVec_Neg) {
  // Pattern: neg(a) + b → fused UnaryVecVec with Neg+Add
  std::vector<float> aData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> bData = {10.0f, 20.0f, 30.0f, 40.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerNeg = ops.unaryOp(UnaryNeg, a);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerNeg, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto neg = builder.ops().unaryOp(UnaryNeg, va);
  auto result = builder.ops().binaryOp(BinaryAdd, neg, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecUnary_Relu) {
  // Pattern: relu(a - b) → fused VecVecUnary with Sub+Relu
  std::vector<float> aData = {3.0f, 1.0f, 7.0f, 2.0f};
  std::vector<float> bData = {5.0f, 0.5f, 4.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerDiff = ops.binaryOp(BinarySub, a, b);
  auto eagerResult = ops.unaryOp(UnaryRelu, eagerDiff);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto diff = builder.ops().binaryOp(BinarySub, va, vb);
  auto result = builder.ops().unaryOp(UnaryRelu, diff);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecUnary_Gelu) {
  // Pattern: gelu(a * b) → fused VecVecUnary with Mul+Gelu
  std::vector<float> aData = {0.5f, 1.0f, -0.5f, 2.0f};
  std::vector<float> bData = {1.0f, 0.5f, 2.0f, 0.25f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerProd = ops.binaryOp(BinaryMul, a, b);
  auto eagerResult = ops.unaryOp(UnaryGelu, eagerProd);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto prod = builder.ops().binaryOp(BinaryMul, va, vb);
  auto result = builder.ops().unaryOp(UnaryGelu, prod);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecUnary_Exp) {
  // Pattern: exp(a + b) → fused VecVecUnary with Add+Exp
  std::vector<float> aData = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> bData = {0.5f, 0.4f, 0.3f, 0.2f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.unaryOp(UnaryExp, eagerSum);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().unaryOp(UnaryExp, sum);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryUnaryVecVec_Exp) {
  // Pattern: exp(a) * b → fused UnaryVecVec with Exp+Mul
  std::vector<float> aData = {0.1f, 0.2f, 0.3f, 0.4f};
  std::vector<float> bData = {2.0f, 3.0f, 4.0f, 5.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerExp = ops.unaryOp(UnaryExp, a);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerExp, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto e = builder.ops().unaryOp(UnaryExp, va);
  auto result = builder.ops().binaryOp(BinaryMul, e, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecVecScalarBuf_SubMul) {
  // Pattern: (a - b) * scalarBuf → fused VecVecVecScalarBuf with Sub+Mul
  std::vector<float> aData = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> bData = {1.0f, 2.0f, 3.0f, 4.0f};
  float scalarVal = 0.5f;
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());
  auto s = runtime_->createTensor({1}, DataType::Float32, &scalarVal);

  auto &ops = runtime_->ops();
  auto eagerDiff = ops.binaryOp(BinarySub, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerDiff, s);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto vs = builder.input(s);
  auto diff = builder.ops().binaryOp(BinarySub, va, vb);
  auto result = builder.ops().binaryOp(BinaryMul, diff, vs);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecScalarBufVecVec_DivAdd) {
  // Pattern: (a / scalarBuf) + b → fused VecScalarBufVecVec with Div+Add
  std::vector<float> aData = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> bData = {1.0f, 2.0f, 3.0f, 4.0f};
  float scalarVal = 5.0f;
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());
  auto s = runtime_->createTensor({1}, DataType::Float32, &scalarVal);

  auto &ops = runtime_->ops();
  auto eagerDiv = ops.binaryOp(BinaryDiv, a, s);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerDiv, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto vs = builder.input(s);
  auto div = builder.ops().binaryOp(BinaryDiv, va, vs);
  auto result = builder.ops().binaryOp(BinaryAdd, div, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryLargerTensor) {
  // Test fusion with a larger, non-trivially-sized tensor (non-aligned to 4)
  constexpr int N = 17;
  std::vector<float> aData(N), bData(N);
  for (int i = 0; i < N; ++i) {
    aData[i] = static_cast<float>(i + 1);
    bData[i] = static_cast<float>(N - i);
  }
  auto a = runtime_->createTensor({static_cast<uint32_t>(N)}, DataType::Float32,
                                  aData.data());
  auto b = runtime_->createTensor({static_cast<uint32_t>(N)}, DataType::Float32,
                                  bData.data());

  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.binaryOp(BinaryMul, eagerSum, 0.1f);
  std::vector<float> eagerOut(N);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), N * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().binaryOp(BinaryMul, sum, 0.1f);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(N);
  runtime_->copyFromTensor(results[0], graphOut.data(), N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinary2D) {
  // Test fusion with a 2D tensor
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> bData = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  auto a = runtime_->createTensor({2, 4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({2, 4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerResult = ops.unaryOp(UnarySilu, eagerSum);
  std::vector<float> eagerOut(8);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 8 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().unaryOp(UnarySilu, sum);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(8);
  runtime_->copyFromTensor(results[0], graphOut.data(), 8 * sizeof(float));
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecVecScalar_MaxAdd) {
  // Pattern: max(a, b) + 1.0 → fused VecVecVecScalar with Max+Add
  std::vector<float> aData = {3.0f, -1.0f, 7.0f, 0.0f};
  std::vector<float> bData = {1.0f, 2.0f, 4.0f, -3.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerMax = ops.binaryOp(BinaryMax, a, b);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerMax, 1.0f);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto mx = builder.ops().binaryOp(BinaryMax, va, vb);
  auto result = builder.ops().binaryOp(BinaryAdd, mx, 1.0f);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, FusedBinaryVecScalarVecVec_PowSub) {
  // Pattern: (a ^ 2.0) - b → fused VecScalarVecVec with Pow+Sub
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {0.5f, 1.0f, 2.0f, 3.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerPow = ops.binaryOp(BinaryPow, a, 2.0f);
  auto eagerResult = ops.binaryOp(BinarySub, eagerPow, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto p = builder.ops().binaryOp(BinaryPow, va, 2.0f);
  auto result = builder.ops().binaryOp(BinarySub, p, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryUnaryVecVec_Sigmoid) {
  // Pattern: sigmoid(a) + b → fused UnaryVecVec with Sigmoid+Add
  std::vector<float> aData = {-2.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> bData = {1.0f, 1.0f, 1.0f, 1.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerSig = ops.unaryOp(UnarySigmoid, a);
  auto eagerResult = ops.binaryOp(BinaryAdd, eagerSig, b);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sig = builder.ops().unaryOp(UnarySigmoid, va);
  auto result = builder.ops().binaryOp(BinaryAdd, sig, vb);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryVecVecUnary_Tanh) {
  // Pattern: tanh(a * b) → fused VecVecUnary with Mul+Tanh
  std::vector<float> aData = {0.5f, 1.0f, -0.5f, 0.25f};
  std::vector<float> bData = {1.0f, 0.5f, 2.0f, 3.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  auto eagerProd = ops.binaryOp(BinaryMul, a, b);
  auto eagerResult = ops.unaryOp(UnaryTanh, eagerProd);
  std::vector<float> eagerOut(4);
  runtime_->copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto prod = builder.ops().binaryOp(BinaryMul, va, vb);
  auto result = builder.ops().unaryOp(UnaryTanh, prod);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  EXPECT_TRUE(pass.run(*graph, runtime_->store()));

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 1u);

  std::vector<float> graphOut(4);
  runtime_->copyFromTensor(results[0], graphOut.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphOut[i], eagerOut[i], 1e-5f);
  }
}

TEST_F(GraphTest, FusedBinaryNoFusionComparison) {
  // Comparison ops should NOT be fused (isBinaryArithmetic excludes them)
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {4.0f, 3.0f, 2.0f, 1.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto cmp = builder.ops().binaryOp(BinaryGreater, va, vb);
  auto result = builder.ops().binaryOp(BinaryMul, cmp, 2.0f);
  builder.markOutput(result);
  auto graph = builder.build();

  graph::FusedBinaryPass pass;
  bool changed = pass.run(*graph, runtime_->store());
  EXPECT_FALSE(changed);
}

TEST_F(GraphTest, FusedBinaryFullPipeline_MultipleOutputs) {
  // Full pipeline: two separate fusable chains, both fused independently
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_->createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_->createTensor({4}, DataType::Float32, bData.data());

  auto &ops = runtime_->ops();
  // Reference: output1 = (a + b) * 2.0, output2 = relu(a - b)
  auto eagerSum = ops.binaryOp(BinaryAdd, a, b);
  auto eagerOut1 = ops.binaryOp(BinaryMul, eagerSum, 2.0f);
  auto eagerDiff = ops.binaryOp(BinarySub, a, b);
  auto eagerOut2 = ops.unaryOp(UnaryRelu, eagerDiff);
  std::vector<float> refOut1(4), refOut2(4);
  runtime_->copyFromTensor(eagerOut1, refOut1.data(), 4 * sizeof(float));
  runtime_->copyFromTensor(eagerOut2, refOut2.data(), 4 * sizeof(float));

  GraphBuilder builder(*runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto out1 = builder.ops().binaryOp(BinaryMul, sum, 2.0f);
  auto diff = builder.ops().binaryOp(BinarySub, va, vb);
  auto out2 = builder.ops().unaryOp(UnaryRelu, diff);
  builder.markOutput(out1);
  builder.markOutput(out2);
  auto graph = builder.build();

  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(*graph, runtime_->store());

  GraphExecutor executor(ops, runtime_->store());
  auto results = executor.execute(*graph);
  ASSERT_EQ(results.size(), 2u);

  std::vector<float> graphOut1(4), graphOut2(4);
  runtime_->copyFromTensor(results[0], graphOut1.data(), 4 * sizeof(float));
  runtime_->copyFromTensor(results[1], graphOut2.data(), 4 * sizeof(float));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut1[i], refOut1[i]);
    EXPECT_FLOAT_EQ(graphOut2[i], refOut2[i]);
  }
}

} // namespace
} // namespace graph
} // namespace cut
