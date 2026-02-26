#include <gtest/gtest.h>

#include <ComputeCommon.h>
#include <ComputeOps.h>
#include <OpNode.h>
#include <Operations.h>
#include <Runtime.h>
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
    if (!runtime_.isVulkanAvailable()) {
      GTEST_SKIP() << "Vulkan not available";
    }
    runtime_.init(BackendType::Vulkan);
  }

  void TearDown() override { runtime_.shutdown(); }

  Runtime runtime_;
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
  auto tensor = runtime_.createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto x = builder.input(tensor);
  builder.markOutput(x);
  auto graph = builder.build();

  EXPECT_EQ(graph.size(), 1u);
  EXPECT_EQ(graph.outputs().size(), 1u);
  EXPECT_TRUE(graph.node(x).isInputNode());
  EXPECT_EQ(graph.node(x).outputShape(), (std::vector<uint32_t>{4, 8}));
  EXPECT_EQ(graph.node(x).outputDtype(), DataType::Float32);
}

TEST_F(GraphTest, LinearChain) {
  auto a = runtime_.createTensor({8}, DataType::Float32);
  auto b = runtime_.createTensor({8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  auto result = builder.ops().unaryOp(UnarySqrt, sum);
  builder.markOutput(result);
  auto graph = builder.build();

  EXPECT_EQ(graph.size(), 4u);
  EXPECT_EQ(graph.node(result).outputShape(), (std::vector<uint32_t>{8}));

  auto order = graph.topologicalOrder();
  EXPECT_EQ(order.size(), 4u);
  // inputs come first, then sum, then result
  uint32_t sumId = graph.nodeId(sum);
  uint32_t resultId = graph.nodeId(result);
  EXPECT_TRUE(std::find(order.begin(), order.end(), sumId) != order.end());
  auto sumPos = std::find(order.begin(), order.end(), sumId);
  auto resultPos = std::find(order.begin(), order.end(), resultId);
  EXPECT_TRUE(sumPos < resultPos);
}

TEST_F(GraphTest, DiamondDAG) {
  // Test fan-out + fan-in: x → a, x → b, (a, b) → c
  auto t = runtime_.createTensor({16}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto x = builder.input(t);
  auto a = builder.ops().unaryOp(UnarySin, x);
  auto b = builder.ops().unaryOp(UnaryCos, x);
  auto c = builder.ops().binaryOp(BinaryAdd, a, b);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph.size(), 4u);
  // x has refCount 2 (used by both a and b)
  graph.recomputeRefCounts();
  EXPECT_EQ(graph.node(x).graphRefCount(), 2u);
}

TEST_F(GraphTest, MatMulShapeInference) {
  auto a = runtime_.createTensor({3, 4}, DataType::Float32);
  auto b = runtime_.createTensor({4, 5}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto c = builder.ops().matmul(va, vb);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(c).outputShape(), (std::vector<uint32_t>{3, 5}));
  EXPECT_EQ(graph.node(c).outputDtype(), DataType::Float32);
}

TEST_F(GraphTest, TransposeShapeInference) {
  auto a = runtime_.createTensor({3, 7}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto t = builder.ops().transpose(va);
  builder.markOutput(t);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(t).outputShape(), (std::vector<uint32_t>{7, 3}));
}

TEST_F(GraphTest, ReshapeShapeInference) {
  auto a = runtime_.createTensor({12}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {3, 4});
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(r).outputShape(), (std::vector<uint32_t>{3, 4}));
}

TEST_F(GraphTest, ReshapeWithNegativeOne) {
  auto a = runtime_.createTensor({12}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {3, -1});
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(r).outputShape(), (std::vector<uint32_t>{3, 4}));
}

TEST_F(GraphTest, ReduceGlobalShapeInference) {
  auto a = runtime_.createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reduce(ReduceSum, va);
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(r).outputShape(), (std::vector<uint32_t>{1}));
}

TEST_F(GraphTest, ReduceDimShapeInference) {
  auto a = runtime_.createTensor({4, 8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reduce(ReduceSum, va, 0);
  builder.markOutput(r);
  auto graph = builder.build();

  EXPECT_EQ(graph.node(r).outputShape(), (std::vector<uint32_t>{8}));
}

// ============================================================================
// Optimization Pass Tests
// ============================================================================

TEST_F(GraphTest, IdentityReshapeElimination) {
  // reshape(x, same_shape) → x
  auto a = runtime_.createTensor({8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {8}); // Same shape → identity
  builder.markOutput(r);
  auto graph = builder.build();

  uint32_t vaId = graph.nodeId(va);

  EXPECT_EQ(graph.size(), 2u);

  IdentityReshapePass pass;
  bool changed = pass.run(graph);
  EXPECT_TRUE(changed);

  // Output should now point to the input node
  EXPECT_EQ(graph.outputs()[0], vaId);
}

TEST_F(GraphTest, ReshapeChainElimination) {
  // reshape(reshape(x, {1, 8}), {8}) → reshape(x, {8})
  auto a = runtime_.createTensor({8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r1 = builder.ops().reshape(va, {1, 8}); // 8 → 1x8
  auto r2 = builder.ops().reshape(r1, {8});    // 1x8 → 8
  builder.markOutput(r2);
  auto graph = builder.build();

  uint32_t vaId = graph.nodeId(va);
  uint32_t r2Id = graph.nodeId(r2);

  EXPECT_EQ(graph.size(), 3u);

  ReshapeChainPass chainPass;
  chainPass.run(graph);

  // r2 should now point directly to va's output (skip r1)
  EXPECT_EQ(graph.node(r2Id).graphInputIds()[0], vaId);
}

TEST_F(GraphTest, TransposeCancelElimination) {
  auto a = runtime_.createTensor({3, 5}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto t1 = builder.ops().transpose(va);
  auto t2 = builder.ops().transpose(t1);
  builder.markOutput(t2);
  auto graph = builder.build();

  uint32_t vaId = graph.nodeId(va);

  TransposeCancelPass pass;
  bool changed = pass.run(graph);
  EXPECT_TRUE(changed);

  // Output should now point to the original input
  EXPECT_EQ(graph.outputs()[0], vaId);
}

TEST_F(GraphTest, DeadCodeElimination) {
  auto a = runtime_.createTensor({8}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto dead = builder.ops().unaryOp(UnarySin, va); // Not used by output
  auto live = builder.ops().unaryOp(UnaryCos, va);
  builder.markOutput(live);
  auto graph = builder.build();

  EXPECT_EQ(graph.size(), 3u);

  DeadCodePass pass;
  bool changed = pass.run(graph);
  EXPECT_TRUE(changed);

  // The dead node should be marked as removed
  EXPECT_TRUE(graph.node(dead).isGraphRemoved());
  EXPECT_FALSE(graph.node(live).isGraphRemoved());
}

TEST_F(GraphTest, FullOptimizationPipeline) {
  // Simulates FFN-like pattern: reshape → matmul → reshape back
  // The outer reshapes may be eliminable
  auto x = runtime_.createTensor({8}, DataType::Float32);
  auto w = runtime_.createTensor({8, 4}, DataType::Float32);

  GraphBuilder builder(runtime_);
  auto vx = builder.input(x);
  auto vw = builder.input(w, true);

  auto reshaped = builder.ops().reshape(vx, {1, 8});
  auto mm = builder.ops().matmul(reshaped, vw);          // [1, 4]
  auto activated = builder.ops().unaryOp(UnarySilu, mm); // [1, 4]
  auto out = builder.ops().reshape(activated, {4});      // [4]
  builder.markOutput(out);

  auto graph = builder.build();
  size_t origSize = graph.size();

  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(graph);

  // The topological order should only contain non-removed nodes
  auto order = graph.topologicalOrder();
  EXPECT_LE(order.size(), origSize);
}

// ============================================================================
// Executor Round-Trip Tests
// ============================================================================

TEST_F(GraphTest, ExecutorBinaryOp) {
  // Create data
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_.createTensor({4}, DataType::Float32, bData.data());

  // Build graph
  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto sum = builder.ops().binaryOp(BinaryAdd, va, vb);
  builder.markOutput(sum);
  auto graph = builder.build();

  // Execute
  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);
  ASSERT_EQ(results.size(), 1u);

  // Read back and verify
  std::vector<float> output(4);
  runtime_.copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], aData[i] + bData[i]);
  }
}

TEST_F(GraphTest, ExecutorUnaryOp) {
  std::vector<float> aData = {1.0f, 4.0f, 9.0f, 16.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, aData.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto sq = builder.ops().unaryOp(UnarySqrt, va);
  builder.markOutput(sq);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  std::vector<float> output(4);
  runtime_.copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(aData[i]), 1e-5f);
  }
}

TEST_F(GraphTest, ExecutorMatMul) {
  // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
  // Result = [[19, 22], [43, 50]]
  std::vector<float> aData = {1, 2, 3, 4};
  std::vector<float> bData = {5, 6, 7, 8};
  auto a = runtime_.createTensor({2, 2}, DataType::Float32, aData.data());
  auto b = runtime_.createTensor({2, 2}, DataType::Float32, bData.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto vb = builder.input(b);
  auto mm = builder.ops().matmul(va, vb);
  builder.markOutput(mm);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  std::vector<float> output(4);
  runtime_.copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  EXPECT_NEAR(output[0], 19.0f, 1e-4f);
  EXPECT_NEAR(output[1], 22.0f, 1e-4f);
  EXPECT_NEAR(output[2], 43.0f, 1e-4f);
  EXPECT_NEAR(output[3], 50.0f, 1e-4f);
}

TEST_F(GraphTest, ExecutorReshape) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_.createTensor({6}, DataType::Float32, data.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto r = builder.ops().reshape(va, {2, 3});
  builder.markOutput(r);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  // Verify shape via reading back all elements
  std::vector<float> output(6);
  runtime_.copyFromTensor(results[0], output.data(), 6 * sizeof(float));

  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(output[i], data[i]);
  }
}

TEST_F(GraphTest, ExecutorVecScalarOp) {
  std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto scaled = builder.ops().binaryOp(BinaryMul, va, 0.5f);
  builder.markOutput(scaled);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  std::vector<float> output(4);
  runtime_.copyFromTensor(results[0], output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], data[i] * 0.5f);
  }
}

TEST_F(GraphTest, ExecutorTranspose) {
  // 2x3 matrix → 3x2
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_.createTensor({2, 3}, DataType::Float32, data.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto t = builder.ops().transpose(va);
  builder.markOutput(t);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  std::vector<float> output(6);
  runtime_.copyFromTensor(results[0], output.data(), 6 * sizeof(float));

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
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto sum = builder.ops().reduce(ReduceSum, va);
  builder.markOutput(sum);
  auto graph = builder.build();

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);

  float output = 0.0f;
  runtime_.copyFromTensor(results[0], &output, sizeof(float));

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

  auto x = runtime_.createTensor({8}, DataType::Float32, xData.data());
  auto w = runtime_.createTensor({8, 4}, DataType::Float32, wData.data());

  // Eager execution
  auto &ops = runtime_.ops();
  auto eager_x2d = ops.reshape(x, {1, 8});
  auto eager_mm = ops.matmul(eager_x2d, w);
  auto eager_act = ops.unaryOp(UnarySilu, eager_mm);
  auto eager_out = ops.reshape(eager_act, {4});

  std::vector<float> eagerResult(4);
  runtime_.copyFromTensor(eager_out, eagerResult.data(), 4 * sizeof(float));

  // Graph execution (with optimization)
  GraphBuilder builder(runtime_);
  auto vx = builder.input(x);
  auto vw = builder.input(w, true);
  auto vx2d = builder.ops().reshape(vx, {1, 8});
  auto vmm = builder.ops().matmul(vx2d, vw);
  auto vact = builder.ops().unaryOp(UnarySilu, vmm);
  auto vout = builder.ops().reshape(vact, {4});
  builder.markOutput(vout);

  auto graph = builder.build();
  auto optimizer = GraphOptimizer::createDefault();
  optimizer.optimize(graph);

  GraphExecutor executor(ops, runtime_);
  auto results = executor.execute(graph);

  std::vector<float> graphResult(4);
  runtime_.copyFromTensor(results[0], graphResult.data(), 4 * sizeof(float));

  // Should match exactly
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphResult[i], eagerResult[i], 1e-5f);
  }
}

TEST_F(GraphTest, MultiOutputGraph) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  GraphBuilder builder(runtime_);
  auto va = builder.input(a);
  auto s = builder.ops().unaryOp(UnarySin, va);
  auto c = builder.ops().unaryOp(UnaryCos, va);
  builder.markOutput(s);
  builder.markOutput(c);
  auto graph = builder.build();

  EXPECT_EQ(graph.outputs().size(), 2u);

  GraphExecutor executor(runtime_.ops(), runtime_);
  auto results = executor.execute(graph);
  ASSERT_EQ(results.size(), 2u);

  std::vector<float> sinOut(4), cosOut(4);
  runtime_.copyFromTensor(results[0], sinOut.data(), 4 * sizeof(float));
  runtime_.copyFromTensor(results[1], cosOut.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(sinOut[i], std::sin(data[i]), 1e-5f);
    EXPECT_NEAR(cosOut[i], std::cos(data[i]), 1e-5f);
  }
}

// ============================================================================
// Runtime Graph Mode Tests (beginGraph / executeGraph / copyFromTensor)
// ============================================================================

TEST_F(GraphTest, RuntimeGraphModeBinaryOp) {
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_.createTensor({4}, DataType::Float32, bData.data());

  // Eager execution for reference
  auto eagerResult = runtime_.ops().binaryOp(BinaryAdd, a, b);
  std::vector<float> eagerOut(4);
  runtime_.copyFromTensor(eagerResult, eagerOut.data(), 4 * sizeof(float));

  // Graph mode execution
  runtime_.beginGraph();
  auto graphResult = runtime_.ops().binaryOp(BinaryAdd, a, b);
  // copyFromTensor triggers executeGraph automatically
  std::vector<float> graphOut(4);
  runtime_.copyFromTensor(graphResult, graphOut.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(graphOut[i], eagerOut[i]);
  }
}

TEST_F(GraphTest, RuntimeGraphModeUnaryOp) {
  std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  runtime_.beginGraph();
  auto result = runtime_.ops().unaryOp(UnarySqrt, a);
  std::vector<float> output(4);
  runtime_.copyFromTensor(result, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(data[i]), 1e-5f);
  }
}

TEST_F(GraphTest, RuntimeGraphModeMatMul) {
  std::vector<float> aData = {1, 2, 3, 4};
  std::vector<float> bData = {5, 6, 7, 8};
  auto a = runtime_.createTensor({2, 2}, DataType::Float32, aData.data());
  auto b = runtime_.createTensor({2, 2}, DataType::Float32, bData.data());

  runtime_.beginGraph();
  auto result = runtime_.ops().matmul(a, b);
  std::vector<float> output(4);
  runtime_.copyFromTensor(result, output.data(), 4 * sizeof(float));

  EXPECT_NEAR(output[0], 19.0f, 1e-4f);
  EXPECT_NEAR(output[1], 22.0f, 1e-4f);
  EXPECT_NEAR(output[2], 43.0f, 1e-4f);
  EXPECT_NEAR(output[3], 50.0f, 1e-4f);
}

TEST_F(GraphTest, RuntimeGraphModeChain) {
  // Test multi-op chain: (a + b) * 2.0
  std::vector<float> aData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> bData = {5.0f, 6.0f, 7.0f, 8.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, aData.data());
  auto b = runtime_.createTensor({4}, DataType::Float32, bData.data());

  runtime_.beginGraph();
  auto &ops = runtime_.ops();
  auto sum = ops.binaryOp(BinaryAdd, a, b);
  float two = 2.0f;
  auto scaled =
      ops.binaryOp(BinaryMul, sum, DataReference(&two, sizeof(float)));
  std::vector<float> output(4);
  runtime_.copyFromTensor(scaled, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(output[i], (aData[i] + bData[i]) * 2.0f);
  }
}

TEST_F(GraphTest, RuntimeGraphModeFFN) {
  // FFN-like: reshape → matmul → SiLU → reshape
  std::vector<float> xData(8);
  std::vector<float> wData(8 * 4);
  for (int i = 0; i < 8; ++i)
    xData[i] = static_cast<float>(i + 1) * 0.1f;
  for (int i = 0; i < 32; ++i)
    wData[i] = static_cast<float>(i + 1) * 0.01f;

  auto x = runtime_.createTensor({8}, DataType::Float32, xData.data());
  auto w = runtime_.createTensor({8, 4}, DataType::Float32, wData.data());

  // Eager reference
  auto &ops = runtime_.ops();
  auto eager_x2d = ops.reshape(x, {1, 8});
  auto eager_mm = ops.matmul(eager_x2d, w);
  auto eager_act = ops.unaryOp(UnarySilu, eager_mm);
  auto eager_out = ops.reshape(eager_act, {4});
  std::vector<float> eagerResult(4);
  runtime_.copyFromTensor(eager_out, eagerResult.data(), 4 * sizeof(float));

  // Graph mode
  runtime_.beginGraph();
  auto graph_x2d = ops.reshape(x, {1, 8});
  auto graph_mm = ops.matmul(graph_x2d, w);
  auto graph_act = ops.unaryOp(UnarySilu, graph_mm);
  auto graph_out = ops.reshape(graph_act, {4});
  std::vector<float> graphResult(4);
  runtime_.copyFromTensor(graph_out, graphResult.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(graphResult[i], eagerResult[i], 1e-5f);
  }
}

TEST_F(GraphTest, RuntimeGraphModeExplicitExecute) {
  // Test explicit executeGraph() call (not triggered by copyFromTensor)
  std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  runtime_.beginGraph();
  auto result = runtime_.ops().unaryOp(UnarySqrt, a);
  runtime_.executeGraph();
  EXPECT_FALSE(runtime_.isGraphMode());

  // Now read the resolved result
  std::vector<float> output(4);
  runtime_.copyFromTensor(result, output.data(), 4 * sizeof(float));

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(output[i], std::sqrt(data[i]), 1e-5f);
  }
}

TEST_F(GraphTest, RuntimeGraphModeReduce) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto a = runtime_.createTensor({4}, DataType::Float32, data.data());

  runtime_.beginGraph();
  auto result = runtime_.ops().reduce(ReduceSum, a);
  float output = 0.0f;
  runtime_.copyFromTensor(result, &output, sizeof(float));

  EXPECT_NEAR(output, 10.0f, 1e-5f);
}

TEST_F(GraphTest, RuntimeGraphModeTranspose) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto a = runtime_.createTensor({2, 3}, DataType::Float32, data.data());

  runtime_.beginGraph();
  auto result = runtime_.ops().transpose(a);
  std::vector<float> output(6);
  runtime_.copyFromTensor(result, output.data(), 6 * sizeof(float));

  EXPECT_NEAR(output[0], 1.0f, 1e-5f);
  EXPECT_NEAR(output[1], 4.0f, 1e-5f);
  EXPECT_NEAR(output[2], 2.0f, 1e-5f);
  EXPECT_NEAR(output[3], 5.0f, 1e-5f);
  EXPECT_NEAR(output[4], 3.0f, 1e-5f);
  EXPECT_NEAR(output[5], 6.0f, 1e-5f);
}

} // namespace
} // namespace graph
} // namespace cut
