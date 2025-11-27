#include <gtest/gtest.h>

#include <Utils.h>

using namespace cut;

class MockComputeInterface : public ComputeInterface {
public:
  ComputeHandle createBuffer(size_t, const void * = nullptr, bool = false) {
    return bufferContainer_.createFloat();
  }

  void copyDataToBuffer(const void *, const ComputeHandle &, size_t, size_t,
                        size_t, bool = false, bool = false) {}

  void copyDataFromBuffer(const ComputeHandle &, void *, size_t, size_t, size_t,
                          bool = false, bool = false) {};

  ComputeHandle createShaderModule(const std::vector<uint32_t> &) {
    return shaderContainer_.createInteger();
  }

  void submit(const ComputeHandle &) {};

  MockContainer bufferContainer_;
  MockContainer shaderContainer_;
};

class ComputeTestEnvironment : public ::testing::Test {
public:
  void SetUp() {
    EXPECT_NO_THROW(interface = std::make_unique<MockComputeInterface>());
    EXPECT_NE(interface, nullptr);
  }

  void TearDown() { interface.reset(); }

  std::unique_ptr<cut::ComputeInterface> interface;
};

TEST_F(ComputeTestEnvironment, Dispatch) {
  auto buffer1 = interface->createBuffer(0);
  auto buffer2 = interface->createBuffer(1);

  auto shader = interface->createShaderModule({});

  auto disaptch1 = interface->createDispatch(
      shader, {1, 1, 1},
      {ComputeBinding(0, buffer1), ComputeBinding(1, buffer2)});

  auto disaptch2 = interface->createDispatchFromRef(disaptch1);

  EXPECT_THROW(auto disaptch3 = interface->createDispatchFromRef(disaptch2),
               std::runtime_error);
}

TEST_F(ComputeTestEnvironment, DispatchList) {
  auto buffer1 = interface->createBuffer(0);
  auto buffer2 = interface->createBuffer(1);

  auto shader = interface->createShaderModule({});

  auto disaptch1 = interface->createDispatch(
      shader, {1, 1, 1},
      {ComputeBinding(0, buffer1), ComputeBinding(1, buffer2)});

  auto disaptch2 = interface->createDispatch(shader, {1, 1, 1},
                                             {ComputeBinding(0, buffer1)});

  auto disaptch3 = interface->createDispatch(shader, {1, 1, 1},
                                             {ComputeBinding(0, buffer2)});

  auto disaptch4 = interface->createDispatchFromRef(disaptch1);
}
