#include <gtest/gtest.h>

#include <ComputeCommon.h>

using namespace cut;

// ThreadGroupSize tests

TEST(ThreadGroupSize, DefaultConstructorInitializesToZero) {
  ThreadGroupSize tgs;
  EXPECT_EQ(tgs.tgSizeX, 0);
  EXPECT_EQ(tgs.tgSizeY, 0);
  EXPECT_EQ(tgs.tgSizeZ, 0);
}

TEST(ThreadGroupSize, AggregateInitialization) {
  ThreadGroupSize tgs{8, 8, 1};
  EXPECT_EQ(tgs.tgSizeX, 8);
  EXPECT_EQ(tgs.tgSizeY, 8);
  EXPECT_EQ(tgs.tgSizeZ, 1);
}

TEST(ThreadGroupSize, PartialInitialization) {
  ThreadGroupSize tgs{16};
  EXPECT_EQ(tgs.tgSizeX, 16);
  EXPECT_EQ(tgs.tgSizeY, 0);
  EXPECT_EQ(tgs.tgSizeZ, 0);
}

// DataReference tests

TEST(DataReference, ConstructFromPrimitive) {
  int value = 42;
  DataReference ref(value);
  EXPECT_EQ(ref.ptr, &value);
  EXPECT_EQ(ref.size, sizeof(int));
}

TEST(DataReference, ConstructFromStruct) {
  struct TestStruct {
    float x, y, z, w;
  };
  TestStruct data{1.0f, 2.0f, 3.0f, 4.0f};
  DataReference ref(data);
  EXPECT_EQ(ref.ptr, &data);
  EXPECT_EQ(ref.size, sizeof(TestStruct));
}

TEST(DataReference, ConstructFromPointerAndSize) {
  std::vector<uint32_t> data = {1, 2, 3, 4};
  DataReference ref(data.data(), static_cast<uint32_t>(data.size() * sizeof(uint32_t)));
  EXPECT_EQ(ref.ptr, data.data());
  EXPECT_EQ(ref.size, 16);
}

TEST(DataReference, ConstructFromArray) {
  float array[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  DataReference ref(array);
  EXPECT_EQ(ref.ptr, array);
  EXPECT_EQ(ref.size, sizeof(array));
}
