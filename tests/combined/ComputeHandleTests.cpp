#include <gtest/gtest.h>

#include <Utils.h>

TEST(MockContainer, InitTest) {
  auto container = std::make_unique<MockContainer>();
  EXPECT_TRUE(container != nullptr);
}

TEST(MockContainer, AllocTest) {
  auto container = std::make_unique<MockContainer>();

  auto h1 = container->createInteger();
  EXPECT_EQ(container->getIntData(h1), 1);

  auto h2 = container->createFloat();
  EXPECT_EQ(container->getFloatData(h2), 1);

  h1 = h2;
  EXPECT_EQ(container->getFloatData(h1), 1);
  EXPECT_EQ(container->getFloatData(h2), 1);

  h1 = container->createInteger();
  EXPECT_EQ(container->getIntData(h1), 2);
  EXPECT_EQ(container->getFloatData(h2), 1);

  h1.reset();
  EXPECT_THROW(container->getIntData(h1), std::runtime_error);

  h2 = h1;
  EXPECT_THROW(container->getIntData(h2), std::runtime_error);

  h2 = container->createInteger();

  EXPECT_NO_THROW(h1 = h2);
  h2.reset();

  EXPECT_NO_THROW(h1 = h1);
  h1.reset();

  EXPECT_NO_THROW(h1 = h1);

  h1.reset();
}
