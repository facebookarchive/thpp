/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <thpp/cuda/Tensor.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <thpp/test/CommonTestLib.h>

namespace thpp { namespace test {

class TensorTest : public testing::Test {
 protected:
  void SetUp();

  static CudaFloatTensor create(long bias);

  static void check2DTensor(const CudaFloatTensor& a,
                            float a11, float a12, float a21, float a22);

  CudaFloatTensor a;
  CudaFloatTensor b;
};

void TensorTest::SetUp() {
  a = create(1);
  b = create(51);
}

CudaFloatTensor TensorTest::create(long bias) {
  FloatTensor t({10, 20, 30});
  EXPECT_EQ(10 * 20 * 30, t.size());
  for (long i = 0; i < t.size(0); ++i) {
    for (long j = 0; j < t.size(1); ++j) {
      for (long k = 0; k < t.size(2); ++k) {
        t[i][j][k].front() =
          ((i + bias) * 10000) +
          ((j + bias) * 100) +
          (k + bias);
      }
    }
  }

  return CudaFloatTensor(t);
}

void TensorTest::check2DTensor(const CudaFloatTensor& ca, float a11, float a12,
                               float a21, float a22) {
  auto a = ca.toCPU();
  EXPECT_EQ(2, a->ndims());
  EXPECT_LE(2, a->size(0));
  EXPECT_LE(2, a->size(1));
  EXPECT_EQ(a11, (*a)[0][0].front());
  EXPECT_EQ(a12, (*a)[0][1].front());
  EXPECT_EQ(a21, (*a)[1][0].front());
  EXPECT_EQ(a22, (*a)[1][1].front());
}

TEST_F(TensorTest, Simple) {
  auto sub = a[5];
  EXPECT_EQ(20 * 30, sub.size());
  check2DTensor(sub, 60101, 60102, 60201, 60202);

  auto sub2 = a[{-1,5}];
  EXPECT_EQ(10 * 30, sub2.size());
  check2DTensor(sub2, 10601, 10602, 20601, 20602);
}

TEST_F(TensorTest, Add) {
  auto sub = (b + a)[5];
  EXPECT_EQ(20 * 30, sub.size());
  check2DTensor(sub, 625252, 625254, 625452, 625454);
}

TEST_F(TensorTest, Sub) {
  auto sub = (b - a)[5];
  EXPECT_EQ(20 * 30, sub.size());
  check2DTensor(sub, 505050, 505050, 505050, 505050);
}

TEST_F(TensorTest, At) {
  EXPECT_EQ(20304, a.at({1, 2, 3}));
  a.transpose(0, 1);
  EXPECT_EQ(40608, a.at({5, 3, 7}));
  a.transpose(1, 0);
  a.transpose();
  EXPECT_EQ(50403, a.at({2, 3, 4}));
}

TEST_F(TensorTest, UniqueMove) {
  testUniqueMove<CudaTensor<float>>();
}

TEST_F(TensorTest, TensorPtr) {
  testTensorPtr<CudaTensor<float>>();
}

}}  // namespaces
