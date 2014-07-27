/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <thpp/Tensor.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp {
namespace test {

class TensorTest : public testing::Test {
 protected:
  void SetUp();

  static LongTensor create(long bias);

  static void check2DTensor(const LongTensor& a, long a11, long a12, long a21,
                            long a22);

  LongTensor a;
  LongTensor b;
};

void TensorTest::SetUp() {
  a = create(1);
  b = create(51);
}

LongTensor TensorTest::create(long bias) {
  LongTensor t({10, 20, 30});
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
  return t;
}

void TensorTest::check2DTensor(const LongTensor& a, long a11, long a12,
                               long a21, long a22) {
  EXPECT_EQ(2, a.ndims());
  EXPECT_LE(2, a.size(0));
  EXPECT_LE(2, a.size(1));
  EXPECT_EQ(a11, a[0][0].front());
  EXPECT_EQ(a12, a[0][1].front());
  EXPECT_EQ(a21, a[1][0].front());
  EXPECT_EQ(a22, a[1][1].front());
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

TEST_F(TensorTest, NonFloatEqual) {
  EXPECT_TRUE(a.isExactlyEqual(a));
  EXPECT_TRUE(a.isApproximatelyEqual(a));
  EXPECT_FALSE(a.isExactlyEqual(b));
  EXPECT_FALSE(a.isApproximatelyEqual(b));
}

TEST_F(TensorTest, FloatEqual) {
  auto x = Tensor<float>{{1}};
  auto y = Tensor<float>{{1}};
  x.at({0}) = 1.0f;
  y.at({0}) = 1.0f;

  EXPECT_TRUE(x.isExactlyEqual(y));
  EXPECT_TRUE(x.isApproximatelyEqual(y));

  y.at({0}) = 1.000001f;
  EXPECT_FALSE(x.isExactlyEqual(y));
  EXPECT_TRUE(x.isApproximatelyEqual(y));
}

TEST_F(TensorTest, EqualMoreDimensionsThanSize) {
  auto x = Tensor<float>{{2, 2, 2}};
  for (long k = 0; k < x.size(0); ++k) {
    for (long j = 0; j < x.size(1); ++j) {
      for (long i = 0; i < x.size(2); ++i) {
        x.at({k, j, i}) = k * 3 + j * 2 + i;
      }
    }
  }

  auto y = Tensor<float>{{2, 2, 2}};
  for (long k = 0; k < x.size(0); ++k) {
    for (long j = 0; j < x.size(1); ++j) {
      for (long i = 0; i < x.size(2); ++i) {
        y.at({k, j, i}) = k * 3 + j * 2 + i;
      }
    }
  }

  auto z = Tensor<float>{{2, 2, 2}};
  for (long k = 0; k < x.size(0); ++k) {
    for (long j = 0; j < x.size(1); ++j) {
      for (long i = 0; i < x.size(2); ++i) {
        z.at({k, j, i}) = k * 4 + j * 3 + i * 2;
      }
    }
  }

  EXPECT_TRUE(x.isApproximatelyEqual(y));
  EXPECT_FALSE(x.isApproximatelyEqual(z));
  EXPECT_TRUE(x.isExactlyEqual(y));
  EXPECT_FALSE(x.isExactlyEqual(z));
}

TEST_F(TensorTest, EqualMismatch) {
  auto x = Tensor<float>{{1, 1, 1}};
  auto y = Tensor<float>{{1, 1, 1, 1}};

  EXPECT_THROW(x.isApproximatelyEqual(y), std::invalid_argument);
  EXPECT_THROW(x.isExactlyEqual(y), std::invalid_argument);

  auto z = Tensor<float>{{1, 2, 3}};
  auto w = Tensor<float>{{1, 2, 4}};

  EXPECT_THROW(z.isApproximatelyEqual(w), std::invalid_argument);
  EXPECT_THROW(z.isExactlyEqual(w), std::invalid_argument);
}

TEST_F(TensorTest, Str) {
  auto x = Tensor<float>{{2,3,4}};

  EXPECT_EQ("torch.FloatTensor(2x3x4)", x.str());
}

}}  // namespaces
