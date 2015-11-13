/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#pragma once

namespace thpp { namespace test {

template <class T>
void testUniqueMove() {
  auto a = T({2});
  EXPECT_TRUE(a.isUnique());
  EXPECT_EQ(2, a.size());

  auto b = a;
  EXPECT_FALSE(a.isUnique());
  EXPECT_EQ(2, a.size());
  EXPECT_FALSE(b.isUnique());
  EXPECT_EQ(2, b.size());

  auto c = std::move(a);
  EXPECT_TRUE(a.isUnique());
  EXPECT_EQ(0, a.size());
  EXPECT_FALSE(b.isUnique());
  EXPECT_EQ(2, b.size());
  EXPECT_FALSE(c.isUnique());
  EXPECT_EQ(2, c.size());

  b.clear();
  EXPECT_TRUE(a.isUnique());
  EXPECT_EQ(0, a.size());
  EXPECT_TRUE(b.isUnique());
  EXPECT_EQ(0, b.size());
  EXPECT_TRUE(c.isUnique());
  EXPECT_EQ(2, c.size());
}

template <class T>
void testTensorPtr() {
  auto p = T::makePtr({2});
  auto& x = *p;
  x.fill(1);
  EXPECT_EQ(2, x.sumall());

  auto q = p;
  auto& y = *q;
  y.resize({4});
  y.fill(2);

  EXPECT_EQ(8, x.sumall());
  EXPECT_EQ(8, y.sumall());

  EXPECT_TRUE(&p != &q);

  auto z = x;
  z.resize({6});
  z.fill(3);

  EXPECT_EQ(12, y.sumall());
  EXPECT_EQ(18, z.sumall());
}


}}  // namespaces
