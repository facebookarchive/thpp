/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <thpp/Storage.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp {
namespace test {

typedef Storage<float> FloatStorage;
TEST(Storage, Simple) {
  FloatStorage s({2, 3, 4});
  EXPECT_EQ(3, s.size());
  EXPECT_EQ(2, s.at(0));
  EXPECT_EQ(3, s.at(1));
  EXPECT_EQ(4, s.at(2));
}

}}  // namespaces
