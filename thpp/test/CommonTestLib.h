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

#include <thpp/Tensor.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp { namespace test {

template <class T>
void testUniqueMove();

template <class T>
void testTensorPtr();

}}  // namespaces

#include <thpp/test/CommonTestLib-inl.h>
