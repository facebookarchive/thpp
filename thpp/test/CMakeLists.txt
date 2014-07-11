#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#

ADD_EXECUTABLE(storage_test StorageTest.cpp)
TARGET_LINK_LIBRARIES(storage_test thpp gtest gtest_main)
ADD_TEST(storage_test storage_test)

ADD_EXECUTABLE(tensor_test TensorTest.cpp)
TARGET_LINK_LIBRARIES(tensor_test thpp gtest gtest_main)
ADD_TEST(tensor_test tensor_test)

ADD_EXECUTABLE(tensor_serialization_test TensorSerializationTest.cpp)
TARGET_LINK_LIBRARIES(tensor_serialization_test thpp gtest gtest_main)
ADD_TEST(tensor_serialization_test tensor_serialization_test)
