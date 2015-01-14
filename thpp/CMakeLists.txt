#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7 FATAL_ERROR)

# Torch messes this up
SET(SAVED_CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

# Use our own cmake scripts
SET(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

ENABLE_TESTING()

INCLUDE(MultiLevelIncludes)
MLI_SET_DEPTH(1)

FIND_PACKAGE(Folly REQUIRED)
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(Thrift REQUIRED)
INCLUDE_DIRECTORIES(
  ${FOLLY_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${THRIFT_INCLUDE_DIR}
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

ADD_SUBDIRECTORY("gtest-1.7.0")
INCLUDE_DIRECTORIES(
  "${CMAKE_CURRENT_SOURCE_DIR}/gtest-1.7.0/include"
)

SET(src
  Storage.cpp
  StorageSerialization.cpp
  detail/StorageDefs.cpp
  TensorSerialization.cpp
  detail/TensorDefs.cpp
)

SET(h
  Storage.h
  Storage-inl.h
  StorageSerialization-inl.h
  Tensor.h
  Tensor-inl.h
  TensorSerialization-inl.h
)

SET(h_detail
  detail/Storage.h
  detail/StorageGeneric.h
  detail/Tensor.h
  detail/TensorGeneric.h
)

ADD_THRIFT2(src "if/Tensor.thrift")

ADD_LIBRARY(thpp SHARED ${src})

# This is a template library, so some dependent libraries are only used from
# header files, so we'd like to force them to be linked.
SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")

TARGET_LINK_LIBRARIES(thpp
  TH ${FOLLY_LIBRARIES} ${GLOG_LIBRARIES} ${THRIFT_LIBRARIES})

ADD_SUBDIRECTORY("test")

SET(CMAKE_INSTALL_PREFIX ${SAVED_CMAKE_INSTALL_PREFIX})

INSTALL(TARGETS thpp
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

INSTALL(FILES ${h}
  DESTINATION "include/${MLI_INCLUDE_RELATIVE_OUTPUT_DIR}")
INSTALL(FILES ${h_detail}
  DESTINATION "include/${MLI_INCLUDE_RELATIVE_OUTPUT_DIR}/detail")
INSTALL_THRIFT2_HEADERS("if" "include")
