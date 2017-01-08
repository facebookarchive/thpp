#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7 FATAL_ERROR)

OPTION(NO_THRIFT "enabling this will exclude all code that depends on Thrift from the build" OFF)
OPTION(NO_FOLLY  "enabling this will exclude all code that depends on Folly from the build" OFF)
OPTION(NO_TESTS  "enabling this will disable building tests" OFF)

# Torch messes this up
SET(SAVED_CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

# Use our own cmake scripts
SET(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

INCLUDE(MultiLevelIncludes)
MLI_SET_DEPTH(1)

FIND_PACKAGE(Torch REQUIRED)

IF(NOT NO_FOLLY)
  FIND_PACKAGE(Folly REQUIRED)
  INCLUDE_DIRECTORIES(${FOLLY_INCLUDE_DIR})
ELSE()
  ADD_DEFINITIONS(-DNO_FOLLY)
ENDIF()

IF(NOT NO_THRIFT)
  IF(NOT FOLLY_FOUND)
    MESSAGE(SEND_ERROR "Thift serialization requires folly")
  ENDIF()
  FIND_PACKAGE(Thrift REQUIRED)
  INCLUDE_DIRECTORIES(${THRIFT_INCLUDE_DIR})
ELSE()
  ADD_DEFINITIONS(-DNO_THRIFT)
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

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
  Tensor.h
  Tensor-inl.h
  StorageBase.h
  StorageBase-inl.h
  TensorBase.h
  TensorBase-inl.h
  TensorPtr.h
  TensorPtr-inl.h
)

SET(h_detail
  detail/Storage.h
  detail/StorageDefsGeneric.h
  detail/StorageGeneric.h
  detail/Tensor.h
  detail/TensorDefsGeneric.h
  detail/TensorGeneric.h
)

IF(THRIFT_FOUND)
  ADD_THRIFT2(src "if/Tensor.thrift")
  INSTALL_THRIFT2_HEADERS("if" "include")
ENDIF()

ADD_LIBRARY(thpp SHARED ${src})

if (UNIX AND NOT APPLE)
  # Let's check if --no-as-needed is supported by the linker.
  INCLUDE(CheckCCompilerFlag)
  SET(_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  SET(CMAKE_REQUIRED_FLAGS "-Wl,--no-as-needed")
  CHECK_C_COMPILER_FLAG("" HAS_NO_AS_NEEDED)
  SET(CMAKE_REQUIRED_FLAGS ${_CMAKE_REQUIRED_FLAGS})
  IF(NOT HAS_NO_AS_NEEDED)
    MESSAGE(SEND_ERROR "Your linker doesn't support a required --no-as-needed flag")
  ENDIF()

  # This is a template library, so some dependent libraries are only used from
  # header files, so we'd like to force them to be linked.
  SET(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
endif (UNIX AND NOT APPLE)

TARGET_LINK_LIBRARIES(thpp TH)
IF(FOLLY_FOUND)
  TARGET_LINK_LIBRARIES(thpp ${FOLLY_LIBRARIES})
ENDIF()
IF(THRIFT_FOUND)
  TARGET_LINK_LIBRARIES(thpp ${THRIFT_LIBRARIES})
ENDIF()


IF(NOT NO_TESTS)
  ENABLE_TESTING()
  FIND_PACKAGE(Glog REQUIRED)
  INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(thpp ${GLOG_LIBRARIES})
  ADD_SUBDIRECTORY("googletest-release-1.7.0")
  INCLUDE_DIRECTORIES(
    "${CMAKE_CURRENT_SOURCE_DIR}/googletest-release-1.7.0/include"
  )
  ADD_SUBDIRECTORY("test")
ENDIF()


# SET(CMAKE_INSTALL_PREFIX ${SAVED_CMAKE_INSTALL_PREFIX})

INSTALL(TARGETS thpp
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

INSTALL(FILES ${h}
  DESTINATION "include/${MLI_INCLUDE_RELATIVE_OUTPUT_DIR}")
INSTALL(FILES ${h_detail}
  DESTINATION "include/${MLI_INCLUDE_RELATIVE_OUTPUT_DIR}/detail")
