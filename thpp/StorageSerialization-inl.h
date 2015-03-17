/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_STORAGE_H_
#error This file may only be included from thpp/Storage.h
#endif

#include <folly/Format.h>

namespace thpp {
namespace detail {

// Endianness of current machine.
constexpr ThriftTensorEndianness gMachineEndianness =
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  ThriftTensorEndianness::LITTLE;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  ThriftTensorEndianness::BIG;
#else
# error Weird endianness!
#endif

template <class T> struct DataType;

#define X(TYPE, DTYPE, SIZE) \
  template <> struct DataType<TYPE> { \
    static_assert(sizeof(TYPE) == SIZE, \
                  "Invalid size for " #TYPE); \
    static constexpr ThriftTensorDataType value = \
      ThriftTensorDataType::DTYPE; \
    static constexpr size_t size = SIZE; \
  };

X(unsigned char, BYTE, 1)
X(int32_t, INT32, 4)
X(int64_t, INT64, 8)
X(float, FLOAT, 4)
X(double, DOUBLE, 8)

#undef X

template <class T>
constexpr ThriftTensorDataType dataType() {
  return DataType<T>::value;
}

void serialize(ThriftStorage& out,
               folly::IOBuf&& data,
               ThriftTensorDataType dtype,
               ThriftTensorEndianness endianness,
               bool mayShare=true);

template <class ThriftObj>
folly::IOBuf deserialize(ThriftObj&& in,
                         ThriftTensorDataType dtype) {
  if (dtype != in.dataType) {
    throw std::invalid_argument(folly::sformat(
        "Invalid Thrift tensor data type {}, expected {}",
        int(in.dataType), int(dtype)));
  }
  if (in.endianness != gMachineEndianness) {
    throw std::invalid_argument(folly::sformat(
        "Non-native endianness not yet implemented: {}, expected {}",
        int(in.endianness), int(gMachineEndianness)));
  }

  return std::move(in.data);
}

}}  // namespaces
