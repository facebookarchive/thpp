/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_TENSOR_H_
#error This file may only be included from thpp/Tensor.h
#endif

namespace thpp {
namespace detail {

void serialize(
    ThriftTensor& out,
    LongRange sizes,
    LongRange strides,
    folly::IOBuf&& data,
    ThriftTensorDataType dtype,
    size_t elementSize,
    ThriftTensorEndianness endianness = ThriftTensorEndianness::NATIVE,
    bool mayShare = true);

template <class ThriftObj>
folly::IOBuf deserialize(ThriftObj&& in,
                         ThriftTensorDataType dtype);

}}  // namespaces
