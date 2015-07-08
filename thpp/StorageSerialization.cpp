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

namespace thpp {
namespace detail {

void serialize(
    ThriftStorage& out,
    folly::IOBuf&& data,
    ThriftTensorDataType dtype,
    ThriftTensorEndianness endianness,
    SharingMode sharing) {
  DCHECK(!data.isChained());
  if (endianness == ThriftTensorEndianness::NATIVE) {
    endianness = gMachineEndianness;
  } else {
    CHECK(endianness == gMachineEndianness)
      << "Non-native endianness not yet implemented";
  }

  out.dataType = dtype;
  out.endianness = endianness;
  detail::applySharingMode(data, sharing);
  out.data = std::move(data);
}

template folly::IOBuf deserialize(const ThriftStorage& in,
                                  ThriftTensorDataType dtype);

}}  // namespaces
