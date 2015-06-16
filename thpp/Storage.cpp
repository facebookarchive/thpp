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

IOBufAllocator::IOBufAllocator(folly::IOBuf&& iob)
  : iob_(std::move(iob)),
    maxLength_(iob_.isSharedOne() ? iob_.length() :
               std::numeric_limits<uint64_t>::max()) {
  DCHECK(!iob_.isChained());
}

void* IOBufAllocator::malloc(long size) {
  CHECK(false) << "IOBufAllocator::malloc should never be called";
}

void* IOBufAllocator::realloc(void* ptr, long size) {
  CHECK_EQ(ptr, iob_.writableData());
  if (size <= iob_.length()) {
    iob_.trimEnd(iob_.length() - size);
  } else {
    auto extra = size - iob_.length();
    // If we're still using the original buffer (which was shared), we
    // may only use up to the original buffer length; the rest of the buffer
    // might be filled with something else (other fields if decoding Thrift,
    // etc).
    if (size > maxLength_ || extra > iob_.tailroom()) {
      iob_.unshareOne();
      maxLength_ = std::numeric_limits<uint64_t>::max();
    }
    if (extra > iob_.tailroom()) {
      iob_.reserve(0, extra);
    }
    iob_.append(extra);
  }
  return iob_.writableData();
}

void IOBufAllocator::free(void* ptr) {
  CHECK_EQ(ptr, iob_.writableData());
  delete this;
}

bool IOBufAllocator::isUnique(const void* ptr) const {
  CHECK_EQ(ptr, iob_.data());
  return !iob_.isSharedOne();
}

}  // namespace detail

}  // namespaces
