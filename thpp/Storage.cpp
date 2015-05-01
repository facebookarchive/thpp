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
  : iob_(std::move(iob)) {
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
    if (extra > iob_.tailroom()) {
      iob_.unshareOne();
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

}  // namespace detail

}  // namespaces
