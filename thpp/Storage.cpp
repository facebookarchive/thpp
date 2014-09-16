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
  if (size > iob_.capacity()) {
    iob_.unshareOne();
    iob_.reserve(0, size - iob_.capacity());
  }
  if (size < iob_.length()) {
    iob_.trimEnd(iob_.length() - size);
  } else {
    iob_.append(size - iob_.length());
  }
  return iob_.writableData();
}

void IOBufAllocator::free(void* ptr) {
  CHECK_EQ(ptr, iob_.writableData());
  delete this;
}

}  // namespace detail

}  // namespaces
