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

PreserveAllocator::PreserveAllocator(void* preservePtr,
                                     long preserveSize,
                                     THAllocator* prevAllocator,
                                     void* prevAllocatorContext)
  : preservePtr_(preservePtr),
    preserveSize_(preserveSize),
    preserved_(false),
    prevAllocator_(prevAllocator),
    prevAllocatorContext_(prevAllocatorContext) {
}

void* PreserveAllocator::malloc(long size) {
  return (*prevAllocator_->malloc)(prevAllocatorContext_, size);
}

void* PreserveAllocator::realloc(void* ptr, long size) {
  if (ptr != preservePtr_) {
    assert(ptr != preservePtr_);
    return (*prevAllocator_->realloc)(prevAllocatorContext_, ptr, size);
  }
  assert(!preserved_);
  preserved_ = true;
  void* newPtr = (*prevAllocator_->malloc)(prevAllocatorContext_, size);
  memcpy(newPtr, preservePtr_, std::min(size, preserveSize_));
  return newPtr;
}

void PreserveAllocator::free(void* ptr) {
  if (ptr != preservePtr_) {
    (*prevAllocator_->free)(prevAllocatorContext_, ptr);
    return;
  }
  assert(!preserved_);
  preserved_ = true;
}

PreserveAllocator::~PreserveAllocator() {
  if (preserved_) {
    (*prevAllocator_->free)(prevAllocatorContext_, preservePtr_);
  }
}

void storageUnrefFreeFunction(void* buf, void* userData) {
  auto holder = static_cast<StorageHolderBase*>(userData);
  delete holder;
}

StorageHolderBase::StorageHolderBase(
    void* preservePtr,
    long preserveSize,
    THAllocator* prevAllocator,
    void* prevAllocatorContext)
  : allocator_(preservePtr, preserveSize, prevAllocator, prevAllocatorContext) {
}

IOBufAllocator::IOBufAllocator(folly::IOBuf&& iob)
  : iob_(std::move(iob)) {
  DCHECK(!iob_.isChained());
  DCHECK(!iob_.isShared());
}

void* IOBufAllocator::malloc(long size) {
  CHECK(false) << "IOBufAllocator::malloc should never be called";
}

void* IOBufAllocator::realloc(void* ptr, long size) {
  CHECK_EQ(ptr, iob_.writableData());
  if (size > iob_.length()) {
    long extra = size - iob_.length();
    iob_.reserve(0, extra);
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
