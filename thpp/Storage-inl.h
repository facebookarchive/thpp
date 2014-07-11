/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <algorithm>
#include <cstdlib>

#include <folly/Malloc.h>
#include <folly/ScopeGuard.h>

#ifndef THPP_STORAGE_H_
#error This file may only be included from thpp/Storage.h
#endif

namespace thpp {

template <class T>
Storage<T>::Storage() : t_(nullptr) { }

template <class T>
Storage<T>::Storage(std::initializer_list<T> data)
  : Storage(data.begin(), data.end()) { }

template <class T>
template <class It>
Storage<T>::Storage(It begin, It end) {
  // Do not use newWithSize, as it leaks memory on exception.
  auto n = std::distance(begin, end);
  if (n == 0) {
    t_ = nullptr;
    return;
  }
  T* data = static_cast<T*>(folly::checkedMalloc(n * sizeof(T)));
  SCOPE_FAIL { free(data); };
  std::copy(begin, end, data);
  t_ = Ops::_newWithData(data, n);
}

template <class T>
Storage<T>::Storage(size_t n, T value) {
  if (n == 0) {
    t_ = nullptr;
    return;
  }
  T* data = static_cast<T*>(folly::checkedMalloc(n * sizeof(T)));
  SCOPE_FAIL { free(data); };
  std::fill_n(data, n, value);
  t_ = Ops::_newWithData(data, n);
}

template <class T>
Storage<T>::Storage(THType* t) : t_(t) {
  up();
}

template <class T>
Storage<T> Storage<T>::takeOwnership(Range<T*> data) {
  Storage<T> s;
  if (!data.empty()) {
    s.t_ = Ops::_newWithData(data.data(), data.size());
  }
  return s;
}

template <class T>
Storage<T> Storage<T>::wrap(Range<T*> data) {
  Storage<T> s;
  if (!data.empty()) {
    s.t_ = Ops::_newWithData(data.data(), data.size());
    Ops::_clearFlag(s.t_, TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM);
  }
  return s;
}

template <class T>
Storage<T> Storage<T>::withAllocator(THAllocator* allocator,
                                     void* allocatorContext) {
  return wrapWithAllocator(Range<T*>(), allocator, allocatorContext);
}

template <class T>
Storage<T> Storage<T>::wrapWithAllocator(Range<T*> data,
                                         THAllocator* allocator,
                                         void* allocatorContext) {
  Storage<T> s;
  s.t_ = Ops::_newWithDataAndAllocator(
      data.data(), data.size(), allocator, allocatorContext);
  return s;
}

template <class T>
Storage<T>::~Storage() {
  down();
}

template <class T>
Storage<T>::Storage(Storage&& other) noexcept : t_(other.t_) {
  other.t_ = nullptr;
}

template <class T>
Storage<T>::Storage(const Storage& other) : Storage(other.t_) { }

template <class T>
Storage<T>& Storage<T>::operator=(Storage&& other) {
  if (&other != this) {
    down();
    t_ = other.t_;
    other.t_ = nullptr;
  }
  return *this;
}

template <class T>
Storage<T>& Storage<T>::operator=(const Storage& other) {
  if (&other != this) {
    down();
    t_ = other.t_;
    up();
  }
  return *this;
}

template <class T>
void Storage<T>::resizeUninitialized(size_t n) {
  if (n == 0) {
    down();
    t_ = nullptr;
    return;
  }

  if (t_) {
    Ops::_resize(t_, n);
  } else {
    T* data = static_cast<T*>(folly::checkedMalloc(n * sizeof(T)));
    SCOPE_FAIL { free(data); };
    t_ = Ops::_newWithData(data, n);
  }
}

template <class T>
void Storage<T>::resize(size_t n, T value) {
  size_t oldSize = size();
  resizeUninitialized(n);

  if (n > oldSize) {
    std::fill(data() + oldSize, data() + n, value);
  }
}

template <class T>
template <class It>
void Storage<T>::assign(It begin, It end) {
  auto n = std::distance(begin, end);
  resizeUninitialized(n);
  std::copy(begin, end, data());
}

template <class T>
void Storage<T>::assign(size_t n, T value) {
  resizeUninitialized(n);
  std::fill_n(data(), n, value);
}

template <class T>
void Storage<T>::up() {
  if (t_) Ops::_retain(t_);
}

template <class T>
void Storage<T>::down() {
  if (t_) Ops::_free(t_);
}

template <class T>
void Storage<T>::check(size_t index) const {
  if (UNLIKELY(index >= size())) {
    throw std::out_of_range("Storage index out of range");
  }
}

namespace detail {

/**
 * What follows is some ugly acrobatics to allow IOBuf and THStorage to
 * share memory.
 *
 * If we want to create a IOBuf that wraps a THStorage, we need to ensure
 * that the memory doesn't get freed until the last reference to the IOBuf
 * is gone. This is easy; IOBuf allows setting a "free" callback, so we
 * keep a reference to the THStorage object for as long as references to
 * IOBuf exist.
 *
 * However, THStorage allows resizing, which calls realloc(), so we need to
 * ensure that the *original* chunk of memory stays allocated. We do this
 * by interposing a custom THStorage allocator (PreserveAllocator) that defers
 * reallocation / free of a given pointer until its destruction.
 *
 * Conversely, if we want to create a THStorage object that wraps an IOBuf,
 * we'll use a custom allocator that keeps a reference to the IOBuf and
 * calls appropriate methods on the IOBuf. We're relying on the slightly
 * unsafe (and undocumented) behavior that THStorage will only call the
 * "free" method of the allocator once at the end of its lifetime.
 */

/**
 * THAllocator-like object that preserves the chunk of memory
 * from preservePtr to preserveSize, preventing it from being freed until
 * its destruction. It forwards all allocation / reallocation / deallocation
 * requests to another THAllocator otherwise.
 */
class PreserveAllocator {
 public:
  PreserveAllocator(void* preservePtr,
                    long preserveSize,
                    THAllocator* prevAllocator,
                    void* prevAllocatorContext);
  ~PreserveAllocator();

  void* malloc(long size);
  void* realloc(void* ptr, long size);
  void free(void* ptr);

  THAllocator* prevAllocator() const { return prevAllocator_; }
  void* prevAllocatorContext() const { return prevAllocatorContext_; }

 private:
  void* preservePtr_;
  long preserveSize_;
  bool preserved_;
  THAllocator* prevAllocator_;
  void* prevAllocatorContext_;
};

/**
 * Base class for StorageHolder, below.
 */
class StorageHolderBase {
 public:
  virtual ~StorageHolderBase() { }

 protected:
  explicit StorageHolderBase(void* preservePtr, long preserveSize,
                             THAllocator* prevAllocator,
                             void* prevAllocatorContext);
  PreserveAllocator allocator_;
};

/**
 * Holder class that keeps a reference to a THStorage.
 */
template <class T>
class StorageHolder : public StorageHolderBase {
 public:
  explicit StorageHolder(Storage<T> s)
    : StorageHolderBase(s.data(), s.size() * sizeof(T),
                        s.t_->allocator, s.t_->allocatorContext),
      storage_(std::move(s)) {
    storage_.t_->allocator =
      &THAllocatorWrapper<PreserveAllocator>::thAllocator;
    storage_.t_->allocatorContext = &allocator_;
  }

  ~StorageHolder() {
    storage_.t_->allocator = allocator_.prevAllocator();
    storage_.t_->allocatorContext = allocator_.prevAllocatorContext();
  }

 private:
  Storage<T> storage_;
};

class IOBufAllocator {
 public:
  explicit IOBufAllocator(folly::IOBuf&& iob);

  void* malloc(long size);
  void* realloc(void* ptr, long size);
  void free(void* ptr);

 private:
  folly::IOBuf iob_;
};

/**
 * Free function (IOBuf callback) that releases the StorageHolder's reference.
 */
void storageUnrefFreeFunction(void* buf, void* userData);

}  // namespace detail

template <class T>
folly::IOBuf Storage<T>::getIOBuf() {
  if (!t_) return folly::IOBuf();

  return folly::IOBuf(
      folly::IOBuf::TakeOwnershipOp(),
      data(), size() * sizeof(T), size() * sizeof(T),
      detail::storageUnrefFreeFunction,
      static_cast<void*>(new detail::StorageHolder<T>(*this)));
}

template <class T>
Storage<T>::Storage(folly::IOBuf&& iob, bool mayShare)
  : t_(nullptr) {
  setFromIOBuf(std::move(iob), mayShare);
}

template <class T>
Storage<T>::Storage(ThriftStorage& in, bool mayShare)
  : t_(nullptr) {
  setFromIOBuf(detail::deserialize(in, detail::dataType<T>()), mayShare);
}

template <class T>
void Storage<T>::setFromIOBuf(folly::IOBuf&& iob, bool mayShare) {
  if (iob.computeChainDataLength() % sizeof(T) != 0) {
    throw std::invalid_argument("IOBuf size must be multiple of data size");
  }
  // TODO(#4146201): Read-only version that allows a shared IOBuf
  if (mayShare && !iob.isChained() && !iob.isSharedOne()) {
    // extract in variables, as we're about to use std::move(iob)
    T* p = reinterpret_cast<T*>(iob.writableData());
    long len = iob.length() / sizeof(T);
    t_ = Ops::_newWithDataAndAllocator(
        p, len,
        &THAllocatorWrapper<detail::IOBufAllocator>::thAllocator,
        new detail::IOBufAllocator(std::move(iob)));
  } else {
    t_ = Ops::_newWithSize(iob.computeChainDataLength() / sizeof(T));

    uint8_t* dst = reinterpret_cast<uint8_t*>(data());

    if (!iob.empty()) {
      memcpy(dst, iob.data(), iob.length());
      dst += iob.length();
    }

    auto iptr = iob.pop();
    while (iptr) {
      memcpy(dst, iptr->data(), iptr->length());
      dst += iptr->length();
      iptr = iptr->pop();
    }
  }
}

template <class T>
void Storage<T>::serialize(ThriftStorage& out,
                           ThriftTensorEndianness endianness,
                           bool mayShare) const {
  detail::serialize(out, const_cast<Storage*>(this)->getIOBuf(),
                    detail::dataType<T>(), endianness, mayShare);
}

template <class T>
auto Storage<T>::moveAsTH() -> THType* {
  using std::swap;
  THType* out = nullptr;
  swap(out, t_);
  return out;
}

template <class A>
THAllocator THAllocatorWrapper<A>::thAllocator = {
  &THAllocatorWrapper<A>::malloc,
  &THAllocatorWrapper<A>::realloc,
  &THAllocatorWrapper<A>::free,
};

}  // namespaces
