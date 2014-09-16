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
 * If we want to create a THStorage object that wraps an IOBuf,
 * we'll use a custom allocator that keeps a reference to the IOBuf and
 * calls appropriate methods on the IOBuf. We're relying on the slightly
 * unsafe (and undocumented) behavior that THStorage will only call the
 * "free" method of the allocator once at the end of its lifetime.
 *
 * If we want to create an IOBuf that wraps a THStorage, we reduce it to
 * the case above by converting its memory to an IOBuf.
 */

class IOBufAllocator {
 public:
  explicit IOBufAllocator(folly::IOBuf&& iob);

  void* malloc(long size);
  void* realloc(void* ptr, long size);
  void free(void* ptr);

  folly::IOBuf clone() {
    folly::IOBuf buf;
    iob_.cloneInto(buf);
    return buf;
  }

 private:
  folly::IOBuf iob_;
};

}  // namespace detail

template <class T>
folly::IOBuf Storage<T>::getIOBuf() {
  if (!t_) return folly::IOBuf();

  auto iobTHAllocator =
    &THAllocatorWrapper<detail::IOBufAllocator>::thAllocator;

  if (t_->allocator == &THDefaultAllocator) {
    // Switch to using IOBuf allocator.
    // We know that memory from the default allocator was allocated with
    // malloc, just like IOBuf, so we know how to free it.
    t_->allocator = iobTHAllocator;
    t_->allocatorContext = new detail::IOBufAllocator(folly::IOBuf(
            folly::IOBuf::TAKE_OWNERSHIP,
            data(), size() * sizeof(T), size() * sizeof(T)));
  } else {
    // There are three obvious solutions here, all wrong.
    //
    // 1. We could memcpy() into our own IOBuf and proceed as above. This
    //    doesn't work because the memory might be allocated in an unusual
    //    way (different address space (CUDA), mlock()ed, unusual alignment
    //    requirements) and we don't know how to replicate that with IOBuf.
    //
    // 2. We could remember the previous allocator and allocator context and
    //    call that allocator's free method when necessary. This doesn't work
    //    because TH doesn't manage lifetimes of allocators and allocator
    //    contexts, so we can't know when a previous allocator and context
    //    are still valid (not deleted).
    //
    // 3. Just as in 2 above, we could remember the previous allocator and
    //    allocator context, but we'll also keep a reference to the Storage
    //    object, to prevent the allocator from being deleted. This means that
    //    the IOBuf has a reference to the Storage (via the free function
    //    that would ensure that the previous allocator's free method gets
    //    called) and the Storage has a reference to the IOBuf's memory
    //    (via IOBufAllocator), so the reference count for either never drops
    //    to 0, so we have a memory leak.
    CHECK(t_->allocator == iobTHAllocator)
      << "Can not convert to IOBuf, Storage was allocated with unknown "
         "allocator";
  }

  auto allocator = static_cast<detail::IOBufAllocator*>(t_->allocatorContext);
  return allocator->clone();
}

template <class T>
Storage<T>::Storage(folly::IOBuf&& iob) : t_(nullptr) {
  setFromIOBuf(std::move(iob));
}

template <class T>
Storage<T>::Storage(ThriftStorage&& in) : t_(nullptr) {
  setFromIOBuf(detail::deserialize(std::move(in), detail::dataType<T>()));
}

template <class T>
void Storage<T>::setFromIOBuf(folly::IOBuf&& iob) {
  size_t len = iob.computeChainDataLength();
  if (len % sizeof(T) != 0) {
    throw std::invalid_argument("IOBuf size must be multiple of data size");
  }
  len /= sizeof(T);
  iob.coalesce();
  iob.unshareOne();
  T* p = reinterpret_cast<T*>(iob.writableData());
  t_ = Ops::_newWithDataAndAllocator(
      p, len,
      &THAllocatorWrapper<detail::IOBufAllocator>::thAllocator,
      new detail::IOBufAllocator(std::move(iob)));
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
