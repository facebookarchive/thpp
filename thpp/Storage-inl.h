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
               bool mayShare);

template <class ThriftObj>
folly::IOBuf deserialize(const ThriftObj& in,
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

  return in.data;
}

}  // namespace detail

template <class T>
Storage<T>::Storage() : Base(nullptr) { }

template <class T>
Storage<T>::Storage(std::initializer_list<T> data)
  : Storage(data.begin(), data.end()) { }

template <class T>
template <class It>
Storage<T>::Storage(It begin, It end) {
  // Do not use newWithSize, as it leaks memory on exception.
  auto n = std::distance(begin, end);
  if (n == 0) {
    this->t_ = nullptr;
    return;
  }
  T* data = static_cast<T*>(folly::checkedMalloc(n * sizeof(T)));
  SCOPE_FAIL { free(data); };
  std::copy(begin, end, data);
  this->t_ = Ops::_newWithData(data, n);
}

template <class T>
Storage<T>::Storage(size_t n, T value) {
  if (n == 0) {
    this->t_ = nullptr;
    return;
  }
  T* data = static_cast<T*>(folly::checkedMalloc(n * sizeof(T)));
  SCOPE_FAIL { free(data); };
  std::fill_n(data, n, value);
  this->t_ = Ops::_newWithData(data, n);
}

template <class T>
Storage<T>::Storage(THType* t) : Base(t) {
  this->up();
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
  this->down();
}

template <class T>
Storage<T>::Storage(Storage&& other) noexcept : Base(other.t_) {
  other.t_ = nullptr;
}

template <class T>
Storage<T>::Storage(const Storage& other) : Storage(other.t_) { }

template <class T>
Storage<T>& Storage<T>::operator=(Storage&& other) {
  if (&other != this) {
    this->down();
    this->t_ = other.t_;
    other.t_ = nullptr;
  }
  return *this;
}

template <class T>
Storage<T>& Storage<T>::operator=(const Storage& other) {
  if (&other != this) {
    this->down();
    this->t_ = other.t_;
    this->up();
  }
  return *this;
}

template <class T>
void Storage<T>::resize(size_t n, T value) {
  size_t oldSize = this->size();
  this->resizeUninitialized(n);

  if (n > oldSize) {
    std::fill(this->data() + oldSize, this->data() + n, value);
  }
}

template <class T>
template <class It>
void Storage<T>::assign(It begin, It end) {
  auto n = std::distance(begin, end);
  this->resizeUninitialized(n);
  std::copy(begin, end, this->data());
}

template <class T>
void Storage<T>::assign(size_t n, T value) {
  this->resizeUninitialized(n);
  std::fill_n(this->data(), n, value);
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
  bool isUnique(const void* ptr) const;

  folly::IOBuf clone() {
    folly::IOBuf buf;
    iob_.cloneInto(buf);
    return buf;
  }

 private:
  folly::IOBuf iob_;
  uint64_t maxLength_;
};

}  // namespace detail

template <class T>
folly::IOBuf Storage<T>::getIOBuf() {
  if (!this->t_) return folly::IOBuf();

  auto iobTHAllocator =
    &THAllocatorWrapper<detail::IOBufAllocator>::thAllocator;

  if (this->t_->allocator == &THDefaultAllocator) {
    // Switch to using IOBuf allocator.
    // We know that memory from the default allocator was allocated with
    // malloc, just like IOBuf, so we know how to free it.
    this->t_->allocator = iobTHAllocator;
    this->t_->allocatorContext = new detail::IOBufAllocator(folly::IOBuf(
            folly::IOBuf::TAKE_OWNERSHIP,
            this->data(), this->size() * sizeof(T), this->size() * sizeof(T)));
  } else if (this->t_->allocator != iobTHAllocator) {
    throw std::invalid_argument(
        "Cannot convert to IOBuf, Storage was allocated with unknown "
        "allocator");
  }

  // If the storage was allocated with an unknown allocator (neither default
  // nor IOBuf), there are three obvious solutions, all wrong:
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

  auto allocator = static_cast<detail::IOBufAllocator*>(
      this->t_->allocatorContext);
  return allocator->clone();
}

template <class T>
Storage<T>::Storage(folly::IOBuf&& iob, bool mayShare) : Base(nullptr) {
  setFromIOBuf(std::move(iob), mayShare);
}

template <class T>
Storage<T>::Storage(const ThriftStorage& in, bool mayShare) : Base(nullptr) {
  setFromIOBuf(detail::deserialize(in, detail::dataType<T>()), mayShare);
}

template <class T>
void Storage<T>::setFromIOBuf(folly::IOBuf&& iob, bool mayShare) {
  size_t len = iob.computeChainDataLength();
  if (len % sizeof(T) != 0) {
    throw std::invalid_argument("IOBuf size must be multiple of data size");
  }
  len /= sizeof(T);
  iob.coalesce();
  if (!mayShare) {
    iob.unshareOne();
  }
  T* p = reinterpret_cast<T*>(iob.writableData());
  this->t_ = Ops::_newWithDataAndAllocator(
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
bool Storage<T>::isUnique(const THType* th) {
  if (!th) {
    return true;
  }
  if (th->refcount != 1) {
    return false;
  }
  // Even if the refcount is 1, this might share memory with other
  // resources from the outside world. Not possible with the default allocator.
  if (th->allocator == &THDefaultAllocator) {
    return true;
  }

  // Check all our supported allocators. Currently one.
  auto iobTHAllocator =
    &THAllocatorWrapper<detail::IOBufAllocator>::thAllocator;
  if (th->allocator == iobTHAllocator) {
    return static_cast<const detail::IOBufAllocator*>(th->allocatorContext)->
      isUnique(th->data);
  }

  // Unknown allocator. Be on the safe side.
  return false;
}

template <class A>
THAllocator THAllocatorWrapper<A>::thAllocator = {
  &THAllocatorWrapper<A>::malloc,
  &THAllocatorWrapper<A>::realloc,
  &THAllocatorWrapper<A>::free,
};

}  // namespaces
