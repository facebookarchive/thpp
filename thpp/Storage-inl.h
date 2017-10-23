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

#ifndef NO_FOLLY
#include <folly/memory/Malloc.h>
#include <folly/ScopeGuard.h>
#include <folly/Format.h>
#endif

#ifndef THPP_STORAGE_H_
#error This file may only be included from thpp/Storage.h
#endif


namespace thpp {

namespace detail {

#ifndef NO_FOLLY
void applySharingMode(folly::IOBuf& iob, SharingMode sharing);
#endif

////////////////////////////////////////////////////////////////////////////////
#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
////////////////////////////////////////////////////////////////////////////////

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
               SharingMode sharing);

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

////////////////////////////////////////////////////////////////////////////////
#endif // !NO_THRIFT && !NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

extern THAllocator ioBufTHAllocator;
extern THAllocator ioBufTHAllocatorNoRealloc;

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
  auto data = std::unique_ptr<T, decltype(&free)>({
      static_cast<T*>(malloc(n * sizeof(T))),
      free});
  if (!data) throw std::bad_alloc();
  std::copy(begin, end, data.get());
  this->t_ = Ops::_newWithData(data.get(), n);
  data.release();
}

template <class T>
Storage<T>::Storage(size_t n, T value) {
  if (n == 0) {
    this->t_ = nullptr;
    return;
  }
  auto data = std::unique_ptr<T, decltype(&free)>({
      static_cast<T*>(malloc(n * sizeof(T))),
      free});
  if (!data) throw std::bad_alloc();
  std::fill_n(data.get(), n, value);
  this->t_ = Ops::_newWithData(data.get(), n);
  data.release();
}

template <class T>
Storage<T>::Storage(THType* t) : Base(t) {
  this->up();
}

////////////////////////////////////////////////////////////////////////////////
#ifndef NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

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
Storage<T> Storage<T>::wrapWithAllocator(Range<T*> data,
                                         THAllocator* allocator,
                                         void* allocatorContext) {
  Storage<T> s;
  s.t_ = Ops::_newWithDataAndAllocator(
      data.data(), data.size(), allocator, allocatorContext);
  return s;
}

////////////////////////////////////////////////////////////////////////////////
#endif // !NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

// Fallback without folly
template <class T>
Storage<T> Storage<T>::wrapWithAllocator(T* data, size_t size,
                                         THAllocator* allocator,
                                         void* allocatorContext) {
  Storage<T> s;
  s.t_ = Ops::_newWithDataAndAllocator(data, size, allocator, allocatorContext);
  return s;
}

template <class T>
Storage<T> Storage<T>::withAllocator(THAllocator* allocator,
                                     void* allocatorContext) {
  Storage<T> s;
  s.t_ = Ops::_newWithDataAndAllocator(
      nullptr, 0, allocator, allocatorContext);
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

////////////////////////////////////////////////////////////////////////////////
#ifndef NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

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

struct THAllocFreeFuncData {
  THAllocator* allocator;
  void* context;

  THAllocFreeFuncData(THAllocator* allocator, void* context);
};

void THAllocFreeFunc(void* buf, void* userData);

}  // namespace detail

template <class T>
folly::IOBuf Storage<T>::getIOBuf() {
  if (!this->t_) return folly::IOBuf();

  auto iobTHAllocator =
    &detail::ioBufTHAllocator;
  auto iobTHAllocatorNoRealloc =
    &detail::ioBufTHAllocatorNoRealloc;

  auto len = this->size() * sizeof(T);
  auto curAllocator = this->t_->allocator;
  if (curAllocator == &THDefaultAllocator) {
    // Switch to using IOBuf allocator.
    // We know that memory from the default allocator was allocated with
    // malloc, just like IOBuf, so we know how to free it.
    this->t_->allocator = iobTHAllocator;
    this->t_->allocatorContext = new detail::IOBufAllocator(folly::IOBuf(
            folly::IOBuf::TAKE_OWNERSHIP, this->data(), len, len));
  } else if (curAllocator == iobTHAllocator ||
             curAllocator == iobTHAllocatorNoRealloc) {
    // do nothing
  } else {
    // The storage was allocated with an unknown allocator (neither default
    // nor IOBuf), so we must remember the previous allocator and allocator
    // context and call that allocator's free method when necessary.

    auto freeFuncData = new detail::THAllocFreeFuncData(
      this->t_->allocator, this->t_->allocatorContext);

    this->t_->allocator = iobTHAllocatorNoRealloc;
    this->t_->allocatorContext = new detail::IOBufAllocator(folly::IOBuf(
            folly::IOBuf::TAKE_OWNERSHIP, this->data(), len, len,
            detail::THAllocFreeFunc, freeFuncData));
  }


  auto allocator = static_cast<detail::IOBufAllocator*>(
      this->t_->allocatorContext);
  return allocator->clone();
}

template <class T>
Storage<T>::Storage(folly::IOBuf&& iob, SharingMode sharing,
                    bool resizable) : Base(nullptr) {
  setFromIOBuf(std::move(iob), sharing, resizable);
}

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
template <class T>
Storage<T>::Storage(const ThriftStorage& in, SharingMode sharing)
  : Base(nullptr) {
  setFromIOBuf(detail::deserialize(in, detail::dataType<T>()), sharing, true);
}
#endif

template <class T>
void Storage<T>::setFromIOBuf(folly::IOBuf&& iob, SharingMode sharing,
                              bool resizable) {
  size_t len = iob.computeChainDataLength();
  if (len % sizeof(T) != 0) {
    throw std::invalid_argument("IOBuf size must be multiple of data size");
  }
  len /= sizeof(T);

  iob.coalesce();
  detail::applySharingMode(iob, sharing);

  // Ensure properly aligned, make a copy otherwise. coalesce()
  // and/or applySharingMode() might have already done that for us,
  // in which case we're likely already aligned.
  if ((reinterpret_cast<uintptr_t>(iob.data()) % alignof(T)) != 0) {
    iob = folly::IOBuf(folly::IOBuf::COPY_BUFFER, iob.data(), iob.length());
  }

  T* p = reinterpret_cast<T*>(iob.writableData());
  this->t_ = Ops::_newWithDataAndAllocator(
      p, len,
      &detail::ioBufTHAllocator,
      new detail::IOBufAllocator(std::move(iob)));

  if (!resizable) {
    Ops::_clearFlag(this->t_, TH_STORAGE_RESIZABLE);
  }
}

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
template <class T>
void Storage<T>::serialize(ThriftStorage& out,
                           ThriftTensorEndianness endianness,
                           SharingMode sharing) const {
  detail::serialize(out, const_cast<Storage*>(this)->getIOBuf(),
                    detail::dataType<T>(), endianness, sharing);
}
#endif

////////////////////////////////////////////////////////////////////////////////
#endif // !NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

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

#ifndef NO_FOLLY
  // Check all our supported allocators. Currently one.
  auto iobTHAllocator = &detail::ioBufTHAllocator;
  if (th->allocator == iobTHAllocator) {
    return static_cast<const detail::IOBufAllocator*>(th->allocatorContext)->
      isUnique(th->data);
  }
#endif

  // Unknown allocator. Be on the safe side.
  return false;
}

}  // namespaces
