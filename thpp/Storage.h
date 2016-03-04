/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_STORAGE_H_
#define THPP_STORAGE_H_

#ifndef DCHECK
#include <cassert>
#define DCHECK(x) assert(x)
#endif

#include <initializer_list>
#include <memory>
#ifndef NO_THRIFT
#include <thpp/if/gen-cpp2/Tensor_types.h>
#endif
#ifndef NO_FOLLY
#include <folly/Malloc.h>
#include <folly/Range.h>
#include <folly/io/IOBuf.h>
#endif
#include <thpp/StorageBase.h>
#include <thpp/detail/Storage.h>

namespace thpp {

#ifndef NO_FOLLY
using folly::Range;
#endif

/**
 * Wrapper around TH's Storage type, which is a length-aware,
 * reference-counted, heap-allocated array.
 */
template <class T> class Tensor;
template <class T> class CudaTensor;

#ifndef NO_FOLLY
enum SharingMode {
  // Do not share memory with the given IOBuf.
  SHARE_NONE,

  // Share memory managed by IOBuf (no additional bookkeeping required)
  SHARE_IOBUF_MANAGED,

  // Share all memory, including external buffers (which might require you to
  // guarantee that such external buffers remain allocated until all IOBuf
  // and Storage objects that refer to them are visible)
  SHARE_ALL,
};
#endif

template <class T>
class Storage : public StorageBase<T, Storage<T>> {
  typedef StorageBase<T, Storage<T>> Base;
  typedef typename Base::Ops Ops;
  friend Base;  // Yay C++11
  friend class Tensor<T>;
 public:
  typedef typename Base::THType THType;
  Storage();

  explicit Storage(std::initializer_list<T> data);
  template <class It> Storage(It begin, It end);
  Storage(size_t n, T value);

  explicit Storage(THType* t);

////////////////////////////////////////////////////////////////////////////////
#ifndef NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

  explicit Storage(Range<const T*> range)
    : Storage(range.begin(), range.end()) { }


  // Create a Storage object containing the data from an IOBuf.
  // If sharing is not SHARE_NONE, then the Storage object will share memory
  // with the IOBuf, at least until either is resized.
  explicit Storage(folly::IOBuf&& iob,
                   SharingMode sharing = SHARE_IOBUF_MANAGED);
  explicit Storage(const folly::IOBuf& iob,
                   SharingMode sharing = SHARE_IOBUF_MANAGED)
    : Storage(folly::IOBuf(iob), sharing) { }

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
  // Deserialize from Thrift. Throws if wrong type.
  explicit Storage(const ThriftStorage& thriftStorage,
                   SharingMode sharing = SHARE_IOBUF_MANAGED);
#endif

  // Takes ownership of a range allocated with malloc() (NOT new or new[]!)
  static Storage takeOwnership(Range<T*> data);

  // Wrap a range of memory. The range must stay allocated until all Storage
  // objects that refer to it are gone.
  static Storage wrap(Range<T*> data);

  // Wrap a range of memory and use a custom allocator for reallocations.
  // You probably don't need this.
  static Storage wrapWithAllocator(Range<T*> data,
                                   THAllocator* allocator,
                                   void* allocatorContext);

////////////////////////////////////////////////////////////////////////////////
#endif // !NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

  // Use a custom allocator. The allocator is managed by the caller.
  static Storage withAllocator(THAllocator* allocator,
                               void* allocatorContext);

  ~Storage();

  Storage(Storage&& other) noexcept;
  Storage(const Storage& other);
  Storage& operator=(Storage&& other);
  Storage& operator=(const Storage& other);

  void resize(size_t n, T value = 0);

  template <class It> void assign(It begin, It end);
  void assign(size_t n, T value);

////////////////////////////////////////////////////////////////////////////////
#ifndef NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

  // Create a IOBuf that wraps the memory currently allocated to this
  // storage offset. The memory won't be freed until all references to it
  // are gone, either from IOBufs or from Storage objects. Note that
  // if this Storage is resized, it might not share memory with the
  // returned IOBuf any more.
  folly::IOBuf getIOBuf();

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
  // Serialize to Thrift.
  void serialize(ThriftStorage& out,
                 ThriftTensorEndianness endianness =
                     ThriftTensorEndianness::NATIVE,
                 SharingMode sharing = SHARE_IOBUF_MANAGED) const;
#endif

  // This is obvious, except on Cuda, where it isn't.
  T read(size_t offset) const {
    DCHECK_LT(offset, this->size());
    return this->data()[offset];
  }

  void read(size_t offset, T* dest, size_t n) const {
    DCHECK_LE(offset + n, this->size());
    memcpy(dest, this->data() + offset, n * sizeof(T));
  }

  void write(size_t offset, T value) {
    DCHECK_LT(offset, this->size());
    this->data()[offset] = value;
  }

  void write(size_t offset, const T* src, size_t n) {
    DCHECK_LE(offset + n, this->size());
    memcpy(this->data() + offset, src, n * sizeof(T));
  }

////////////////////////////////////////////////////////////////////////////////
#endif // !NO_FOLLY
////////////////////////////////////////////////////////////////////////////////

  bool isUnique() const { return isUnique(this->t_); }
  static bool isUnique(const THType* th);

 private:
  template <class U> friend class Tensor;
  template <class U> friend class CudaTensor;

#ifndef NO_FOLLY
  void setFromIOBuf(folly::IOBuf&& iob, SharingMode sharing);
#endif
};

/**
 * Wrap a THAllocator-like object with a C++ interface into THAllocator.
 */
template <class A>
class THAllocatorWrapper {
 public:
  static THAllocator thAllocator;
 private:
  static void* malloc(void* ctx, long size) {
    return static_cast<A*>(ctx)->malloc(size);
  }
  static void* realloc(void* ctx, void* ptr, long size) {
    return static_cast<A*>(ctx)->realloc(ptr, size);
  }
  static void free(void* ctx, void* ptr) {
    return static_cast<A*>(ctx)->free(ptr);
  }
};

}  // namespaces

#include <thpp/Storage-inl.h>

#endif /* THPP_STORAGE_H_ */
