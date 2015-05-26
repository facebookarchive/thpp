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

#include <thpp/StorageBase.h>
#include <thpp/detail/Storage.h>
#include <thpp/if/gen-cpp2/Tensor_types.h>
#include <folly/Malloc.h>
#include <folly/Range.h>
#include <folly/io/IOBuf.h>

namespace thpp {

using folly::Range;

/**
 * Wrapper around TH's Storage type, which is a length-aware,
 * reference-counted, heap-allocated array.
 */
template <class T> class Tensor;
template <class T> class CudaTensor;

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

  explicit Storage(Range<const T*> range)
    : Storage(range.begin(), range.end()) { }

  explicit Storage(THType* t);

  // Create a Storage object containing the data from an IOBuf.
  explicit Storage(folly::IOBuf&& iob);
  explicit Storage(folly::IOBuf& iob) : Storage(*iob.clone()) { }

  // Deserialize from Thrift. Throws if wrong type.
  explicit Storage(ThriftStorage&& thriftStorage);

  // Takes ownership of a range allocated with malloc() (NOT new or new[]!)
  static Storage takeOwnership(Range<T*> data);

  // Wrap a range of memory. The range must stay allocated until all Storage
  // objects that refer to it are gone.
  static Storage wrap(Range<T*> data);

  // Use a custom allocator. The allocator is managed by the caller.
  static Storage withAllocator(THAllocator* allocator,
                               void* allocatorContext);

  // Wrap a range of memory and use a custom allocator for reallocations.
  // You probably don't need this.
  static Storage wrapWithAllocator(Range<T*> data,
                                   THAllocator* allocator,
                                   void* allocatorContext);

  ~Storage();

  Storage(Storage&& other) noexcept;
  Storage(const Storage& other);
  Storage& operator=(Storage&& other);
  Storage& operator=(const Storage& other);

  void resize(size_t n, T value = 0);

  template <class It> void assign(It begin, It end);
  void assign(size_t n, T value);

  // Create a IOBuf that wraps this storage object. The storage object
  // won't get deleted until all references to the IOBuf are gone.
  folly::IOBuf getIOBuf();

  // Serialize to Thrift.
  void serialize(ThriftStorage& out,
                 ThriftTensorEndianness endianness =
                     ThriftTensorEndianness::NATIVE,
                 bool mayShare = true) const;

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

 private:
  template <class U> friend class Tensor;
  template <class U> friend class CudaTensor;

  void setFromIOBuf(folly::IOBuf&& iob);
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
