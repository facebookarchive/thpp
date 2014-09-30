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

namespace detail {
template <class T> class StorageHolder;
}  // namespace

template <class T>
class Storage {
  friend class detail::StorageHolder<T>;
  friend class Tensor<T>;
  typedef detail::StorageOps<T> Ops;
 public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef typename Ops::type THType;

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

  T* data() { return t_ ? t_->data : nullptr; }
  const T* data() const { return t_ ? t_->data : nullptr; }
  iterator begin() { return data(); }
  const_iterator begin() const { return data(); }
  const_iterator cbegin() const { return data(); }
  iterator end() { return t_ ? (t_->data + t_->size) : nullptr; }
  const_iterator end() const { return t_ ? (t_->data + t_->size) : nullptr; }
  const_iterator cend() const { return end(); }

  T& operator[](size_t index) { return data()[index]; }
  const T& operator[](size_t index) const { return data()[index]; }
  T& at(size_t index) { check(index); return operator[](index); }
  const T& at(size_t index) const { check(index); return operator[](index); }

  size_t size() const { return t_ ? t_->size : 0; }

  void resizeUninitialized(size_t n);
  void resize(size_t n, T value = 0);

  template <class It> void assign(It begin, It end);
  void assign(size_t n, T value);

  bool unique() const { return !t_ || t_->refcount == 1; }

  // Create a IOBuf that wraps this storage object. The storage object
  // won't get deleted until all references to the IOBuf are gone.
  folly::IOBuf getIOBuf();

  // Serialize to Thrift.
  void serialize(ThriftStorage& out,
                 ThriftTensorEndianness endianness =
                     ThriftTensorEndianness::NATIVE,
                 bool mayShare = true) const;

  static constexpr const char* kLuaTypeName = Ops::kLuaTypeName;

  // Get a pointer to the underlying TH object; *this releases ownership
  // of that object.
  THType* moveAsTH();
 private:
  template <class U> friend class Tensor;

  void setFromIOBuf(folly::IOBuf&& iob);
  void up();
  void down();
  void check(size_t index) const;

  THType* th() { return t_; }
  const THType* th() const { return t_; }

  // NOTE: May not have any other fields, as we reinterpret_cast
  // liberally between Ops::type* and Storage*
  THType* t_;
};

template <class T>
constexpr const char* Storage<T>::kLuaTypeName;

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

#include <thpp/StorageSerialization-inl.h>
#include <thpp/Storage-inl.h>

#endif /* THPP_STORAGE_H_ */
