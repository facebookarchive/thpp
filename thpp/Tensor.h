/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_TENSOR_H_
#define THPP_TENSOR_H_

#include <thpp/Storage.h>
#include <thpp/TensorBase.h>
#include <thpp/detail/Tensor.h>
#include <thpp/if/gen-cpp2/Tensor_types.h>
#include <folly/io/IOBuf.h>

namespace thpp {

struct TensorInvalid {};
struct TensorMustAlias {};

/**
 * A Tensor wraps a pointer to a THTensor, and as such it has reference-counted
 * pointer semantics. Unlike shared_ptr, the reference count is NOT atomic, and
 * so Tensor objects may not be shared between threads.
 *
 * There are two levels of sharing (sigh): Tensor objects may alias each other
 * (they point to the same object underneath), and tensor objects that don't
 * alias each other may share storage.
 *
 * Assignment (operator=) assigns to the pointed-to object. If two tensors
 * alias each other, they will always behave identically (they are the same
 * object) and so will continue to alias each other after one is
 * assigned to.
 *
 * The only way to break aliasing is through destruction.
 *
 * It is recommended that you only use aliasing when interfacing with Lua
 * (when creating a Tensor that wraps an existing THTensor*, aka an existing
 * Lua object, and you want metadata modifications in your Tensor to reflect
 * in the underlying Lua object).
 *
 * Tensors may also share memory with other tensors. Operations that
 * manipulate metadata (select, transpose, etc) will make source and
 * destination tensors share memory. To ensure you have a unique copy, use
 * force(UNIQUE) (or set UNIQUE in the optional cloneMode argument to the copy
 * and move constructors).
 *
 * After metadata manipulation, the resulting tensor might not be stored
 * in the usual row-major order in memory. If you need a contiguous
 * representation, use force(CONTIGUOUS) (or set CONTIGUOUS in the optional
 * argument to the copy and move constructors). Note that this may break
 * the memory sharing (it will likely create a UNIQUE copy as well).
 */
template <class T> class CudaTensor;
template <class T>
class Tensor : public TensorBase<T, Storage<T>, Tensor<T>> {
  typedef TensorBase<T, Storage<T>, Tensor<T>> Base;
  typedef typename Base::Ops Ops;
  template <class U> friend class Tensor;
  template <class U> friend class CudaTensor;
 public:
  typedef typename Base::StorageType StorageType;
  typedef typename Base::offset_type offset_type;
  typedef typename Base::size_type size_type;
  typedef typename Base::THType THType;

  // Default constructor; construct an empty, zero-dimensional Tensor.
  Tensor();

  explicit Tensor(TensorInvalid);

  Tensor(StorageType storage, offset_type storageOffset,
         LongStorage sizes, LongStorage strides = LongStorage());

  // Constructors from a list of sizes and a list of strides.
  // If specified, the list of strides must have the same size as the
  // list of sizes.
  explicit Tensor(LongStorage sizes, LongStorage strides = LongStorage());
  explicit Tensor(LongRange sizes, LongRange strides = LongRange());
  explicit Tensor(const std::vector<size_type>& sizes,
                  const std::vector<size_type>& strides =
                    std::vector<size_type>());
  explicit Tensor(
      std::initializer_list<size_type> sizes,
      std::initializer_list<size_type> strides =
        std::initializer_list<size_type>());

  // Deserialize from Thrift. Throws if wrong type.
  explicit Tensor(ThriftTensor&& thriftTensor);

  // Destructor
  ~Tensor();

  // Alias other.
  explicit Tensor(THType* other, TensorMustAlias) noexcept;

  // Do not alias other, create separate object (with separate metadata);
  // might still share data with other, unless UNIQUE requested in
  // cloneMode.
  explicit Tensor(const THType* other, unsigned cloneMode = 0);

  // Take ownership of the given THType*. This is the reverse
  // operation of moveAsTH().
  explicit Tensor(THType*&& other);

  // Move/copy constructors. Enforce requested mode.
  /* implicit */ Tensor(Tensor&& other) noexcept;
  /* may throw */ Tensor(Tensor&& other, unsigned cloneMode);
  /* implicit */ Tensor(const Tensor& other, unsigned cloneMode = 0);

  // Move/copy assignment operators. Will share memory with "other".
  Tensor& operator=(Tensor&& other) noexcept;
  Tensor& operator=(const Tensor& other);

  // Take ownership of the given THType*. This is the reverse
  // operation of moveAsTH().
  Tensor& operator=(THType*&& other);

  // Serialize to Thrift. Non-const because the resulting ThriftTensor
  // may share memory with *this, and so changes in the ThriftTensor may
  // affect changes in *this.
  void serialize(ThriftTensor& out,
                 ThriftTensorEndianness endianness =
                    ThriftTensorEndianness::NATIVE,
                 bool mayShare = true);

  // Copy from another tensor
  template <class U>
  void copy(const Tensor<U>& src);

  // Operator to return the first element at the given index
  T& at(offset_type idx) { return at({idx}); }
  const T& at(offset_type idx) const { return at({idx}); }

  T& at(std::initializer_list<offset_type> indices) {
    return this->data()[this->offsetOf(std::move(indices))];
  }

  const T& at(std::initializer_list<offset_type> indices) const {
    return const_cast<Tensor*>(this)->at(std::move(indices));
  }

  // <max, argmax>
  std::pair<Tensor, Tensor<long>> max(int dim) const;

  // <min, argmin>
  std::pair<Tensor, Tensor<long>> min(int dim) const;
};

}  // namespaces

#include <thpp/Tensor-inl.h>

#endif /* THPP_TENSOR_H_ */
