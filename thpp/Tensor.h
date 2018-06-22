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

#ifdef THPP_COMPAT_TENSOR_H_
#error "thpp-compatibility/ is a wrapper of legacy thpp with ATen's updated TH. You should NOT include thpp/ and thpp-compatibility/ in the same binary"
#endif


#include <thpp/ForwardDeclarations.h>
#include <thpp/Storage.h>
#include <thpp/TensorBase.h>
#include <thpp/detail/Tensor.h>
#ifndef NO_THRIFT
#include <thpp/if/gen-cpp2/Tensor_types.h>
#endif
#ifndef NO_FOLLY
#include <folly/io/IOBuf.h>
#endif

namespace thpp {

/**
 * A Tensor wraps a pointer to a THTensor, and as such it has reference-counted
 * pointer semantics.
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
  friend class TensorPtr<Tensor>;

 public:
  typedef typename Base::StorageType StorageType;
  typedef typename Base::offset_type offset_type;
  typedef typename Base::size_type size_type;
  typedef typename Base::THType THType;

  // Default constructor; construct an empty, zero-dimensional Tensor.
  Tensor();

  Tensor(StorageType storage, offset_type storageOffset,
         LongStorage sizes, LongStorage strides = LongStorage());

#ifndef NO_FOLLY
  Tensor(StorageType storage, offset_type storageOffset,
         LongRange sizes, LongRange strides = LongRange());
#endif

  Tensor(StorageType storage, offset_type storageOffset,
         std::initializer_list<size_type> sizes,
         std::initializer_list<size_type> strides =
           std::initializer_list<size_type>());

  // Constructors from a list of sizes and a list of strides.
  // If specified, the list of strides must have the same size as the
  // list of sizes.
  explicit Tensor(LongStorage sizes, LongStorage strides = LongStorage());
#ifndef NO_FOLLY
  explicit Tensor(LongRange sizes, LongRange strides = LongRange());
#endif
  explicit Tensor(const std::vector<size_type>& sizes,
                  const std::vector<size_type>& strides =
                    std::vector<size_type>());
  explicit Tensor(
      std::initializer_list<size_type> sizes,
      std::initializer_list<size_type> strides =
        std::initializer_list<size_type>());

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
  // Deserialize from Thrift. Throws if wrong type.
  explicit Tensor(const ThriftTensor& thriftTensor,
                  SharingMode sharing = SHARE_IOBUF_MANAGED);
#endif

  // Do not alias other, create separate object (with separate metadata);
  // might still share data with other, unless UNIQUE requested in
  // cloneMode.
  explicit Tensor(const THType* other, unsigned cloneMode = 0);

  // Move/copy constructors. Enforce requested mode.
  /* implicit */ Tensor(const Tensor& other, unsigned cloneMode = 0);
  /* implicit */ /* may throw */ Tensor(Tensor&& other, unsigned cloneMode = 0);

  // Move/copy assignment operators. Will share memory with "other".
  Tensor& operator=(const Tensor& other);
  /* noexcept override */ Tensor& operator=(Tensor&& other);

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
  // Serialize to Thrift. Note that, if sharing is not SHARE_NONE, the
  // resulting ThriftTensor may share memory with *this, so changes in out.data
  // may be reflected in *this.
  void serialize(ThriftTensor& out,
                 ThriftTensorEndianness endianness =
                    ThriftTensorEndianness::NATIVE,
                 SharingMode sharing = SHARE_IOBUF_MANAGED) const;
#endif

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

 private:
  Tensor(detail::SetTH, THType* t, bool incRef);

#if !defined(NO_THRIFT) && !defined(NO_FOLLY)
  static THType* deserializeTH(const ThriftTensor& thriftTensor,
                               SharingMode sharing);
#endif
};

template <class D, class S>
void copyTensor(Tensor<D>& dest, const Tensor<S>& src) {
  dest.copy(src);
}

}  // namespaces

#include <thpp/Tensor-inl.h>

#endif /* THPP_TENSOR_H_ */
