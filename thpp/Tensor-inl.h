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
#error This file may only be included from thpp/Tensor.h
#endif

namespace thpp {

namespace detail {

void serialize(
    ThriftTensor& out,
    LongRange sizes,
    LongRange strides,
    folly::IOBuf&& data,
    ThriftTensorDataType dtype,
    size_t elementSize,
    ThriftTensorEndianness endianness,
    bool mayShare);

template <class ThriftObj>
folly::IOBuf deserialize(const ThriftObj& in,
                         ThriftTensorDataType dtype);
}  // namespace detail

template <class T>
Tensor<T>::Tensor() : Base(Ops::_new()) { }

template <class T>
Tensor<T>::Tensor(TensorInvalid) : Base(nullptr) { }

template <class T>
Tensor<T>::Tensor(StorageType storage, offset_type storageOffset,
                  LongStorage sizes, LongStorage strides) : Tensor() {
  Ops::_setStorage(this->t_, storage.th(), storageOffset, sizes.th(),
                   strides.th());
}

template <class T>
Tensor<T>::Tensor(StorageType storage, offset_type storageOffset,
                  LongRange sizes, LongRange strides)
  : Tensor(std::move(storage), storageOffset,
           LongStorage::wrap(detail::makeMutable(sizes)),
           LongStorage::wrap(detail::makeMutable(strides))) { }

template <class T>
Tensor<T>::Tensor(StorageType storage, offset_type storageOffset,
                  std::initializer_list<size_type> sizes,
                  std::initializer_list<size_type> strides)
  : Tensor(std::move(storage), storageOffset,
           LongStorage(sizes.begin(), sizes.end()),
           LongStorage(strides.begin(), strides.end())) { }


template <class T>
Tensor<T>::Tensor(LongStorage sizes, LongStorage strides) : Tensor() {
  Ops::_setStorage(this->t_, nullptr, 0, sizes.th(), strides.th());
}

template <class T>
Tensor<T>::Tensor(LongRange sizes, LongRange strides)
  : Tensor(LongStorage::wrap(detail::makeMutable(sizes)),
           LongStorage::wrap(detail::makeMutable(strides))) { }

template <class T>
Tensor<T>::Tensor(std::initializer_list<size_type> sizes,
                  std::initializer_list<size_type> strides)
  : Tensor(LongStorage(sizes.begin(), sizes.end()),
           LongStorage(strides.begin(), strides.end())) { }

template <class T>
Tensor<T>::Tensor(const std::vector<size_type>& sizes,
                  const std::vector<size_type>& strides)
    : Tensor(LongStorage(sizes.begin(), sizes.end()),
             LongStorage(strides.begin(), strides.end())) { }

template <class T>
Tensor<T>::Tensor(const ThriftTensor& thriftTensor,
                  bool mayShare) : Base(nullptr) {
  Storage<T> data(detail::deserialize(thriftTensor, detail::dataType<T>()),
                  mayShare);

  LongStorage s(LongStorage::wrap(detail::makeMutable(LongRange(
      thriftTensor.sizes.data(), thriftTensor.sizes.size()))));

  this->t_ = Ops::_newWithStorage(data.th(), 0, s.th(), nullptr);
  DCHECK_EQ(data.size(), this->size());
}

template <class T>
Tensor<T>::~Tensor() {
  this->destroy();
}

template <class T>
Tensor<T>::Tensor(THType* other, TensorMustAlias) noexcept : Base(other) {
  Ops::_retain(this->t_);
}

template <class T>
Tensor<T>::Tensor(Tensor&& other) noexcept : Base(other.t_) {
  other.t_ = nullptr;
}

template <class T>
Tensor<T>::Tensor(Tensor&& other, unsigned cloneMode) {
  if ((other.mode() & cloneMode) != cloneMode) {
    this->t_ = Ops::_newClone(other.mut());
    other.destroy();
  } else {
    this->t_ = other.t_;
    other.t_ = nullptr;
  }
}

template <class T>
Tensor<T>::Tensor(const THType* other, unsigned cloneMode) {
  if ((cloneMode & Base::UNIQUE) ||
      ((cloneMode & Base::CONTIGUOUS) && !Base::isContiguous(other))) {
    this->t_ = Ops::_newClone(Base::mut(other));
  } else {
    this->t_ = Ops::_newWithTensor(Base::mut(other));
  }
}

template <class T>
Tensor<T>::Tensor(THType*&& other) : Base(std::move(other)) { }

template <class T>
Tensor<T>::Tensor(const Tensor& other, unsigned cloneMode)
  : Tensor(other.t_, cloneMode) { }

template <class T>
auto Tensor<T>::operator=(Tensor&& other) noexcept -> Tensor& {
  if (&other != this) {
    if (this->t_) {
      // Careful. If a and b alias each other (a.t_ == b.t_), that assumption
      // must continue to hold if we do a = std::move(c). So the obvious
      // "t_ = other.t_; other.t_ = nullptr;" will not work.
      Ops::_set(this->t_, other.t_);
      other.destroy();
    } else {
      this->t_ = other.t_;
      other.t_ = nullptr;
    }
  }
  return *this;
}

template <class T>
auto Tensor<T>::operator=(const Tensor& other) -> Tensor& {
  if (&other != this) {
    if (this->t_) {
      Ops::_set(this->t_, other.mut());
    } else {
      this->t_ = Ops::_newWithTensor(other.mut());
    }
  }
  return *this;
}

template <class T>
auto Tensor<T>::operator=(THType*&& other) -> Tensor& {
  if (other != this->t_) {
    this->destroy();
    this->t_ = std::move(other);
  }
  return *this;
}

template <class T>
template <class U>
void Tensor<T>::copy(const Tensor<U>& src) {
  Ops::_copyT(this->t_, src.mut());
}

template <class T>
void Tensor<T>::serialize(ThriftTensor& out,
                          ThriftTensorEndianness endianness,
                          bool mayShare) const {
  auto buf = Storage<T>(Ops::_storage(this->mut())).getIOBuf();
  buf.trimStart(Ops::_storageOffset(this->mut()) * sizeof(T));
  detail::serialize(
      out,
      this->sizes(),
      this->strides(),
      std::move(buf),
      detail::dataType<T>(),
      sizeof(T),
      endianness,
      mayShare);
}


// These must be defined here, as LongTensor and ByteTensor must be
// complete types

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::maskedFill(
    const ByteTensor& mask, T value) {
  Ops::_maskedFill(t_, mask.mut(), value);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::maskedCopy(
    const ByteTensor& mask, const TensorBase& src) {
  Ops::_maskedCopy(t_, mask.mut(), src.mut());
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::maskedSelect(
    const ByteTensor& mask) const -> Derived {
  Derived r;
  Ops::_maskedSelect(&r.t_, this->mut(), mask.mut());
  return r;
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::indexSelect(
    int dim, const LongTensor& index) const -> Derived {
  Derived r;
  Ops::_indexSelect(&r.t_, this->mut(), dim, index.mut());
  return r;
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::indexFill(
    int dim, const LongTensor& index, T val) {
  Ops::_indexFill(t_, dim, index.mut(), val);
}

#define TENSOR_ARGM_OP(name) \
  template <class T> \
  auto Tensor<T>::name(int dim) const -> std::pair<Tensor, LongTensor> { \
    std::pair<Tensor, LongTensor> dest; \
    Ops::_ ## name(dest.first.t_, dest.second.t_, this->mut(), dim); \
    return dest; \
  }
TENSOR_ARGM_OP(min)
TENSOR_ARGM_OP(max)
#undef TENSOR_ARGM_OP

}  // namespaces
