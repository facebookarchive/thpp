/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_TENSOR_H_
#error This file may only be included from thpp/cuda/Tensor.h
#endif

#include <folly/Exception.h>

namespace thpp {

template <class T>
CudaTensor<T>::CudaTensor() : Base(Ops::_new()) { }

template <class T>
CudaTensor<T>::CudaTensor(TensorInvalid) : Base(nullptr) { }

template <class T>
CudaTensor<T>::CudaTensor(StorageType storage, offset_type storageOffset,
                          LongStorage sizes, LongStorage strides)
  : CudaTensor() {
  Ops::_setStorage(this->t_, storage.th(), storageOffset, sizes.th(),
                   strides.th());
}

template <class T>
CudaTensor<T>::CudaTensor(LongStorage sizes, LongStorage strides)
  : CudaTensor() {
  Ops::_setStorage(this->t_, nullptr, 0, sizes.th(), strides.th());
}

template <class T>
CudaTensor<T>::CudaTensor(LongRange sizes, LongRange strides)
  : CudaTensor(LongStorage::wrap(detail::makeMutable(sizes)),
               LongStorage::wrap(detail::makeMutable(strides))) { }

template <class T>
CudaTensor<T>::CudaTensor(std::initializer_list<size_type> sizes,
                          std::initializer_list<size_type> strides)
  : CudaTensor(LongStorage(sizes.begin(), sizes.end()),
           LongStorage(strides.begin(), strides.end())) { }

template <class T>
CudaTensor<T>::CudaTensor(const std::vector<size_type>& sizes,
                          const std::vector<size_type>& strides)
  : CudaTensor(LongStorage(sizes.begin(), sizes.end()),
               LongStorage(strides.begin(), strides.end())) { }

template <class T>
CudaTensor<T>::CudaTensor(const Tensor<T>& cpuTensor)
  : CudaTensor(cpuTensor.sizes()) {
  copy(cpuTensor);
}

// The CPU tensor is temporary, it may always share memory with Thrift
template <class T>
CudaTensor<T>::CudaTensor(const ThriftTensor& thriftTensor, bool mayShare)
  : CudaTensor(Tensor<T>(thriftTensor, true)) {
}

template <class T>
CudaTensor<T>::~CudaTensor() {
  this->destroy();
}

template <class T>
CudaTensor<T>::CudaTensor(THType* other, TensorMustAlias) noexcept
  : Base(other) {
  Ops::_retain(this->t_);
}

template <class T>
CudaTensor<T>::CudaTensor(CudaTensor&& other) noexcept : Base(other.t_) {
  other.t_ = nullptr;
}

template <class T>
CudaTensor<T>::CudaTensor(CudaTensor&& other, unsigned cloneMode) {
  if ((other.mode() & cloneMode) != cloneMode) {
    this->t_ = Ops::_newClone(other.mut());
    other.destroy();
  } else {
    this->t_ = other.t_;
    other.t_ = nullptr;
  }
}

template <class T>
CudaTensor<T>::CudaTensor(const THType* other, unsigned cloneMode) {
  if ((cloneMode & Base::UNIQUE) ||
      ((cloneMode & Base::CONTIGUOUS) && !Base::isContiguous(other))) {
    this->t_ = Ops::_newClone(Base::mut(other));
  } else {
    this->t_ = Ops::_newWithTensor(Base::mut(other));
  }
}

template <class T>
CudaTensor<T>::CudaTensor(THType*&& other) : Base(std::move(other)) { }

template <class T>
CudaTensor<T>::CudaTensor(const CudaTensor& other, unsigned cloneMode)
  : CudaTensor(other.t_, cloneMode) { }

template <class T>
auto CudaTensor<T>::operator=(CudaTensor&& other) noexcept -> CudaTensor& {
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
auto CudaTensor<T>::operator=(const CudaTensor& other) -> CudaTensor& {
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
auto CudaTensor<T>::operator=(THType*&& other) -> CudaTensor& {
  if (other != this->t_) {
    this->destroy();
    this->t_ = std::move(other);
  }
  return *this;
}

template <class T>
T CudaTensor<T>::at(std::initializer_list<offset_type> indices) const {
  auto offset = this->storageOffset() + this->offsetOf(std::move(indices));
  typename Base::StorageBuffer buf;
  return this->storageRef(&buf).read(offset);
}

template <class T>
void CudaTensor<T>::copy(const CudaTensor& src) {
  Ops::_copy(this->t_, src.mut());
}

template <class T>
template <class U>
void CudaTensor<T>::copy(const Tensor<U>& src) {
  Ops::_copyFrom(this->t_, src.mut());
}

template <class T>
template <class U>
void CudaTensor<T>::copyTo(Tensor<U>& dest) const {
  Ops::_copyTo(dest.mut(), this->mut());
}

template <class T>
Tensor<T> CudaTensor<T>::toCPU() const {
  Tensor<T> cpuTensor(this->sizes());
  copyTo(cpuTensor);
  return cpuTensor;
}

template <class T>
CudaTensor<T> CudaTensor<T>::toDevice(int device) const {
  int currentDevice = getDevice();
  if (currentDevice == -1 || currentDevice == device) {
    return *this;
  }
  cuda::DeviceGuard guard;
  cuda::setDevice(device);
  CudaTensor<T> result;
  result.resizeAs(*this);
  result.copy(*this);
  return result;
}

#define TENSOR_ARGM_OP(name) \
  template <class T> \
  auto CudaTensor<T>::name(int dim) const \
  -> std::pair<CudaTensor, CudaTensor> { \
    std::pair<CudaTensor, CudaTensor> dest; \
    Ops::_ ## name(dest.first.t_, dest.second.t_, this->mut(), dim); \
    return dest; \
  }
TENSOR_ARGM_OP(min)
TENSOR_ARGM_OP(max)
#undef TENSOR_ARGM_OP

template <class T>
int CudaTensor<T>::getDevice() const {
  return Ops::_getDevice(this->mut());
}

namespace detail {
void cudaTensorSerialize(
    ThriftTensor& out,
    LongRange sizes,
    LongRange strides,
    const void* data,
    ThriftTensorDataType dtype,
    size_t elementSize,
    ThriftTensorEndianness endianness);
}  // namespace detail

template <class T>
void CudaTensor<T>::serialize(ThriftTensor& out,
                              ThriftTensorEndianness endianness,
                              bool mayShare) const {
  toCPU().serialize(out, endianness, true);
}

}  // namespaces
