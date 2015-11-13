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
CudaTensor<T>::CudaTensor(const ThriftTensor& thriftTensor, SharingMode sharing)
  : CudaTensor(Tensor<T>(thriftTensor, SHARE_ALL)) {
}

template <class T>
CudaTensor<T>::CudaTensor(detail::SetTH, THType* t, bool incRef)
  : Base(t) {
  DCHECK(t);
  if (incRef) {
    Ops::_retain(this->t_);
  }
}

template <class T>
CudaTensor<T>::CudaTensor(const THType* other, unsigned cloneMode)
  : Base(Base::cloneTH(other, cloneMode)) { }

template <class T>
CudaTensor<T>::CudaTensor(const CudaTensor& other, unsigned cloneMode)
  : CudaTensor(other.t_, cloneMode) { }

template <class T>
CudaTensor<T>::CudaTensor(CudaTensor&& other, unsigned cloneMode)
  : CudaTensor(other, cloneMode) {
  other.clear();
}

template <class T>
auto CudaTensor<T>::operator=(const CudaTensor& other) -> CudaTensor& {
  if (&other != this) {
    Ops::_set(this->t_, other.mut());
  }
  return *this;
}

template <class T>
auto CudaTensor<T>::operator=(CudaTensor&& other) -> CudaTensor& {
  if (&other != this) {
    *this = other;
    other.clear();
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
typename Tensor<T>::Ptr CudaTensor<T>::toCPU() const {
  auto cpuTensor = Tensor<T>::makePtr(this->sizes());
  copyTo(*cpuTensor);
  return cpuTensor;
}

template <class T>
auto CudaTensor<T>::toDevice(int device) const -> Ptr {
  int currentDevice = getDevice();
  if (currentDevice == -1 || currentDevice == device) {
    return this->copyPtr();
  }
  cuda::DeviceGuard guard;
  cuda::setDevice(device);
  auto result = CudaTensor<T>::makePtr();
  result->resizeAs(*this);
  result->copy(*this);
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
                              SharingMode sharing) const {
  toCPU()->serialize(out, endianness, SHARE_ALL);
}

}  // namespaces
