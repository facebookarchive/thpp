/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_STORAGE_H_
#error This file may only be included from thpp/cuda/Storage.h
#endif

namespace thpp {

template <class T>
CudaStorage<T>::CudaStorage() : Base(nullptr) { }

template <class T>
CudaStorage<T>::CudaStorage(THType* t) : Base(t) {
  this->up();
}

template <class T>
CudaStorage<T>::~CudaStorage() {
  this->down();
}

template <class T>
CudaStorage<T>::CudaStorage(CudaStorage&& other) noexcept : Base(other.t_) {
  other.t_ = nullptr;
}

template <class T>
CudaStorage<T>::CudaStorage(const CudaStorage& other) : CudaStorage(other.t_) {
}

template <class T>
CudaStorage<T>::CudaStorage(const Storage<T>& cpuStorage) : CudaStorage() {
  if (cpuStorage.data()) {
    resizeUninitialized(cpuStorage.size());
    cuda::check(cudaMemcpy(this->data(), cpuStorage.data(),
                           cpuStorage.size() * sizeof(T),
                           cudaMemcpyHostToDevice));
  }
}

template <class T>
CudaStorage<T>& CudaStorage<T>::operator=(CudaStorage&& other) {
  if (&other != this) {
    this->down();
    this->t_ = other.t_;
    other.t_ = nullptr;
  }
  return *this;
}

template <class T>
CudaStorage<T>& CudaStorage<T>::operator=(const CudaStorage& other) {
  if (&other != this) {
    this->down();
    this->t_ = other.t_;
    this->up();
  }
  return *this;
}

template <class T>
CudaStorage<T>::CudaStorage(ThriftStorage&& thriftStorage)
  : CudaStorage(Storage<T>(std::move(thriftStorage))) {
}

namespace detail {
void cudaStorageSerialize(ThriftStorage& out,
                          const void* src, size_t size,
                          ThriftTensorDataType dtype,
                          ThriftTensorEndianness endianness);
}  // namespace detail

template <class T>
void CudaStorage<T>::serialize(ThriftStorage& out,
                               ThriftTensorEndianness endianness,
                               bool mayShare) const {
  toCPU().serialize(out, endianness, true);
}

template <class T>
Storage<T> CudaStorage<T>::toCPU() const {
  Storage<T> cpuStorage;
  if (this->data()) {
    cpuStorage.resizeUninitialized(this->size());
    cuda::check(cudaMemcpy(cpuStorage.data(), this->data(),
                           this->size() * sizeof(T),
                           cudaMemcpyDeviceToHost));
  }
  return cpuStorage;
}

}  // namespaces
