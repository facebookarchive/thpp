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
    this->read(0, cpuStorage.data(), cpuStorage.size());
  }
}

template <class T>
void CudaStorage<T>::read(size_t offset, T* dest, size_t n) const {
  DCHECK_LE(offset + n, this->size());
  cuda::check(cudaMemcpy(dest, this->data() + offset, n * sizeof(T),
                         cudaMemcpyDeviceToHost));
}

template <class T>
T CudaStorage<T>::read(size_t offset) const {
  T result;
  this->read(offset, &result, 1);
  return result;
}

template <class T>
void CudaStorage<T>::write(size_t offset, const T* src, size_t n) {
  DCHECK_LE(offset + n, this->size());
  cuda::check(cudaMemcpy(this->data() + offset, src, n * sizeof(T),
                         cudaMemcpyHostToDevice));
}

template <class T>
void CudaStorage<T>::write(size_t offset, T value) {
  this->write(offset, &value, 1);
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
CudaStorage<T>::CudaStorage(const ThriftStorage& thriftStorage,
                            bool mayShare)
  : CudaStorage(Storage<T>(thriftStorage, true)) {
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
    this->write(0, cpuStorage.data(), this->size());
  }
  return cpuStorage;
}

}  // namespaces
