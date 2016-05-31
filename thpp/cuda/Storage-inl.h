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
                            SharingMode sharing)
  : CudaStorage(Storage<T>(thriftStorage, SHARE_ALL)) {
}

template <class T>
CudaStorage<T>::CudaStorage(folly::IOBuf&& iob,
                            SharingMode sharing,
                            bool resizable)
  : Base(nullptr) {
  setFromIOBuf(std::move(iob), sharing, resizable);
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

namespace detail {

class CudaIOBufAllocator {
 public:
  explicit CudaIOBufAllocator(folly::IOBuf&& iob);

  cudaError_t malloc(THCState* state, void** ptr, long size);
  cudaError_t realloc(THCState* state, void** ptr,
                      long oldSize, long newSize);
  cudaError_t free(THCState* state, void* ptr);

 private:
  folly::IOBuf iob_;
};

}  // namespace detail

template <class T>
void CudaStorage<T>::setFromIOBuf(folly::IOBuf&& iob, SharingMode sharing,
                                  bool resizable) {
  if (iob.isChained()) {
    throw std::invalid_argument("IOBuf may not be chained");
  }
  size_t len = iob.length();
  if (len % sizeof(T) != 0) {
    throw std::invalid_argument("IOBuf size must be multiple of data size");
  }
  len /= sizeof(T);

  switch (sharing) {
  case SHARE_NONE:
    throw std::invalid_argument("SHARE_NONE not supported");
  case SHARE_IOBUF_MANAGED:
    if (!iob.isManagedOne()) {
      throw std::invalid_argument("SHARE_IOBUF_MANAGED requires managed IOBuf");
    }
    break;
  case SHARE_ALL:
    break;
  }

  if (resizable) {
    throw std::invalid_argument("NYI: Resizable IOBuf CUDA storage");
  }

  // Ensure properly aligned
  if ((reinterpret_cast<uintptr_t>(iob.data()) % alignof(T)) != 0) {
    throw std::invalid_argument("IOBuf is not properly aligned");
  }

  T* p = reinterpret_cast<T*>(iob.writableData());

  cudaPointerAttributes attr;
  cuda::check(cudaPointerGetAttributes(&attr, p));
  if (attr.memoryType != cudaMemoryTypeDevice) {
    throw std::invalid_argument("IOBuf does not point to CUDA memory");
  }

  this->t_ = Ops::_newWithDataAndAllocator(
      p, len,
      &THCAllocatorWrapper<detail::CudaIOBufAllocator>::thcAllocator,
      new detail::CudaIOBufAllocator(std::move(iob)));
  Ops::_clearFlag(this->t_, TH_STORAGE_RESIZABLE);
}

template <class A>
THCAllocator THCAllocatorWrapper<A>::thcAllocator = {
  &THCAllocatorWrapper<A>::malloc,
  &THCAllocatorWrapper<A>::realloc,
  &THCAllocatorWrapper<A>::free,
};

}  // namespaces
