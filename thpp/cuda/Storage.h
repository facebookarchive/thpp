/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_STORAGE_H_
#define THPP_CUDA_STORAGE_H_

#include <thpp/Storage.h>
#include <thpp/cuda/detail/Storage.h>
#include <folly/Malloc.h>
#include <folly/Range.h>

namespace thpp {

template <class T> class CudaTensor;

template <class T>
class CudaStorage : public StorageBase<T, CudaStorage<T>> {
  typedef StorageBase<T, CudaStorage<T>> Base;
  typedef typename Base::Ops Ops;
  friend Base;  // Yay C++11
 public:
  typedef typename Base::THType THType;
  CudaStorage();

  explicit CudaStorage(THType* t);

  explicit CudaStorage(const Storage<T>& cpuStorage);

  // Deserialize from Thrift. Throws if wrong type.
  explicit CudaStorage(ThriftStorage&& thriftStorage);

  ~CudaStorage();

  CudaStorage(CudaStorage&& other) noexcept;
  CudaStorage(const CudaStorage& other);
  CudaStorage& operator=(CudaStorage&& other);
  CudaStorage& operator=(const CudaStorage& other);

  // Serialize to Thrift.
  void serialize(ThriftStorage& out,
                 ThriftTensorEndianness endianness =
                     ThriftTensorEndianness::NATIVE,
                 bool mayShare = true) const;

  Storage<T> toCPU() const;

  T read(size_t offset) const;
  void read(size_t offset, T* dest, size_t n) const;
  void write(size_t offset, T value);
  void write(size_t offset, const T* src, size_t n);

 private:
  template <class U> friend class CudaTensor;
};

}  // namespaces

#include <thpp/cuda/Storage-inl.h>

#endif /* THPP_CUDA_STORAGE_H_ */
