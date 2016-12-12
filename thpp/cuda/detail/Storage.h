/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_DETAIL_STORAGE_H_
#define THPP_CUDA_DETAIL_STORAGE_H_

#include <THCStorage.h>
#include <thpp/cuda/State.h>
#include <thpp/detail/Storage.h>

namespace thpp {

template <class T> class CudaStorage;

namespace detail {

// Only float is currently supported.
template <> struct StorageOps<CudaStorage<float>> {
  typedef THCudaStorage type;

  static THCudaStorage* _newWithSize(long size) {
    return THCudaStorage_newWithSize(getTHCState(), size);
  }
  static THCudaStorage* _newWithData(float* data, long size) {
    return THCudaStorage_newWithData(getTHCState(), data, size);
  }
  static void _setFlag(THCudaStorage* storage, const char flag) {
    THCudaStorage_setFlag(getTHCState(), storage, flag);
  }
  static void _clearFlag(THCudaStorage* storage, const char flag) {
    THCudaStorage_clearFlag(getTHCState(), storage, flag);
  }
  static void _retain(THCudaStorage* storage) {
    THCudaStorage_retain(getTHCState(), storage);
  }
  static void _free(THCudaStorage* storage) {
    THCudaStorage_free(getTHCState(), storage);
  }
  static void _resize(THCudaStorage* storage, long size) {
    THCudaStorage_resize(getTHCState(), storage, size);
  }
  static THCudaStorage* _newWithDataAndAllocator(
      float* data, long size,
      THCDeviceAllocator* allocator, void* allocatorContext) {
    return THCudaStorage_newWithDataAndAllocator(
        getTHCState(), data, size, allocator, allocatorContext);
  }

  static constexpr const char* kLuaTypeName = "torch.CudaStorage";
};

}  // namespace detail

}  // namespaces

#endif /* THPP_CUDA_DETAIL_STORAGE_H_ */
