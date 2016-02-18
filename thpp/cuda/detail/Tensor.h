/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_DETAIL_TENSOR_H_
#define THPP_CUDA_DETAIL_TENSOR_H_

#include <THC.h>
#include <thpp/cuda/detail/Storage.h>

namespace thpp {

template <class T> class CudaTensor;

typedef CudaTensor<float> CudaFloatTensor;

namespace detail {

template <class T> struct TensorOps;

template <> struct TensorOps<CudaTensor<float>> {
  typedef float value_type;
  typedef float accurate_type;
  typedef THCudaTensor type;
  typedef CudaTensor<float> ArgTensorType;

  static float* _data(THCudaTensor* t) {
    return THCudaTensor_data(getTHCState(), t);
  }
  static THCudaStorage* _storage(THCudaTensor* t) {
    return THCudaTensor_storage(getTHCState(), t);
  }
  static long _storageOffset(THCudaTensor* t) {
    return THCudaTensor_storageOffset(getTHCState(), t);
  }
  static THCudaTensor* _new() {
    return THCudaTensor_new(getTHCState());
  }
  static THCudaTensor* _newWithTensor(THCudaTensor* other) {
    return THCudaTensor_newWithTensor(getTHCState(), other);
  }
  static THCudaTensor* _newWithStorage(THCudaStorage* storage,
                                       long storageOffset,
                                       THLongStorage* size,
                                       THLongStorage* stride) {
    return THCudaTensor_newWithStorage(
        getTHCState(), storage, storageOffset, size, stride);
  }
  static THCudaTensor* _newClone(THCudaTensor* self) {
    return THCudaTensor_newClone(getTHCState(), self);
  }
  static THCudaTensor* _newContiguous(THCudaTensor* self) {
    return THCudaTensor_newContiguous(getTHCState(), self);
  }
  static void _resize(THCudaTensor* self, THLongStorage* size,
                      THLongStorage* stride) {
    THCudaTensor_resize(getTHCState(), self, size, stride);
  }
  static void _resizeAs(THCudaTensor* self, THCudaTensor* src) {
    THCudaTensor_resizeAs(getTHCState(), self, src);
  }
  static void _set(THCudaTensor* self, THCudaTensor* src) {
    THCudaTensor_set(getTHCState(), self, src);
  }
  static void _setStorage(THCudaTensor* self, THCudaStorage* storage,
                          long offset, THLongStorage* size,
                          THLongStorage* stride) {
    THCudaTensor_setStorage(getTHCState(), self, storage, offset, size, stride);
  }
  static void _setStorage1d(THCudaTensor* self, THCudaStorage* storage,
                            long offset, long size0, long stride0) {
    THCudaTensor_setStorage1d(
        getTHCState(), self, storage, offset, size0, stride0);
  }
  static void _narrow(THCudaTensor* self, THCudaTensor* src, int dim,
                      long firstIndex, long size) {
    THCudaTensor_narrow(getTHCState(), self, src, dim, firstIndex, size);
  }
  static void _select(THCudaTensor* self, THCudaTensor* src, int dim,
                      long index) {
    THCudaTensor_select(getTHCState(), self, src, dim, index);
  }
  static void _transpose(THCudaTensor* self, THCudaTensor* src, int dim1,
                         int dim2) {
    THCudaTensor_transpose(getTHCState(), self, src, dim1, dim2);
  }
  static void _squeeze(THCudaTensor* self, THCudaTensor* src) {
    THCudaTensor_squeeze(getTHCState(), self, src);
  }
  static void _squeeze1d(THCudaTensor* self, THCudaTensor* src, int dim) {
    THCudaTensor_squeeze1d(getTHCState(), self, src, dim);
  }
  static int _isContiguous(const THCudaTensor* self) {
    return THCudaTensor_isContiguous(getTHCState(), self);
  }
  static long _nElement(const THCudaTensor* self) {
    return THCudaTensor_nElement(getTHCState(), self);
  }
  static void _retain(THCudaTensor* self) {
    return THCudaTensor_retain(getTHCState(), self);
  }
  static void _free(THCudaTensor* self) {
    return THCudaTensor_free(getTHCState(), self);
  }

  static void _copy(THCudaTensor* self, THCudaTensor* src) {
    THCudaTensor_copy(getTHCState(), self, src);
  }

  // THCudaTensorCopy.h
  template <class T>
  static void _copyFrom(THCudaTensor* self, T* src);
  template <class T>
  static void _copyTo(T* dest, THCudaTensor* src);

  // THCudaTensorMath.h
  static void _fill(THCudaTensor* r, float value) {
    THCudaTensor_fill(getTHCState(), r, value);
  }
  static void _zero(THCudaTensor* r) {
    THCudaTensor_zero(getTHCState(), r);
  }
  // Two overloads each: with data on device (as THCudaTensor) or on host
  // (as THByteTensor)
  static void _maskedFill(THCudaTensor* tensor, THByteTensor* mask,
                          float value) {
    THCudaTensor_maskedFillByte(getTHCState(), tensor, mask, value);
  }
  static void _maskedFill(THCudaTensor* tensor, THCudaTensor* mask,
                          float value) {
    THCudaTensor_maskedFill(getTHCState(), tensor, mask, value);
  }
  static void _maskedCopy(THCudaTensor* tensor, THByteTensor* mask,
                          THCudaTensor* src) {
    THCudaTensor_maskedCopyByte(getTHCState(), tensor, mask, src);
  }
  static void _maskedCopy(THCudaTensor* tensor, THCudaTensor* mask,
                          THCudaTensor* src) {
    THCudaTensor_maskedCopy(getTHCState(), tensor, mask, src);
  }
  static void _maskedSelect(THCudaTensor* tensor, THCudaTensor* src,
                            THByteTensor* mask) {
    THCudaTensor_maskedSelectByte(getTHCState(), tensor, src, mask);
  }
  static void _maskedSelect(THCudaTensor* tensor, THCudaTensor* src,
                            THCudaTensor* mask) {
    THCudaTensor_maskedSelect(getTHCState(), tensor, src, mask);
  }
  static void _indexSelect(THCudaTensor* tensor, THCudaTensor* src, int dim,
                           THLongTensor* index) {
    THCudaTensor_indexSelect_long(getTHCState(), tensor, src, dim, index);
  }
  static void _indexCopy(THCudaTensor* tensor, int dim, THLongTensor* index,
                         THCudaTensor* src) {
    THCudaTensor_indexCopy_long(getTHCState(), tensor, dim, index, src);
  }
  static void _indexFill(THCudaTensor* tensor, int dim, THLongTensor* index,
                         float val) {
    THCudaTensor_indexFill_long(getTHCState(), tensor, dim, index, val);
  }
  static float _dot(THCudaTensor* t, THCudaTensor* src) {
    return THCudaTensor_dot(getTHCState(), t, src);
  }
  static float _minall(THCudaTensor* t) {
    return THCudaTensor_minall(getTHCState(), t);
  }
  static float _maxall(THCudaTensor* t) {
    return THCudaTensor_maxall(getTHCState(), t);
  }
  static float _sumall(THCudaTensor* t) {
    return THCudaTensor_sumall(getTHCState(), t);
  }
  static float _prodall(THCudaTensor* t) {
    return THCudaTensor_prodall(getTHCState(), t);
  }
  static void _add(THCudaTensor* r, THCudaTensor* t, float value) {
    return THCudaTensor_add(getTHCState(), r, t, value);
  }
  static void _mul(THCudaTensor* r, THCudaTensor* t, float value) {
    return THCudaTensor_mul(getTHCState(), r, t, value);
  }
  static void _div(THCudaTensor* r, THCudaTensor* t, float value) {
    return THCudaTensor_div(getTHCState(), r, t, value);
  }
  static void _cadd(THCudaTensor* r, THCudaTensor* t, float value,
                    THCudaTensor* src) {
    return THCudaTensor_cadd(getTHCState(), r, t, value, src);
  }
  static void _cmul(THCudaTensor* r, THCudaTensor* t, THCudaTensor* src) {
    return THCudaTensor_cmul(getTHCState(), r, t, src);
  }
  static void _cdiv(THCudaTensor* r, THCudaTensor* t, THCudaTensor* src) {
    return THCudaTensor_cdiv(getTHCState(), r, t, src);
  }
  static void _addcmul(THCudaTensor* r, THCudaTensor* t, float value,
                       THCudaTensor* src1, THCudaTensor* src2) {
    return THCudaTensor_addcmul(getTHCState(), r, t, value, src1, src2);
  }
  static void _addcdiv(THCudaTensor* r, THCudaTensor* t, float value,
                       THCudaTensor* src1, THCudaTensor* src2) {
    return THCudaTensor_addcdiv(getTHCState(), r, t, value, src1, src2);
  }
  static void _addmv(THCudaTensor* r, float beta, THCudaTensor* t, float alpha,
                     THCudaTensor* mat, THCudaTensor* vec) {
    return THCudaTensor_addmv(getTHCState(), r, beta, t, alpha, mat, vec);
  }
  static void _addmm(THCudaTensor* r, float beta, THCudaTensor* t, float alpha,
                     THCudaTensor* m1, THCudaTensor* m2) {
    return THCudaTensor_addmm(getTHCState(), r, beta, t, alpha, m1, m2);
  }
  static void _addr(THCudaTensor* r, float beta, THCudaTensor* t, float alpha,
                    THCudaTensor* vec1, THCudaTensor* vec2) {
    return THCudaTensor_addr(getTHCState(), r, beta, t, alpha, vec1, vec2);
  }
  static void _max(THCudaTensor* values, THCudaTensor* indices,
                   THCudaTensor* t, int dim) {
    return THCudaTensor_max(getTHCState(), values, indices, t, dim);
  }
  static void _min(THCudaTensor* values, THCudaTensor* indices,
                   THCudaTensor* t, int dim) {
    return THCudaTensor_min(getTHCState(), values, indices, t, dim);
  }
  static void _sum(THCudaTensor* r, THCudaTensor* t, int dim) {
    return THCudaTensor_sum(getTHCState(), r, t, dim);
  }
  static void _prod(THCudaTensor* r, THCudaTensor* t, int dim) {
    return THCudaTensor_prod(getTHCState(), r, t, dim);
  }
  static void _cumsum(THCudaTensor* r, THCudaTensor* t, int dim) {
    return THCudaTensor_cumsum(getTHCState(), r, t, dim);
  }
  static void _cumprod(THCudaTensor* r, THCudaTensor* t, int dim) {
    return THCudaTensor_cumprod(getTHCState(), r, t, dim);
  }
  static void _sign(THCudaTensor* r, THCudaTensor* t) {
    return THCudaTensor_sign(getTHCState(), r, t);
  }

  // CUDA-specific
  static int _getDevice(THCudaTensor* self) {
    return THCudaTensor_getDevice(getTHCState(), self);
  }
  static constexpr const char* kLuaTypeName = "torch.CudaTensor";
};

#define S(TYPE) \
  template <> inline void TensorOps<CudaTensor<float>>::_copyFrom< \
      TH##TYPE##Tensor>(THCudaTensor* self, TH##TYPE##Tensor* src) { \
        return THCudaTensor_copy##TYPE(getTHCState(), self, src); \
      } \
  template <> inline void TensorOps<CudaTensor<float>>::_copyTo< \
      TH##TYPE##Tensor>(TH##TYPE##Tensor* dest, THCudaTensor* src) { \
        return TH##TYPE##Tensor_copyCudaFloat(getTHCState(), dest, src); \
      }

S(Byte)
S(Char)
S(Short)
S(Int)
S(Long)
S(Float)
S(Double)

#undef S

}  // namespace detail

}  // namespaces

#endif /* THPP_CUDA_DETAIL_TENSOR_H_ */
