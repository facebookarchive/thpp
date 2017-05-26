/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "thpp/detail/TensorGeneric.h"
#else

typedef Tensor<real> TH_CONCAT_2(Real, Tensor);

namespace detail {
template <> struct TensorOps<Tensor<real>> {
  typedef real value_type;
  typedef accreal accurate_type;
  typedef THTensor type;
  typedef Tensor<long> ArgTensorType;

  static real* _data(THTensor* t) {
    return THTensor_(data)(t);
  }
  static THStorage* _storage(const THTensor* t) {
    return THTensor_(storage)(t);
  }
  static long _storageOffset(const THTensor* t) {
    return THTensor_(storageOffset)(t);
  }
  static THTensor* _new() {
    return THTensor_(new)();
  }
  static THTensor* _newWithTensor(THTensor* other) {
    return THTensor_(newWithTensor)(other);
  }
  static THTensor* _newWithStorage(THStorage* storage,
                                   long storageOffset,
                                   THLongStorage* size,
                                   THLongStorage* stride) {
    return THTensor_(newWithStorage)(storage, storageOffset, size, stride);
  }
  static THTensor* _newClone(THTensor* self) {
    return THTensor_(newClone)(self);
  }
  static THTensor* _newContiguous(THTensor* self) {
    return THTensor_(newContiguous)(self);
  }
  static void _resize(THTensor* self, THLongStorage* size,
                      THLongStorage* stride) {
    THTensor_(resize)(self, size, stride);
  }
  static void _resizeAs(THTensor* self, THTensor* src) {
    THTensor_(resizeAs)(self, src);
  }
  static void _set(THTensor* self, THTensor* src) {
    THTensor_(set)(self, src);
  }
  static void _setStorage(THTensor* self, THStorage* storage,
                          long offset, THLongStorage* size,
                          THLongStorage* stride) {
    THTensor_(setStorage)(self, storage, offset, size, stride);
  }
  static void _setStorage1d(THTensor* self, THStorage* storage,
                            long offset, long size0, long stride0) {
    THTensor_(setStorage1d)(self, storage, offset, size0, stride0);
  }
  static void _narrow(THTensor* self, THTensor* src, int dim,
                      long firstIndex, long size) {
    THTensor_(narrow)(self, src, dim, firstIndex, size);
  }
  static void _select(THTensor* self, THTensor* src, int dim, long index) {
    THTensor_(select)(self, src, dim, index);
  }
  static void _transpose(THTensor* self, THTensor* src, int dim1, int dim2) {
    THTensor_(transpose)(self, src, dim1, dim2);
  }
  static void _squeeze(THTensor* self, THTensor* src) {
    THTensor_(squeeze)(self, src);
  }
  static void _squeeze1d(THTensor* self, THTensor* src, int dim) {
    THTensor_(squeeze1d)(self, src, dim);
  }
  static int _isContiguous(const THTensor* self) {
    return THTensor_(isContiguous)(self);
  }
  static long _nElement(const THTensor* self) {
    return THTensor_(nElement)(self);
  }
  static void _retain(THTensor* self) {
    return THTensor_(retain)(self);
  }
  static void _free(THTensor* self) {
    return THTensor_(free)(self);
  }

  // THTensorCopy.h
  static void _copy(THTensor* self, THTensor* src) {
    return THTensor_(copy)(self, src);
  }

  template <class T>
  static void _copyT(THTensor* self, T* src);

  // THTensorMath.h
  static void _fill(THTensor* r, real value) {
    THTensor_(fill)(r, value);
  }
  static void _zero(THTensor* r) {
    THTensor_(zero)(r);
  }
  static void _maskedFill(THTensor* tensor, THByteTensor* mask, real value) {
    THTensor_(maskedFill)(tensor, mask, value);
  }
  static void _maskedCopy(THTensor* tensor, THByteTensor* mask, THTensor* src) {
    THTensor_(maskedCopy)(tensor, mask, src);
  }
  static void _maskedSelect(THTensor* tensor, THTensor* src,
                            THByteTensor* mask) {
    THTensor_(maskedSelect)(tensor, src, mask);
  }
  static void _indexSelect(THTensor* tensor, THTensor* src, int dim,
                           THLongTensor* index) {
    THTensor_(indexSelect)(tensor, src, dim, index);
  }
  static void _indexCopy(THTensor* tensor, int dim, THLongTensor* index,
                         THTensor* src) {
    THTensor_(indexCopy)(tensor, dim, index, src);
  }
  static void _indexFill(THTensor* tensor, int dim, THLongTensor* index,
                         real val) {
    THTensor_(indexFill)(tensor, dim, index, val);
  }
  static accreal _dot(THTensor* t, THTensor* src) {
    return THTensor_(dot)(t, src);
  }
  static real _minall(THTensor* t) {
    return THTensor_(minall)(t);
  }
  static real _maxall(THTensor* t) {
    return THTensor_(maxall)(t);
  }
  static accreal _sumall(THTensor* t) {
    return THTensor_(sumall)(t);
  }
  static accreal _prodall(THTensor* t) {
    return THTensor_(prodall)(t);
  }
  static void _add(THTensor* r, THTensor* t, real value) {
    return THTensor_(add)(r, t, value);
  }
  static void _mul(THTensor* r, THTensor* t, real value) {
    return THTensor_(mul)(r, t, value);
  }
  static void _div(THTensor* r, THTensor* t, real value) {
    return THTensor_(div)(r, t, value);
  }
  static void _cadd(THTensor* r, THTensor* t, real value, THTensor* src) {
    return THTensor_(cadd)(r, t, value, src);
  }
  static void _cmul(THTensor* r, THTensor* t, THTensor* src) {
    return THTensor_(cmul)(r, t, src);
  }
  static void _cdiv(THTensor* r, THTensor* t, THTensor* src) {
    return THTensor_(cdiv)(r, t, src);
  }
  static void _addcmul(THTensor* r, THTensor* t, real value, THTensor* src1,
                      THTensor* src2) {
    return THTensor_(addcmul)(r, t, value, src1, src2);
  }
  static void _addcdiv(THTensor* r, THTensor* t, real value, THTensor* src1,
                      THTensor* src2) {
    return THTensor_(addcdiv)(r, t, value, src1, src2);
  }
  static void _addmv(THTensor* r, real beta, THTensor* t, real alpha,
                    THTensor* mat, THTensor* vec) {
    return THTensor_(addmv)(r, beta, t, alpha, mat, vec);
  }
  static void _addmm(THTensor* r, real beta, THTensor* t, real alpha,
                    THTensor* m1, THTensor* m2) {
    return THTensor_(addmm)(r, beta, t, alpha, m1, m2);
  }
  static void _addr(THTensor* r, real beta, THTensor* t, real alpha,
                   THTensor* vec1, THTensor* vec2) {
    return THTensor_(addr)(r, beta, t, alpha, vec1, vec2);
  }
  static void _max(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(max)(values, indices, t, dim, 1);
  }
  static void _min(THTensor* values, THLongTensor* indices,
                   THTensor* t, int dim) {
    return THTensor_(min)(values, indices, t, dim, 1);
  }
  static void _sum(THTensor* r, THTensor* t, int dim) {
    return THTensor_(sum)(r, t, dim, 1);
  }
  static void _prod(THTensor* r, THTensor* t, int dim) {
    return THTensor_(prod)(r, t, dim, 1);
  }
  static void _cumsum(THTensor* r, THTensor* t, int dim) {
    return THTensor_(cumsum)(r, t, dim);
  }
  static void _cumprod(THTensor* r, THTensor* t, int dim) {
    return THTensor_(cumprod)(r, t, dim);
  }
  static void _sign(THTensor* r, THTensor* t) {
    return THTensor_(sign)(r, t);
  }

#define S1(X) #X
#define S(X) S1(X)
  static constexpr const char* kLuaTypeName = "torch."
    S(TH_CONCAT_2(Real, Tensor));
#undef S
#undef S1
};

#define S(TYPE) \
  template <> inline void TensorOps<Tensor<real>>::_copyT<TH##TYPE##Tensor>( \
      THTensor* self, TH##TYPE##Tensor* src) { \
    return THTensor_(copy##TYPE)(self, src); \
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

#endif
