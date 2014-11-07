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

#include <cmath>
#include <type_traits>
#include <folly/Conv.h>
#include <folly/Likely.h>
#include <folly/ScopeGuard.h>

namespace thpp {

namespace {
template <class T>
Range<T*> makeMutable(Range<const T*> r) {
  return Range<T*>(const_cast<T*>(r.begin()), const_cast<T*>(r.end()));
}
}  // namespace

template <class T>
Tensor<T>::Tensor() : t_(Ops::_new()) { }

template <class T>
Tensor<T>::Tensor(TensorInvalid) : t_(nullptr) { }

template <class T>
Tensor<T>::Tensor(StorageType storage, long storageOffset,
                  LongStorage sizes, LongStorage strides) : Tensor() {
  Ops::_setStorage(t_, storage.th(), storageOffset, sizes.th(), strides.th());
}

template <class T>
Tensor<T>::Tensor(LongStorage sizes, LongStorage strides) : Tensor() {
  Ops::_setStorage(t_, nullptr, 0, sizes.th(), strides.th());
}

template <class T>
Tensor<T>::Tensor(LongRange sizes, LongRange strides)
  : Tensor(LongStorage::wrap(makeMutable(sizes)),
           LongStorage::wrap(makeMutable(strides))) { }

template <class T>
Tensor<T>::Tensor(std::initializer_list<long> sizes,
                  std::initializer_list<long> strides)
  : Tensor(LongStorage(sizes.begin(), sizes.end()),
           LongStorage(strides.begin(), strides.end())) { }

template <class T>
Tensor<T>::Tensor(const std::vector<long>& sizes,
                  const std::vector<long>& strides)
    : Tensor(LongStorage(sizes.begin(), sizes.end()),
             LongStorage(strides.begin(), strides.end())) { }

template <class T>
Tensor<T>::Tensor(ThriftTensor&& thriftTensor) : t_(nullptr) {
  auto buf = detail::deserialize(std::move(thriftTensor),
                                 detail::dataType<T>());
  Storage<T> data(std::move(buf));

  LongStorage s(LongStorage::wrap(makeMutable(LongRange(
      thriftTensor.sizes.data(), thriftTensor.sizes.size()))));

  t_ = Ops::_newWithStorage(data.th(), 0, s.th(), nullptr);
  DCHECK_EQ(data.size(), size());
}

template <class T>
Tensor<T>::~Tensor() {
  destroy();
}

template <class T>
Tensor<T>::Tensor(THType* other, TensorMustAlias) noexcept : t_(other) {
  Ops::_retain(t_);
}

template <class T>
Tensor<T>::Tensor(Tensor&& other) noexcept : t_(other.t_) {
  other.t_ = nullptr;
}

template <class T>
Tensor<T>::Tensor(Tensor&& other, unsigned cloneMode) {
  if ((other.mode() & cloneMode) != cloneMode) {
    t_ = Ops::_newClone(other.mut());
    other.destroy();
  } else {
    t_ = other.t_;
    other.t_ = nullptr;
  }
}

template <class T>
Tensor<T>::Tensor(const THType* other, unsigned cloneMode) {
  if ((cloneMode & UNIQUE) ||
      ((cloneMode & CONTIGUOUS) && !isContiguous(other))) {
    t_ = Ops::_newClone(mut(other));
  } else {
    t_ = Ops::_newWithTensor(mut(other));
  }
}

template <class T>
Tensor<T>::Tensor(THType*&& other)
    : t_(std::move(other)) {
}

template <class T>
Tensor<T>::Tensor(const Tensor& other, unsigned cloneMode)
  : Tensor(other.t_, cloneMode) { }

template <class T>
auto Tensor<T>::operator=(Tensor&& other) noexcept -> Tensor& {
  if (&other != this) {
    if (t_) {
      // Careful. If a and b alias each other (a.t_ == b.t_), that assumption
      // must continue to hold if we do a = std::move(c). So the obvious
      // "t_ = other.t_; other.t_ = nullptr;" will not work.
      Ops::_set(t_, other.t_);
      other.destroy();
    } else {
      t_ = other.t_;
      other.t_ = nullptr;
    }
  }
  return *this;
}

template <class T>
auto Tensor<T>::operator=(const Tensor& other) -> Tensor& {
  if (&other != this) {
    if (t_) {
      Ops::_set(t_, other.mut());
    } else {
      t_ = Ops::_newWithTensor(other.mut());
    }
  }
  return *this;
}

template <class T>
auto Tensor<T>::operator=(THType*&& other) -> Tensor& {
  if (other != t_) {
    destroy();
    t_ = std::move(other);
  }
  return *this;
}

template <class T>
auto Tensor<T>::moveAsTH() -> THType* {
  using std::swap;
  THType* out = nullptr;
  swap(out, t_);
  return out;
}

template <class T>
void Tensor<T>::serialize(ThriftTensor& out,
                          ThriftTensorEndianness endianness,
                          bool mayShare) {
  auto buf = Storage<T>(Ops::_storage(t_)).getIOBuf();
  buf.trimStart(Ops::_storageOffset(t_) * sizeof(T));
  detail::serialize(
      out,
      sizes(),
      strides(),
      std::move(buf),
      detail::dataType<T>(),
      sizeof(T),
      endianness,
      mayShare);
}

template <class T>
void Tensor<T>::force(unsigned newMode) {
  if ((mode() & newMode) == newMode)
    return;

  *this = Tensor(std::move(*this), newMode);
}

template <class T>
LongRange Tensor<T>::sizes() const {
  return LongRange(t_->size, t_->nDimension);
}

template <class T>
LongRange Tensor<T>::strides() const {
  return LongRange(t_->stride, t_->nDimension);
}

template <class T>
bool Tensor<T>::isUnique(const THType* th) {
  return !th->storage || th->storage->refcount == 1;
}

template <class T>
bool Tensor<T>::isContiguous(const THType* th) {
  return Ops::_isContiguous(th);
}

template <class T>
bool Tensor<T>::isExactlyEqual(const Tensor<T>& other) const {
  if (ndims() != other.ndims()) {
    throw std::invalid_argument("isExactlyEqual: dimension mismatch");
  }

  for (int i = 0; i < ndims(); ++i) {
    if (size(i) != other.size(i)) {
      throw std::invalid_argument("isExactlyEqual: size mismatch");
    }
  }

  if (ndims() == 1) {
    for (int i = 0; i < size(0); ++i) {
      if (at({i}) != other.at({i})) {
        return false;
      }
    }
  } else {
    for (int i = 0; i < size(0); ++i) {
      if (!(*this)[i].isExactlyEqual(other[i])) {
        return false;
      }
    }
  }

  return true;
}

template <class T>
bool Tensor<T>::isApproximatelyEqual(const Tensor<T>& other,
                                     float relativeError) const {
  if (!std::is_floating_point<T>::value) {
    return isExactlyEqual(other);
  }

  if (ndims() != other.ndims()) {
    throw std::invalid_argument("isExactlyEqual: dimension mismatch");
  }

  for (int i = 0; i < ndims(); ++i) {
    if (size(i) != other.size(i)) {
      throw std::invalid_argument("isExactlyEqual: size mismatch");
    }
  }

  if (ndims() == 1) {
    const auto adjRelativeError = 0.5f * relativeError;

    for (int i = 0; i < size(0); ++i) {
      const auto a = at({i});
      const auto b = other.at({i});

      // Handle special cases
      if (a == b || (std::isnan(a) && std::isnan(b))) {
        continue;
      } else if (!std::isfinite(a) && !std::isfinite(b)) {
        if (std::signbit(a) == std::signbit(b)) {
          continue;
        } else {
          return false;
        }
      }

      // Compare the difference against the mean values
      if (std::abs(a - b) > adjRelativeError * (std::abs(a) + std::abs(b))) {
        return false;
      }
    }
  } else {
    for (int i = 0; i < size(0); ++i) {
      if (!(*this)[i].isApproximatelyEqual(other[i], relativeError)) {
        return false;
      }
    }
  }

  return true;
}

template <class T>
long Tensor<T>::size() const {
  return Ops::_nElement(t_);
}

template <class T>
void Tensor<T>::narrow(const Tensor& src, int dim, long firstIndex, long size) {
  Ops::_narrow(t_, src.mut(), dim, firstIndex, size);
}

template <class T>
void Tensor<T>::select(const Tensor& src, int dim, long index) {
  if (src.ndims() == 1) {
    if (UNLIKELY(dim != 0)) {
      throw std::invalid_argument("invalid dimension for vector select");
    }
    if (UNLIKELY(index < 0 || index >= src.size(0))) {
      throw std::invalid_argument("invalid index for vector select");
    }
    auto s = src.mut();
    Ops::_setStorage1d(t_, s->storage,
                       s->storageOffset + index * s->stride[0],
                       1, 1);
  } else {
    Ops::_select(t_, src.mut(), dim, index);
  }
}

template <class T>
void Tensor<T>::transpose(const Tensor& src, int dim1, int dim2) {
  Ops::_transpose(t_, src.mut(), dim1, dim2);
}

template <class T>
void Tensor<T>::transpose() {
  std::reverse(t_->stride, t_->stride + t_->nDimension);
  std::reverse(t_->size, t_->size + t_->nDimension);
}

template <class T>
void Tensor<T>::unfold(const Tensor& src, int dim, long size, long step) {
  Ops::_unfold(t_, src.mut(), dim, size, step);
}

template <class T>
void Tensor<T>::squeeze(const Tensor& src) {
  Ops::_squeeze(t_, src.mut());
}

template <class T>
void Tensor<T>::squeeze(const Tensor& src, int dim) {
  Ops::_squeeze1d(t_, src.mut(), dim);
}

template <class T>
void Tensor<T>::resize(std::initializer_list<long> newSizes,
                       std::initializer_list<long> newStrides) {
  resize(LongStorage(newSizes.begin(), newSizes.end()),
         LongStorage(newStrides.begin(), newStrides.end()));
}

template <class T>
void Tensor<T>::resize(LongStorage sizes, LongStorage strides) {
  Ops::_resize(t_, sizes.th(), strides.th());
}

template <class T>
void Tensor<T>::resize(LongRange sizes, LongRange strides) {
  resize(LongStorage::wrap(makeMutable(sizes)),
         LongStorage::wrap(makeMutable(strides)));
}

template <class T>
void Tensor<T>::resizeAs(const Tensor& src) {
  Ops::_resizeAs(t_, src.mut());
}

template <class T>
bool Tensor<T>::isScalar() const {
  return ndims() == 1 && size(0) == 1 && stride(0) == 1;
}

template <class T>
T* Tensor<T>::data() {
  return Ops::_data(t_);
}

template <class T>
const T* Tensor<T>::data() const {
  return Ops::_data(t_);
}

template <class T>
auto Tensor<T>::storage() -> StorageType {
  return StorageType(Ops::_storage(t_));
}

template <class T>
long Tensor<T>::storageOffset() const {
  return Ops::_storageOffset(t_);
}

template <class T>
Tensor<T> Tensor<T>::operator[](long index) const {
  Tensor<T> nt(*this);
  nt.select(0, index);
  return nt;
}

template <class T>
Tensor<T> Tensor<T>::operator[](std::initializer_list<long> indexes) const {
  Tensor<T> nt(*this);
  int dim = 0;
  for (long index : indexes) {
    if (index == -1) {
      ++dim;
    } else {
      nt.select(dim, index);
    }
  }
  return nt;
}

template <class T>
T& Tensor<T>::at(std::initializer_list<long> indexes) {
  if (indexes.size() > ndims() || indexes.size() == 0) {
    throw std::invalid_argument("must provide 1 to ndims() indices");
  }

  auto ptr = data();
  auto dim = 0;
  for (auto it = indexes.begin(); it != indexes.end(); ++it) {
    const auto offset = *it;
    if (offset >= size(dim)) {
      throw std::invalid_argument("index out of range");
    }

    ptr += offset * stride(dim++);
  }

  return *ptr;
}

template <class T>
const T& Tensor<T>::at(std::initializer_list<long> indexes) const {
  if (indexes.size() > ndims() || indexes.size() == 0) {
    throw std::invalid_argument("must provide 1 to ndims() indices");
  }

  auto ptr = data();
  auto dim = 0;
  for (auto it = indexes.begin(); it != indexes.end(); ++it) {
    const auto offset = *it;
    if (offset >= size(dim)) {
      throw std::invalid_argument("index out of range");
    }

    ptr += offset * stride(dim++);
  }

  return *ptr;
}

template <class T>
void Tensor<T>::fill(T value) {
  Ops::_fill(t_, value);
}

template <class T>
void Tensor<T>::zero() {
  Ops::_zero(t_);
}

template <class T>
template <class U>
void Tensor<T>::copy(const Tensor<U>& src) {
  Ops::_copyT(t_, src.mut());
}

template <class T>
void Tensor<T>::maskedFill(const ByteTensor& mask, T value) {
  Ops::_maskedFill(t_, mask.mut(), value);
}

template <class T>
void Tensor<T>::maskedCopy(const ByteTensor& mask, const Tensor& src) {
  Ops::_maskedCopy(t_, mask.mut(), src.mut());
}

template <class T>
auto Tensor<T>::maskedSelect(const ByteTensor& mask) const -> Tensor {
  Tensor r;
  Ops::_maskedSelect(&r.t_, mut(), mask.mut());
  return r;
}

template <class T>
auto Tensor<T>::indexSelect(int dim, const LongTensor& index) const -> Tensor {
  Tensor r;
  Ops::_indexSelect(&r.t_, mut(), dim, index.mut());
  return r;
}

template <class T>
void Tensor<T>::indexFill(int dim, const LongTensor& index, T val) {
  Ops::_indexFill(t_, dim, index.mut(), val);
}

template <class T>
auto Tensor<T>::dot(const Tensor& other) const -> accurate_type {
  return Ops::_dot(t_, other.t_);
}

#define TENSOR_REDUCE_OP(ret, name) \
  template <class T> \
  auto Tensor<T>::name() const -> ret { \
    return Ops::_ ## name(mut()); \
  }
TENSOR_REDUCE_OP(T, minall)
TENSOR_REDUCE_OP(T, maxall)
TENSOR_REDUCE_OP(accurate_type, sumall)
TENSOR_REDUCE_OP(accurate_type, trace)
#undef TENSOR_REDUCE_OP

#define TENSOR_ST_OP(name) \
  template <class T> \
  void Tensor<T>::name(const Tensor& src, T value) { \
    Ops::_ ## name(t_, src.mut(), value); \
  }
TENSOR_ST_OP(add)
TENSOR_ST_OP(mul)
TENSOR_ST_OP(div)
#undef TENSOR_ST_OP

#define TENSOR_TST_OP(name) \
  template <class T> \
  void Tensor<T>::name(const Tensor& a, T value, const Tensor& b) { \
    Ops::_ ## name(t_, a.mut(), value, b.mut()); \
  }
TENSOR_TST_OP(cadd)
#undef TENSOR_TST_OP

#define TENSOR_TT_OP(name) \
  template <class T> \
  void Tensor<T>::name(const Tensor& a, const Tensor& b) { \
    Ops::_ ## name(t_, a.mut(), b.mut()); \
  }
TENSOR_TT_OP(cmul)
TENSOR_TT_OP(cdiv)
#undef TENSOR_TT_OP

#define TENSOR_TSTT_OP(name) \
  template <class T> \
  void Tensor<T>::name(const Tensor& a, T value, const Tensor& b, \
                       const Tensor& c) { \
    Ops::_ ## name(t_, a.mut(), value, b.mut(), c.mut()); \
  }
TENSOR_TSTT_OP(addcmul)
TENSOR_TSTT_OP(addcdiv)
#undef TENSOR_TSTT_OP

#define TENSOR_STSTT_OP(name) \
  template <class T> \
  void Tensor<T>::name(T val1, const Tensor& a, \
                       T val2, const Tensor& b, const Tensor& c) { \
    Ops::_ ## name(t_, val1, a.mut(), val2, b.mut(), c.mut()); \
  }
TENSOR_STSTT_OP(addmv)
TENSOR_STSTT_OP(addmm)
TENSOR_STSTT_OP(addr)
#undef TENSOR_STSTT_OP

#define TENSOR_ARGM_OP(name) \
  template <class T> \
  auto Tensor<T>::name(int dim) const -> std::pair<Tensor, LongTensor> { \
    std::pair<Tensor, LongTensor> dest; \
    Ops::_ ## name(dest.first.t_, dest.second.t_, mut(), dim); \
    return dest; \
  }
TENSOR_ARGM_OP(min)
TENSOR_ARGM_OP(max)
#undef TENSOR_ARGM_OP

#define TENSOR_DIM_OP(name) \
  template <class T> \
  auto Tensor<T>::name(int dim) const -> Tensor { \
    Tensor dest; \
    Ops::_ ## name(dest.t_, mut(), dim); \
    return dest; \
  }
TENSOR_DIM_OP(sum)
TENSOR_DIM_OP(prod)
TENSOR_DIM_OP(cumsum)
#undef TENSOR_DIM_OP

template <class T>
auto Tensor<T>::sign() const -> Tensor {
  Tensor dest;
  Ops::_sign(dest.t_, mut());
  return dest;
}

template <class T>
auto Tensor<T>::cross(const Tensor& b, int dim) const -> Tensor {
  Tensor dest;
  Ops::_cross(dest.t_, mut(), b.mut(), dim);
  return dest;
}

template <class T>
void Tensor<T>::clear() {
  Ops::_setStorage(t_, nullptr, 0, nullptr, nullptr);
}

template <class T>
void Tensor<T>::destroy() {
  if (t_) {
    Ops::_free(t_);
    t_ = nullptr;
  }
}

template <class T>
Tensor<T> operator-(const Tensor<T>& a) {
  Tensor<T> r;
  r.mul(a, -1);
  return r;
}

template <class T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
  Tensor<T> r;
  r.cadd(a, 1, b);
  return r;
}

template <class T>
Tensor<T>& operator+=(Tensor<T>& a, const Tensor<T>& b) {
  a.cadd(1, b);
  return a;
}

template <class T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
  Tensor<T> r;
  r.cadd(a, -1, b);
  return r;
}

template <class T>
Tensor<T>& operator-=(Tensor<T>& a, const Tensor<T>& b) {
  a.cadd(-1, b);
  return a;
}

template <class T>
Tensor<T> operator*(const Tensor<T>& a, T b) {
  Tensor<T> r;
  r.mul(a, b);
  return r;
}

template <class T>
Tensor<T>& operator*=(Tensor<T>& a, T b) {
  a.mul(b);
  return a;
}

template <class T>
Tensor<T> operator/(const Tensor<T>& a, T b) {
  Tensor<T> r;
  r.div(a, b);
  return r;
}

template <class T>
Tensor<T>& operator/=(Tensor<T>& a, T b) {
  a.div(b);
  return a;
}

template <class T>
std::string Tensor<T>::str() const {
  std::string out;
  auto sz = sizes();
  out.reserve(20 + 4 * sz.size());
  folly::toAppend(kLuaTypeName, "(", &out);

  bool first = true;
  for (long s : sz) {
    if (!first) {
      out += "x";
    }
    first = false;
    folly::toAppend(s, &out);
  }

  out += ")";
  return out;
}

}  // namespaces
