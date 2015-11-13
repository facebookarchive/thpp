/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_TENSORBASE_H_
#error This file may only be included from thpp/TensorBase.h
#endif

#include <cmath>
#include <type_traits>
#include <folly/Conv.h>
#include <folly/Likely.h>

namespace thpp {

namespace detail {

template <class T, class StorageT, class Derived>
inline Derived& D(TensorBase<T, StorageT, Derived>& v) {
  return static_cast<Derived&>(v);
}

template <class T, class StorageT, class Derived>
inline const Derived& D(const TensorBase<T, StorageT, Derived>& v) {
  return static_cast<const Derived&>(v);
}

}  // namespace detail

template <class T, class StorageT, class Derived>
TensorBase<T, StorageT, Derived>::TensorBase(THType* t) : t_(t) {
  DCHECK(t_);
}

template <class T, class StorageT, class Derived>
TensorBase<T, StorageT, Derived>::~TensorBase() {
  DCHECK(t_);
  Ops::_free(t_);
#ifndef NDEBUG
  t_ = nullptr;
#endif
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::force(unsigned newMode) {
  if ((mode() & newMode) == newMode)
    return;

  *D() = Derived(std::move(*D()), newMode);
}

template <class T, class StorageT, class Derived>
LongRange TensorBase<T, StorageT, Derived>::sizes() const {
  return LongRange(t_->size, t_->nDimension);
}

template <class T, class StorageT, class Derived>
LongRange TensorBase<T, StorageT, Derived>::strides() const {
  return LongRange(t_->stride, t_->nDimension);
}

template <class T, class StorageT, class Derived>
bool TensorBase<T, StorageT, Derived>::isUnique(const THType* th) {
  return StorageType::isUnique(th->storage);
}

template <class T, class StorageT, class Derived>
bool TensorBase<T, StorageT, Derived>::isContiguous(const THType* th) {
  return Ops::_isContiguous(th);
}

template <class T, class StorageT, class Derived>
bool TensorBase<T, StorageT, Derived>::isExactlyEqual(
    const TensorBase& other) const {
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
      if (D()->at({i}) != other.D()->at({i})) {
        return false;
      }
    }
  } else {
    for (int i = 0; i < size(0); ++i) {
      if (!(*D())[i].isExactlyEqual(other[i])) {
        return false;
      }
    }
  }

  return true;
}

template <class T, class StorageT, class Derived>
bool TensorBase<T, StorageT, Derived>::isApproximatelyEqual(
    const TensorBase& other,
    float relativeError) const {
  if (!std::is_floating_point<T>::value) {
    return isExactlyEqual(other);
  }

  if (ndims() != other.ndims()) {
    throw std::invalid_argument("isApproximatelyEqual: dimension mismatch");
  }

  for (int i = 0; i < ndims(); ++i) {
    if (size(i) != other.size(i)) {
      throw std::invalid_argument("isApproximatelyEqual: size mismatch");
    }
  }

  if (ndims() == 1) {
    const auto adjRelativeError = 0.5f * relativeError;

    for (int i = 0; i < size(0); ++i) {
      const auto a = D()->at({i});
      const auto b = other.D()->at({i});

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
      if (!(*D())[i].isApproximatelyEqual(other[i], relativeError)) {
        return false;
      }
    }
  }

  return true;
}

template <class T, class StorageT, class Derived>
long TensorBase<T, StorageT, Derived>::size() const {
  return Ops::_nElement(t_);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::narrow(
    const TensorBase& src, int dim, long firstIndex, long size) {
  Ops::_narrow(t_, src.mut(), dim, firstIndex, size);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::select(
    const TensorBase& src, int dim, long index) {
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

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::transpose(
    const TensorBase& src, int dim1, int dim2) {
  Ops::_transpose(t_, src.mut(), dim1, dim2);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::transpose() {
  std::reverse(t_->stride, t_->stride + t_->nDimension);
  std::reverse(t_->size, t_->size + t_->nDimension);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::unfold(
    const TensorBase& src, int dim, long size, long step) {
  Ops::_unfold(t_, src.mut(), dim, size, step);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::squeeze(const TensorBase& src) {
  Ops::_squeeze(t_, src.mut());
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::squeeze(const TensorBase& src, int dim) {
  Ops::_squeeze1d(t_, src.mut(), dim);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::resize(
    std::initializer_list<long> newSizes,
                       std::initializer_list<long> newStrides) {
  resize(LongStorage(newSizes.begin(), newSizes.end()),
         LongStorage(newStrides.begin(), newStrides.end()));
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::resize(
    LongStorage sizes, LongStorage strides) {
  Ops::_resize(t_, sizes.th(), strides.th());
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::resize(
    LongRange sizes, LongRange strides) {
  resize(LongStorage::wrap(detail::makeMutable(sizes)),
         LongStorage::wrap(detail::makeMutable(strides)));
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::resizeAs(const TensorBase& src) {
  Ops::_resizeAs(t_, src.mut());
}

template <class T, class StorageT, class Derived>
bool TensorBase<T, StorageT, Derived>::isScalar() const {
  return ndims() == 1 && size(0) == 1 && stride(0) == 1;
}

template <class T, class StorageT, class Derived>
T* TensorBase<T, StorageT, Derived>::data() {
  return Ops::_data(t_);
}

template <class T, class StorageT, class Derived>
const T* TensorBase<T, StorageT, Derived>::data() const {
  return Ops::_data(t_);
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::storage() -> StorageType {
  return StorageType(Ops::_storage(t_));
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::storageRef(StorageBuffer* buf)
  -> StorageType& {
  auto pbuf = reinterpret_cast<typename StorageT::THType**>(buf);
  *pbuf = Ops::_storage(t_);
  // This relies on the fact that StorageT doesn't contain any members
  // other than the pointer to the appopriate THStorage
  return *reinterpret_cast<StorageT*>(pbuf);
}

template <class T, class StorageT, class Derived>
long TensorBase<T, StorageT, Derived>::storageOffset() const {
  return Ops::_storageOffset(t_);
}

template <class T, class StorageT, class Derived>
Derived TensorBase<T, StorageT, Derived>::operator[](long index) const {
  Derived nt(*D());
  nt.select(0, index);
  return nt;
}

template <class T, class StorageT, class Derived>
Derived TensorBase<T, StorageT, Derived>::operator[](
    std::initializer_list<long> indexes) const {
  Derived nt(*D());
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

template <class T, class StorageT, class Derived>
size_t TensorBase<T, StorageT, Derived>::offsetOf(
    std::initializer_list<long> indexes) const {
  if (indexes.size() != ndims()) {
    throw std::invalid_argument("must provide ndims() indices");
  }

  size_t offset = 0;
  auto dim = 0;
  for (auto it = indexes.begin(); it != indexes.end(); ++it) {
    const auto idx = *it;
    if (idx >= size(dim)) {
      throw std::invalid_argument("index out of range");
    }

    offset += idx * stride(dim++);
  }

  return offset;
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::fill(T value) {
  Ops::_fill(t_, value);
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::zero() {
  Ops::_zero(t_);
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::dot(
    const TensorBase& other) const -> accurate_type {
  return Ops::_dot(t_, other.t_);
}

#define TENSOR_REDUCE_OP(ret, name) \
  template <class T, class StorageT, class Derived> \
  auto TensorBase<T, StorageT, Derived>::name() const -> ret { \
    return Ops::_ ## name(mut()); \
  }
TENSOR_REDUCE_OP(T, minall)
TENSOR_REDUCE_OP(T, maxall)
TENSOR_REDUCE_OP(accurate_type, sumall)
#undef TENSOR_REDUCE_OP

#define TENSOR_ST_OP(name) \
  template <class T, class StorageT, class Derived> \
  void TensorBase<T, StorageT, Derived>::name( \
      const TensorBase& src, T value) { \
    Ops::_ ## name(t_, src.mut(), value); \
  }
TENSOR_ST_OP(add)
TENSOR_ST_OP(mul)
TENSOR_ST_OP(div)
#undef TENSOR_ST_OP

#define TENSOR_TST_OP(name) \
  template <class T, class StorageT, class Derived> \
  void TensorBase<T, StorageT, Derived>::name( \
      const TensorBase& a, T value, const TensorBase& b) { \
    Ops::_ ## name(t_, a.mut(), value, b.mut()); \
  }
TENSOR_TST_OP(cadd)
#undef TENSOR_TST_OP

#define TENSOR_TT_OP(name) \
  template <class T, class StorageT, class Derived> \
  void TensorBase<T, StorageT, Derived>::name( \
      const TensorBase& a, const TensorBase& b) { \
    Ops::_ ## name(t_, a.mut(), b.mut()); \
  }
TENSOR_TT_OP(cmul)
TENSOR_TT_OP(cdiv)
#undef TENSOR_TT_OP

#define TENSOR_TSTT_OP(name) \
  template <class T, class StorageT, class Derived> \
  void TensorBase<T, StorageT, Derived>::name( \
      const TensorBase& a, T value, const TensorBase& b, \
                       const TensorBase& c) { \
    Ops::_ ## name(t_, a.mut(), value, b.mut(), c.mut()); \
  }
TENSOR_TSTT_OP(addcmul)
TENSOR_TSTT_OP(addcdiv)
#undef TENSOR_TSTT_OP

#define TENSOR_STSTT_OP(name) \
  template <class T, class StorageT, class Derived> \
  void TensorBase<T, StorageT, Derived>::name(T val1, const TensorBase& a, \
                       T val2, const TensorBase& b, const TensorBase& c) { \
    Ops::_ ## name(t_, val1, a.mut(), val2, b.mut(), c.mut()); \
  }
TENSOR_STSTT_OP(addmv)
TENSOR_STSTT_OP(addmm)
TENSOR_STSTT_OP(addr)
#undef TENSOR_STSTT_OP

#define TENSOR_DIM_OP(name) \
  template <class T, class StorageT, class Derived> \
  auto TensorBase<T, StorageT, Derived>::name(int dim) const -> Derived { \
    Derived dest; \
    Ops::_ ## name(dest.t_, mut(), dim); \
    return dest; \
  }
TENSOR_DIM_OP(sum)
TENSOR_DIM_OP(prod)
TENSOR_DIM_OP(cumsum)
#undef TENSOR_DIM_OP

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::sign() const -> Derived {
  Derived dest;
  Ops::_sign(dest.t_, mut());
  return dest;
}

template <class T, class StorageT, class Derived>
auto TensorBase<T, StorageT, Derived>::cloneTH(const THType* other,
                                               unsigned cloneMode) -> THType* {
  if ((cloneMode & UNIQUE) ||
      ((cloneMode & CONTIGUOUS) && !isContiguous(other))) {
    return Ops::_newClone(mut(other));
  }

  return Ops::_newWithTensor(mut(other));
}

template <class T, class StorageT, class Derived>
void TensorBase<T, StorageT, Derived>::clear() {
  Ops::_setStorage(t_, nullptr, 0, nullptr, nullptr);
}

template <class T, class StorageT, class Derived>
Derived operator-(const TensorBase<T, StorageT, Derived>& a) {
  Derived r;
  r.mul(a, -1);
  return r;
}

template <class T, class StorageT, class Derived>
Derived operator+(const TensorBase<T, StorageT, Derived>& a,
                  const TensorBase<T, StorageT, Derived>& b) {
  Derived r;
  r.cadd(a, 1, b);
  return r;
}

template <class T, class StorageT, class Derived>
Derived& operator+=(TensorBase<T, StorageT, Derived>& a,
                    const TensorBase<T, StorageT, Derived>& b) {
  a.cadd(1, b);
  return detail::D(a);
}

template <class T, class StorageT, class Derived>
Derived operator-(const TensorBase<T, StorageT, Derived>& a,
                  const TensorBase<T, StorageT, Derived>& b) {
  Derived r;
  r.cadd(a, -1, b);
  return r;
}

template <class T, class StorageT, class Derived>
Derived& operator-=(TensorBase<T, StorageT, Derived>& a,
                    const TensorBase<T, StorageT, Derived>& b) {
  a.cadd(-1, b);
  return detail::D(a);
}

template <class T, class StorageT, class Derived>
Derived operator*(const TensorBase<T, StorageT, Derived>& a, T b) {
  Derived r;
  r.mul(a, b);
  return r;
}

template <class T, class StorageT, class Derived>
Derived& operator*=(TensorBase<T, StorageT, Derived>& a, T b) {
  a.mul(b);
  return detail::D(a);
}

template <class T, class StorageT, class Derived>
Derived operator/(const TensorBase<T, StorageT, Derived>& a, T b) {
  Derived r;
  r.div(a, b);
  return r;
}

template <class T, class StorageT, class Derived>
Derived& operator/=(TensorBase<T, StorageT, Derived>& a, T b) {
  a.div(b);
  return detail::D(a);
}

template <class T, class StorageT, class Derived>
std::string TensorBase<T, StorageT, Derived>::str() const {
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
