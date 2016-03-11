/*
 * Copyright 2015 Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef NO_FOLLY
#include <folly/Optional.h>
#endif

namespace thpp {

namespace detail {
struct MakeTensorPtr {};
struct SetTH {};
}  // namespace

/**
 * A shared pointer to a tensor.
 *
 * thpp::Tensor objects share data, but not metadata. That is, when copying
 * a thpp::Tensor object, the new object initially shares data with the
 * original tensor, but the metadata (sizes, strides) is separate.
 *
 * TensorPtr allows you to share (reference counted) access to a tensor
 * (including metadata). [1]
 *
 * [1] TensorPtr<Tensor> is very similar to std::shared_ptr<Tensor>, except
 * that TensorPtr uses Torch's TH*Tensor internal reference counting mechanism.
 */
template <class Tensor>
class TensorPtr {
  template <class T, class... Args>
  friend TensorPtr<T> makeTensorPtr(Args&&... args);

  template <class... Args>
  explicit TensorPtr(detail::MakeTensorPtr, Args&&... args);
 public:
  typedef Tensor element_type;
  typedef typename Tensor::THType THType;
  static constexpr const char* kLuaTypeName = Tensor::kLuaTypeName;

  // Create an empty TensorPtr
  TensorPtr() noexcept;

  // Create a TensorPtr from a given THFloatTensor / THLongTensor / etc
  // raw pointer. Increments the reference count.
  explicit TensorPtr(THType* th) noexcept;

  // Move and copy constructors
  TensorPtr(TensorPtr&& other) noexcept;
  TensorPtr(const TensorPtr& other) noexcept;

  ~TensorPtr();

  // Move and copy assignment operators
  TensorPtr& operator=(TensorPtr&& other) noexcept;
  TensorPtr& operator=(const TensorPtr& other) noexcept;

  // Dereference
  Tensor& operator*() const noexcept { return *get(); }
  Tensor* operator->() const noexcept { return get(); }
  Tensor* get() const noexcept;

  // True iff non-empty
  explicit operator bool() const noexcept { return hasTensor_; }

  // Return a pointer to the underlying THFloatTensor / THLongTensor etc.
  // Does not change the reference count! This is similar to
  // get()->asTH(), except that it works (and returns nullptr) if the pointer
  // is empty.
  THType* th() const noexcept;

  // Steal the reference to the underlying THFloatTensor / THLongTensor etc.
  // The TensorPtr is empty at the end of this operation, but the reference
  // count is not changed. It is your responsibility to call
  // THFloatTensor_(free) (or equivalent for other types) on the returned value
  // (or pass it to code that steals the reference, such as luaT_pushudata).
  THType* moveAsTH() noexcept;

  // Do two TensorPtr objects point to the same tensor?
  bool operator==(const TensorPtr& other) const noexcept;

 private:
  void destroy() noexcept;
  void construct(THType* th, bool incRef) noexcept;

  // Not using folly::Optional, so we don't accidentally call Tensor's
  // (copy / move) (constructor / assignment operator), which has value,
  // not reference, semantics.
  bool hasTensor_;
  union {
    mutable Tensor tensor_;
  };

  template<class U>
  friend bool operator==(const U& y, const TensorPtr& x) { return x == y; }
  template<class U>
  friend bool operator!=(const U& y, const TensorPtr& x) { return !static_cast<bool>(x == y); }
  template<class U>
  friend bool operator!=(const TensorPtr& y, const U& x) { return !static_cast<bool>(y == x); }
};

template <class Tensor, class... Args>
inline TensorPtr<Tensor> makeTensorPtr(Args&&... args) {
  return TensorPtr<Tensor>(
      detail::MakeTensorPtr(), std::forward<Args>(args)...);
}

}  // namespaces

#include <thpp/TensorPtr-inl.h>
