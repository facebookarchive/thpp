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

namespace thpp {

template <class Tensor>
TensorPtr<Tensor>::TensorPtr() noexcept : hasTensor_(false) {
}

template <class Tensor>
TensorPtr<Tensor>::TensorPtr(THType* th) noexcept
  : hasTensor_(th) {
  if (hasTensor_) {
    construct(th, true);
  }
}

template <class Tensor>
TensorPtr<Tensor>::TensorPtr(TensorPtr&& other) noexcept
  : hasTensor_(other.hasTensor_) {
  if (hasTensor_) {
    construct(other.tensor_.mut(), false);
    other.hasTensor_ = false;
  }
}

template <class Tensor>
TensorPtr<Tensor>::TensorPtr(const TensorPtr& other) noexcept
  : hasTensor_(other.hasTensor_) {
  if (hasTensor_) {
    construct(other.tensor_.mut(), true);
  }
}

template <class Tensor>
template <class... Args>
TensorPtr<Tensor>::TensorPtr(detail::MakeTensorPtr, Args&&... args)
  : hasTensor_(true),
    tensor_(std::forward<Args>(args)...) {
}

template <class Tensor>
TensorPtr<Tensor>::~TensorPtr() {
  destroy();
}

template <class Tensor>
void TensorPtr<Tensor>::destroy() noexcept {
  if (hasTensor_) {
    tensor_.~Tensor();
    hasTensor_ = false;
  }
}

template <class Tensor>
void TensorPtr<Tensor>::construct(THType* th, bool incRef)
  noexcept {
  DCHECK(hasTensor_);
  new (&tensor_) Tensor(detail::SetTH(), th, incRef);
}

template <class Tensor>
auto TensorPtr<Tensor>::operator=(TensorPtr&& other) noexcept -> TensorPtr& {
  if (this != &other) {
    destroy();
    if (other.hasTensor_) {
      hasTensor_ = true;
      construct(other.tensor_.mut(), false);
      other.hasTensor_ = false;
    }
  }
  return *this;
}

template <class Tensor>
auto TensorPtr<Tensor>::operator=(const TensorPtr& other) noexcept
  -> TensorPtr& {
  if (this != &other) {
    destroy();
    if (other.hasTensor_) {
      hasTensor_ = true;
      construct(other.tensor_.mut(), true);
    }
  }
  return *this;
}

template <class Tensor>
Tensor* TensorPtr<Tensor>::get() const noexcept {
  return hasTensor_ ? &tensor_ : nullptr;
}

template <class Tensor>
auto TensorPtr<Tensor>::moveAsTH() noexcept -> THType* {
  auto p = hasTensor_ ? tensor_.mut() : nullptr;
  hasTensor_ = false;
  return p;
}

template <class Tensor>
bool TensorPtr<Tensor>::operator==(const TensorPtr& other) const noexcept {
  return (hasTensor_ == other.hasTensor_ &&
          (!hasTensor_ || tensor_.mut() == other.tensor_.mut()));
}


}  // namespaces
