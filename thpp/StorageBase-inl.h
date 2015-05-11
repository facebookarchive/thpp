/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_STORAGEBASE_H_
#error This file may only be included from thpp/StorageBase.h
#endif

namespace thpp {

template <class T, class Derived>
void StorageBase<T, Derived>::up() {
  if (t_) Ops::_retain(t_);
}

template <class T, class Derived>
void StorageBase<T, Derived>::down() {
  if (t_) Ops::_free(t_);
}

template <class T, class Derived>
void StorageBase<T, Derived>::check(size_t index) const {
  if (UNLIKELY(index >= size())) {
    throw std::out_of_range("Storage index out of range");
  }
}

template <class T, class Derived>
auto StorageBase<T, Derived>::moveAsTH() -> THType* {
  using std::swap;
  THType* out = nullptr;
  swap(out, t_);
  return out;
}

template <class T, class Derived>
void StorageBase<T, Derived>::resizeUninitialized(size_t n) {
  if (n == 0) {
    down();
    t_ = nullptr;
    return;
  }

  if (t_) {
    Ops::_resize(t_, n);
  } else {
    t_ = Ops::_newWithSize(n);
  }
}

}  // namespaces
