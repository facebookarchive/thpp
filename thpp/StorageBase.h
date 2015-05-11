/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_STORAGEBASE_H_
#define THPP_STORAGEBASE_H_

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <folly/Likely.h>

namespace thpp {

template <class T, class StorageT, class Derived> class TensorBase;

namespace detail {
template <class T> struct StorageOps;
}  // namespace detail

template <class T, class Derived>
class StorageBase {
  template <class TT, class TStorageT, class TDerived>
  friend class TensorBase;
 protected:
  typedef detail::StorageOps<Derived> Ops;

 public:
  typedef T value_type;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T* iterator;
  typedef const T* const_iterator;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef typename Ops::type THType;

  T* data() { return t_ ? t_->data : nullptr; }
  const T* data() const { return t_ ? t_->data : nullptr; }
  iterator begin() { return data(); }
  const_iterator begin() const { return data(); }
  const_iterator cbegin() const { return data(); }
  iterator end() { return t_ ? (t_->data + t_->size) : nullptr; }
  const_iterator end() const { return t_ ? (t_->data + t_->size) : nullptr; }
  const_iterator cend() const { return end(); }

  T& operator[](size_t index) { return data()[index]; }
  const T& operator[](size_t index) const { return data()[index]; }
  T& at(size_t index) { check(index); return operator[](index); }
  const T& at(size_t index) const { check(index); return operator[](index); }

  bool unique() const { return !t_ || t_->refcount == 1; }

  size_t size() const { return t_ ? t_->size : 0; }

  static constexpr const char* kLuaTypeName = Ops::kLuaTypeName;

  // Get a pointer to the underlying TH object; *this releases ownership
  // of that object.
  THType* moveAsTH();

  void resizeUninitialized(size_t n);

 protected:
  StorageBase() { }  // leave t_ uninitialized
  explicit StorageBase(THType* t) : t_(t) { }

  THType* th() { return t_; }
  const THType* th() const { return t_; }

  void up();
  void down();
  void check(size_t index) const;

  // NOTE: May not have any other fields, as we reinterpret_cast
  // liberally between Ops::type* and Storage*
  THType* t_;

 private:
  inline Derived* D() { return static_cast<Derived*>(this); }
  inline const Derived* D() const { return static_cast<const Derived*>(this); }
};

template <class T, class Derived>
constexpr const char* StorageBase<T, Derived>::kLuaTypeName;

// Define IsStorage<T> to be used in template specializations

template <class T, class Enable=void>
struct IsStorage : public std::false_type { };

template <class T>
struct IsStorage<
  T,
  typename std::enable_if<
    std::is_base_of<
      StorageBase<typename T::value_type, T>,
      T>::value>::type>
  : public std::true_type { };

}  // namespaces

#include <thpp/StorageBase-inl.h>

#endif /* THPP_STORAGEBASE_H_ */
