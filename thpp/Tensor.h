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
#define THPP_TENSOR_H_

#include <iostream>
#include <string>
#include <vector>

#include <thpp/Storage.h>
#include <thpp/detail/Tensor.h>
#include <thpp/if/gen-cpp2/Tensor_types.h>
#include <folly/Range.h>
#include <folly/io/IOBuf.h>

namespace thpp {

struct TensorInvalid {};
struct TensorMustAlias {};

/**
 * A Tensor wraps a pointer to a THTensor, and as such it has reference-counted
 * pointer semantics. Unlike shared_ptr, the reference count is NOT atomic, and
 * so Tensor objects may not be shared between threads.
 *
 * There are two levels of sharing (sigh): Tensor objects may alias each other
 * (they point to the same object underneath), and tensor objects that don't
 * alias each other may share storage.
 *
 * Assignment (operator=) assigns to the pointed-to object. If two tensors
 * alias each other, they will always behave identically (they are the same
 * object) and so will continue to alias each other after one is
 * assigned to.
 *
 * The only way to break aliasing is through destruction.
 *
 * It is recommended that you only use aliasing when interfacing with Lua
 * (when creating a Tensor that wraps an existing THTensor*, aka an existing
 * Lua object, and you want metadata modifications in your Tensor to reflect
 * in the underlying Lua object).
 *
 * Tensors may also share memory with other tensors. Operations that
 * manipulate metadata (select, transpose, etc) will make source and
 * destination tensors share memory. To ensure you have a unique copy, use
 * force(UNIQUE) (or set UNIQUE in the optional cloneMode argument to the copy
 * and move constructors).
 *
 * After metadata manipulation, the resulting tensor might not be stored
 * in the usual row-major order in memory. If you need a contiguous
 * representation, use force(CONTIGUOUS) (or set CONTIGUOUS in the optional
 * argument to the copy and move constructors). Note that this may break
 * the memory sharing (it will likely create a UNIQUE copy as well).
 */
template <class T>
class Tensor {
 private:
  template <class U> friend class Tensor;
  typedef detail::TensorOps<T> Ops;
 public:
  typedef typename Ops::type THType;
  typedef Storage<T> StorageType;
  typedef T value_type;
  typedef typename Ops::accurate_type accurate_type;

  // Default constructor; construct an empty, zero-dimensional Tensor.
  Tensor();

  explicit Tensor(TensorInvalid);

  Tensor(StorageType storage, long storageOffset,
         LongStorage sizes, LongStorage strides = LongStorage());

  // Constructors from a list of sizes and a list of strides.
  // If specified, the list of strides must have the same size as the
  // list of sizes.
  explicit Tensor(LongStorage sizes, LongStorage strides = LongStorage());
  explicit Tensor(LongRange sizes, LongRange strides = LongRange());
  explicit Tensor(const std::vector<long>& sizes,
                  const std::vector<long>& strides = std::vector<long>());
  explicit Tensor(
      std::initializer_list<long> sizes,
      std::initializer_list<long> strides = std::initializer_list<long>());

  // Deserialize from Thrift. Throws if wrong type.
  explicit Tensor(ThriftTensor&& thriftTensor);

  // Destructor
  ~Tensor();

  // Tensor mode. Bitwise OR of:
  // UNIQUE: this tensor is unique and does not share storage with any
  //         other tensor.
  // CONTIGUOUS:  this tensor is contiguous in row-major (that is, C) order
  enum Mode : unsigned {
    UNIQUE = 1U << 0,
    CONTIGUOUS = 1U << 1,
  };

  // Alias other.
  explicit Tensor(THType* other, TensorMustAlias) noexcept;

  // Do not alias other, create separate object (with separate metadata);
  // might still share data with other, unless UNIQUE requested in
  // cloneMode.
  explicit Tensor(const THType* other, unsigned cloneMode = 0);

  // Take ownership of the given THType*. This is the reverse
  // operation of moveAsTH().
  explicit Tensor(THType*&& other);

  // Move/copy constructors. Enforce requested mode.
  /* implicit */ Tensor(Tensor&& other) noexcept;
  Tensor(Tensor&& other, unsigned cloneMode);
  /* implicit */ Tensor(const Tensor& other, unsigned cloneMode = 0);

  // Move/copy assignment operators. Will share memory with "other".
  Tensor& operator=(Tensor&& other) noexcept;
  Tensor& operator=(const Tensor& other);

  // Take ownership of the given THType*. This is the reverse
  // operation of moveAsTH().
  Tensor& operator=(THType*&& other);

  // Serialize to Thrift. Non-const because the resulting ThriftTensor
  // may share memory with *this, and so changes in the ThriftTensor may
  // affect changes in *this.
  void serialize(ThriftTensor& out,
                 ThriftTensorEndianness endianness =
                    ThriftTensorEndianness::NATIVE,
                 bool mayShare = true);

  // const version that won't share, but will always copy
  void serializeUnshared(ThriftTensor& out,
                         ThriftTensorEndianness endianness =
                            ThriftTensorEndianness::NATIVE) const {
    const_cast<Tensor*>(this)->serialize(out, endianness, false);
  }

  // Get a pointer to the underlying TH object; *this releases ownership
  // of that object.
  THType* moveAsTH();

  // Force the tensor to have a certain mode. May copy data.
  void force(unsigned mode);

  // Return current mode.
  unsigned mode() const {
    return mode(t_);
  }
  static unsigned mode(const THType* th) {
    return (isUnique(th) ? UNIQUE : 0) | (isContiguous(th) ? CONTIGUOUS : 0);
  }

  // Is this tensor unique?
  bool isUnique() const {
    return isUnique(t_);
  }
  static bool isUnique(const THType* th);

  // Is this tensor contiguous?
  bool isContiguous() const {
    return isContiguous(t_);
  }
  static bool isContiguous(const THType* th);

  /// Compares two tensors for exact equality
  bool isExactlyEqual(const Tensor<T>& other) const;

  /// Compares two tensors for approximate equality. For integral
  /// types, forwards to isExactlyEqual; for floating point types,
  /// uses the given relativeError to determine equality.
  bool isApproximatelyEqual(const Tensor<T>& other,
                            float relativeError = 0.0001f) const;

  // Return number of elements.
  long size() const;

  // Return list of sizes.
  LongRange sizes() const;

  // Return list of strides.
  LongRange strides() const;

  // Return number of dimensions.
  int ndims() const { return t_->nDimension; }

  // Return size along dimension dim.
  long size(int dim) const { return sizes().at(dim); }

  // Return stride along dimension dim.
  long stride(int dim) const { return strides().at(dim); }

  // Narrow the tensor along a given dimension; the dimension dim is narrowed
  // to [firstIndex, firstIndex + size)
  void narrow(const Tensor& src, int dim, long firstIndex, long size);
  void narrow(int dim, long firstIndex, long size) {
    narrow(*this, dim, firstIndex, size);
  }

  // Select one slice of the tensor along a given dimension. The tensor's
  // dimensionality is reduced by 1.
  void select(const Tensor& src, int dim, long index);
  void select(int dim, long index) { select(*this, dim, index); }

  // Transpose two dimensions.
  void transpose(const Tensor& src, int dim1, int dim2);
  void transpose(int dim1, int dim2) { transpose(*this, dim1, dim2); }

  // Full transpose (reverse the order of axes)
  void transpose(const Tensor& src) { *this = src; transpose(); }
  void transpose();

  // Unfold dimension dim along two dimensions: slices of size 'size' (with
  // given step between slices) are unfolded among a new dimension that is
  // added.
  // See http://torch5.sourceforge.net/manual/torch/index-6-8-3.html
  void unfold(const Tensor& src, int dim, long size, long step);
  void unfold(int dim, long size, long step) { unfold(*this, dim, size, step); }

  // Squeeze: remove all 1-sized dimensions.
  void squeeze(const Tensor& src);
  void squeeze() { squeeze(*this); }

  // Squeeze: remove one dimension if it is 1-sized.
  void squeeze(const Tensor& src, int dim);
  void squeeze(int dim) { squeeze(*this, dim); }

  void resize(
      std::initializer_list<long> newSizes,
      std::initializer_list<long> newStrides = std::initializer_list<long>());
  void resize(LongStorage newSizes, LongStorage newStrides = LongStorage());
  void resize(LongRange newSizes, LongRange newStrides = LongRange());
  void resizeAs(const Tensor& src);

  StorageType storage();
  long storageOffset() const;

  // Fill with one value.
  void fill(T value);

  // Fill with zeros.
  void zero();

  // Copy from another tensor
  template <class U>
  void copy(const Tensor<U>& src);

  // Given a ByteTensor of the exact same dimensionality as *this, whose
  // values are 0 or 1, set elements of *this to value iff the corresponding
  // elements in mask are 1.
  void maskedFill(const ByteTensor& mask, T value);

  // Copy corresponding elements of src to *this iff the corresponding elements
  // in mask are 1
  void maskedCopy(const ByteTensor& mask, const Tensor& src);

  // Select elements from *this iff the corresponding elements in mask
  // are 1. Returns a 1d tensor with one entry for each selected element.
  Tensor maskedSelect(const ByteTensor& mask) const;

  // Select along dimension dim, copying only indices from index.
  // Returns a tensor with matching dimensionality, but only index.size()
  // elements along dimension dim.
  Tensor indexSelect(int dim, const LongTensor& index) const;

  // Fill along dimension dim, setting entries corresponding to indices
  // from index to val.
  void indexFill(int dim, const LongTensor& index, T val);

  // Dot product
  accurate_type dot(const Tensor& other) const;

  // Minimum value among all elements
  T minall() const;

  // Maximum value among all elements
  T maxall() const;

  // Sum of all elements
  accurate_type sumall() const;

  // Add a value to each element in the tensor
  void add(const Tensor& src, T value);
  void add(T value) { add(*this, value); }

  // Multiply each element in the tensor by a value
  void mul(const Tensor& src, T value);
  void mul(T value) { mul(*this, value); }

  // Divide each element in the tensor by a value
  void div(const Tensor& src, T value);
  void div(T value) { div(*this, value); }

  // *this = a + value * b
  void cadd(const Tensor& a, T value, const Tensor& b);
  void cadd(T value, const Tensor& b) { cadd(*this, value, b); }

  // *this = a .* b
  void cmul(const Tensor& a, const Tensor& b);
  void cmul(const Tensor& b) { cmul(*this, b); }

  // *this = a ./ b
  void cdiv(const Tensor& a, const Tensor& b);
  void cdiv(const Tensor& b) { cmul(*this, b); }

  // *this = a + value * (b .* c)
  void addcmul(const Tensor& a, T value, const Tensor& b, const Tensor& c);
  void addcmul(T value, const Tensor& b, const Tensor& c) {
    addcmul(*this, value, b, c);
  }

  // *this = a + value * (b ./ c)
  void addcdiv(const Tensor& a, T value, const Tensor& b, const Tensor& c);
  void addcdiv(T value, const Tensor& b, const Tensor& c) {
    addcdiv(*this, value, b, c);
  }

  // *this = beta * t + alpha * mat * vec
  void addmv(T beta, const Tensor& t, T alpha, const Tensor& mat,
             const Tensor& vec);
  void addmv(T beta, T alpha, const Tensor& mat, const Tensor& vec) {
    addmv(beta, *this, alpha, mat, vec);
  }

  // *this = beta * t + alpha * (m1 X m2)
  void addmm(T beta, const Tensor& t, T alpha, const Tensor& m1,
             const Tensor& m2);
  void addmm(T beta, T alpha, const Tensor& m1, const Tensor& m2) {
    addmm(beta, *this, alpha, m1, m2);
  }

  // outer product
  // *this = beta * m + alpha * (v1 (X) v2)
  void addr(T beta, const Tensor& m, T alpha, const Tensor& v1,
            const Tensor& v2);
  void addr(T beta, T alpha, const Tensor& v1, const Tensor& v2) {
    addr(beta, *this, alpha, v1, v2);
  }

  // number of elements, same as size()
  long numel() const { return size(); }

  // The following functions perform operations along one dimension.
  // The returned tensors will have the same shape as *this except that they
  // have a size of 1 along dimension dim. (That is, they're not squeezed)

  // <max, argmax>
  std::pair<Tensor, LongTensor> max(int dim) const;

  // <min, argmin>
  std::pair<Tensor, LongTensor> min(int dim) const;

  // sum
  Tensor sum(int dim) const;

  // product
  Tensor prod(int dim) const;

  // cumulative sum
  Tensor cumsum(int dim) const;

  // cumulative product
  Tensor cumprod(int dim) const;

  // Element-wise sign
  Tensor sign() const;

  // Trace (must be matrix)
  accurate_type trace() const;

  // Cross product along dim (if >= 0) or along the first dimension with size
  // 3. (Yeah, it's weird like that):
  // https://github.com/torch/torch7/blob/master/doc/maths.md
  Tensor cross(const Tensor& b, int dim = -1) const;

  // TODO(tudorb): TH doesn't distinguish between a 1-element 1-dimensional
  // array (aka 1-element vector) and a scalar.
  bool isScalar() const;

  // Access the underlying data
  T* data();
  const T* data() const;

  // First element
  const T& front() const { return *data(); }
  T& front() { return *data(); }

  // Index along the first dimension
  Tensor operator[](long index) const;

  // Index along dimensions 0, 1, ..., indices.size() - 1.
  // Pass -1 as an index to keep that dimension unchanged.
  //
  // Example: given a 5-dimensional tensor foo,
  // foo[-1,2,-1,2,1] returns a 2-dimensional tensor corresponding
  // to the hyperplane that has d1=2, d3=2, d4=1 in foo.
  Tensor operator[](std::initializer_list<long> indices) const;

  // Operator to return the first element at the given index
  T& at(long idx) { return at({idx}); }
  const T& at(long idx) const { return at({idx}); }

  T& at(std::initializer_list<long> indices);
  const T& at(std::initializer_list<long> indices) const;

  // Clear the tensor.
  void clear();

  static constexpr const char* kLuaTypeName = Ops::kLuaTypeName;

  std::string str() const;

 private:
  void destroy();
  THType* mut() const { return mut(t_); }
  static THType* mut(const THType* th) { return const_cast<THType*>(th); }

  THType* t_;
};

template <class T>
constexpr const char* Tensor<T>::kLuaTypeName;

// Unary -
template <class T>
Tensor<T> operator-(const Tensor<T>& a);

// Binary operators. We don't define multiplication and division as they're
// ambiguous: do you mean pointwise? matrix multiplication? inner product?
// outer product?
template <class T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b);
template <class T>
Tensor<T>& operator+=(Tensor<T>& a, const Tensor<T>& b);

template <class T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b);
template <class T>
Tensor<T>& operator-=(Tensor<T>& a, const Tensor<T>& b);

// Multiplication / division by scalar
template <class T>
Tensor<T> operator*(const Tensor<T>& a, T b);
template <class T>
Tensor<T> operator*(T a, const Tensor<T>& b) {
  return b * a;
}
template <class T>
Tensor<T>& operator*=(Tensor<T>& a, T b);

template <class T>
Tensor<T> operator/(const Tensor<T>& a, T b);
template <class T>
Tensor<T>& operator/=(Tensor<T>& a, T b);

template <class T>
std::ostream& operator<<(std::ostream& s, const Tensor<T>& t) {
  return s << t.str();
}

}  // namespaces

#include <thpp/TensorSerialization-inl.h>
#include <thpp/Tensor-inl.h>

#endif /* THPP_TENSOR_H_ */
