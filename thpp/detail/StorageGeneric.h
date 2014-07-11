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
#define TH_GENERIC_FILE "thpp/detail/StorageGeneric.h"
#else

typedef Range<const real*> TH_CONCAT_2(Real, Range);
typedef Range<real*> TH_CONCAT_3(Mutable, Real, Range);
typedef Storage<real> TH_CONCAT_2(Real, Storage);

namespace detail {
template <> struct StorageOps<real> {
  typedef THStorage type;

  static THStorage* _newWithSize(long size) {
    return THStorage_(newWithSize)(size);
  }
  static THStorage* _newWithData(real* data, long size) {
    return THStorage_(newWithData)(data, size);
  }
  static THStorage* _newWithDataAndAllocator(real* data, long size,
                                             THAllocator* allocator,
                                             void* allocatorContext) {
    return THStorage_(newWithDataAndAllocator)(data, size,
                                               allocator, allocatorContext);
  }
  static void _setFlag(THStorage* storage, const char flag) {
    return THStorage_(setFlag)(storage, flag);
  }
  static void _clearFlag(THStorage* storage, const char flag) {
    return THStorage_(clearFlag)(storage, flag);
  }
  static void _retain(THStorage* storage) {
    return THStorage_(retain)(storage);
  }
  static void _free(THStorage* storage) {
    return THStorage_(free)(storage);
  }
  static void _resize(THStorage* storage, long size) {
    return THStorage_(resize)(storage, size);
  }

#define S1(X) #X
#define S(X) S1(X)
  static constexpr const char* kLuaTypeName = "torch."
    S(TH_CONCAT_2(Real, Storage));
#undef S
#undef S1
};
}  // namespace detail

#endif
