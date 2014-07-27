/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_INCLUDE_TENSOR_DEFS
#error This file may only be included from TensorDefs.cpp
#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "thpp/detail/TensorDefsGeneric.h"
#else

namespace thpp { namespace detail {

constexpr const char* TensorOps<real>::kLuaTypeName;

}}  // namespaces

#endif /* TH_GENERIC_FILE */
