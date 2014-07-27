/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <thpp/detail/Tensor.h>

#define THPP_INCLUDE_TENSOR_DEFS
#include "thpp/detail/TensorDefsGeneric.h"
#include <THGenerateAllTypes.h>
#undef THPP_INCLUDE_TENSOR_DEFS
