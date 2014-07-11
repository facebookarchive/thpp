/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_DETAIL_TENSOR_H_
#define THPP_DETAIL_TENSOR_H_

#include <TH.h>
#include <thpp/detail/Storage.h>
#include <folly/Preprocessor.h>
#include <folly/Range.h>

namespace thpp {

template <class T> class Tensor;

namespace detail {
template <class T> struct TensorOps;
}  // namespace detail

#include <thpp/detail/TensorGeneric.h>
#include <THGenerateAllTypes.h>

}  // namespaces

#endif /* THPP_DETAIL_TENSOR_H_ */
