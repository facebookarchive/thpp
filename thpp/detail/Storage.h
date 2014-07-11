/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#ifndef THPP_DETAIL_STORAGE_H_
#define THPP_DETAIL_STORAGE_H_

#include <THStorage.h>
#include <folly/Range.h>

namespace thpp {

using folly::Range;
template <class T> class Storage;

namespace detail {
template <class T> struct StorageOps;
}  // namespace detail

#include <thpp/detail/StorageGeneric.h>
#include <THGenerateAllTypes.h>

}  // namespaces

#endif /* THPP_DETAIL_STORAGE_H_ */
