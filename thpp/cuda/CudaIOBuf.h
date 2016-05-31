/*
 * Copyright 2016 Facebook, Inc.
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

#include <folly/io/IOBuf.h>

namespace thpp {

// Create an IOBuf of the given capacity. The memory is allocated on the
// requested CUDA device. (-1 = current device)
// Just like IOBuf::CREATE, the buffer is created empty (the initial length
// is 0). Use IOBuf::append() to increase the length.
folly::IOBuf createCudaIOBuf(uint64_t capacity, int device = -1);

}  // namespaces
