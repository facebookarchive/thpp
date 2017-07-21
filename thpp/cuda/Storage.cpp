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

#include <thpp/cuda/Storage.h>

namespace thpp {

namespace detail {

CudaIOBufAllocator::CudaIOBufAllocator(folly::IOBuf&& iob)
  : iob_(std::move(iob)) { }

cudaError_t CudaIOBufAllocator::malloc(
    void* /*ctx*/,
    void** /*ptr*/,
    size_t /*size*/,
    cudaStream_t /*stream*/) {
  LOG(FATAL) << "CudaIOBufAllocator::malloc should never be called";
  return cudaSuccess;  // not reached
}

cudaError_t CudaIOBufAllocator::realloc(
    void* /*ctx*/,
    void** /*ptr*/,
    size_t /*oldSize*/,
    size_t /*newSize*/,
    cudaStream_t /*stream*/) {
  LOG(FATAL) << "CudaIOBufAllocator::realloc should never be called";
  return cudaSuccess;  // not reached
}

cudaError_t CudaIOBufAllocator::free(void* /*stat*/, void* ptr) {
  CHECK_EQ(ptr, iob_.writableData());
  delete this;
  return cudaSuccess;
}

}  // namespace detail

}  // namespaces
