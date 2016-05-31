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

#include <thpp/cuda/CudaIOBuf.h>
#include <thpp/cuda/State.h>

namespace thpp {

namespace {

void freeCudaIOBuf(void* ptr, void* userData) {
  cuda::check(cudaFree(ptr));
}

}  // namespace

folly::IOBuf createCudaIOBuf(uint64_t capacity, int device) {
  cuda::DeviceGuard guard;
  if (device != -1) {
    cuda::setDevice(device);
  }

  void* ptr;
  cuda::check(cudaMalloc(&ptr, capacity));

  return folly::IOBuf(folly::IOBuf::TAKE_OWNERSHIP,
                      ptr, capacity, 0 /* initial length */,
                      freeCudaIOBuf);
}

}  // namespaces
