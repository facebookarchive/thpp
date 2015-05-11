/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <thpp/cuda/Storage.h>
#include <cuda_runtime.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp { namespace cuda { namespace test {

TEST(Storage, Simple) {
  constexpr size_t n = 100;
  CudaStorage<float> storage;
  storage.resizeUninitialized(n);
  EXPECT_EQ(cudaSuccess, cudaMemset(storage.data(), 42, n));
  char buf[n];
  EXPECT_EQ(cudaSuccess, cudaMemcpy(buf, storage.data(), n,
                                    cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(42, buf[i]);
  }
}

}}}  // namespaces
