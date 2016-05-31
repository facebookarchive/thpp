/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <thpp/cuda/CudaIOBuf.h>
#include <thpp/cuda/Storage.h>
#include <cuda_runtime.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp { namespace cuda { namespace test {

namespace {

void testStorage(CudaStorage<float>& storage) {
  auto byteSize = storage.size() * sizeof(float);

  EXPECT_EQ(cudaSuccess, cudaMemset(storage.data(), 42, byteSize));
  char buf[byteSize];
  memset(buf, 0, byteSize);

  EXPECT_EQ(cudaSuccess, cudaMemcpy(buf, storage.data(), byteSize,
                                    cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < byteSize; ++i) {
    EXPECT_EQ(42, buf[i]);
  }
}

}  // namespace

TEST(Storage, Simple) {
  constexpr size_t n = 100;
  CudaStorage<float> storage;
  storage.resizeUninitialized(n);
  EXPECT_EQ(n, storage.size());
  testStorage(storage);
}

TEST(Storage, CudaIOBufEmpty) {
  constexpr size_t n = 100;
  CudaStorage<float> storage(createCudaIOBuf(n * sizeof(float)),
                             SHARE_IOBUF_MANAGED,
                             false /* resizable */);
  EXPECT_EQ(0, storage.size());  // buffer is created empty!
}

TEST(Storage, CudaIOBuf) {
  constexpr size_t n = 100;
  auto buf = createCudaIOBuf(n * sizeof(float));
  buf.append(n * sizeof(float));
  CudaStorage<float> storage(std::move(buf), SHARE_IOBUF_MANAGED, false);
  EXPECT_EQ(n, storage.size());
  testStorage(storage);
}

}}}  // namespaces
