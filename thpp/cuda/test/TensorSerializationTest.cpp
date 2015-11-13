/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <thpp/cuda/Tensor.h>

#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp { namespace test {

Tensor<float> createTensor(std::vector<long> sizes,
                           std::vector<long> strides = {}) {
  Tensor<float> tensor(LongRange(sizes.data(), sizes.size()),
                       LongRange(strides.data(), strides.size()));

  std::vector<long> counter;
  counter.resize(sizes.size());
  int idx = counter.size();
  float val = 0;
  while (idx >= 0) {
    if (idx == counter.size()) {
      Tensor<float> t(tensor);
      for (int i = counter.size() - 1; i >= 0; --i) {
        t.select(i, counter[i]);
      }
      t.front() = val++;
      --idx;
      continue;
    }
    if (++counter[idx] == sizes[idx]) {
      counter[idx] = 0;
      --idx;
    } else {
      idx = counter.size();
    }
  }

  return tensor;
}

void runTest(std::vector<long> sizes,
             std::vector<long> strides = {}) {
  Tensor<float> src = createTensor(sizes, strides);
  CudaTensor<float> srcCuda;
  srcCuda.resize(src.sizes(), src.strides());
  srcCuda.copy(src);

  ThriftTensor serialized;
  srcCuda.serialize(serialized);

  src.force(Tensor<float>::CONTIGUOUS);

  CudaTensor<float> deserializedCuda(std::move(serialized));
  auto deserialized = deserializedCuda.toCPU();

  EXPECT_TRUE(src.sizes() == deserialized->sizes());
  EXPECT_TRUE(src.strides() == deserialized->strides());
  EXPECT_EQ(0, memcmp(src.data(), deserialized->data(),
                      sizeof(float) * src.size()));
}

TEST(SerializationTest, Simple) {
  runTest({1});
  runTest({2});
  runTest({2}, {1});
  runTest({2}, {2});
  runTest({2}, {200});
  runTest({20, 10});
  runTest({20, 10}, {10, 1});
  runTest({20, 10}, {40, 4});
  runTest({20, 10}, {400, 4});
  runTest({20, 10}, {0, 1});
  runTest({20, 10}, {0, 0});
  runTest({20, 30, 10});
  runTest({20, 30, 10}, {300, 10, 1});
  runTest({20, 30, 10}, {10, 200, 1});
  runTest({20, 30, 10}, {1, 20, 600});
  runTest({20, 30}, {8192 * 30, 8192});
}

}}  // namespaces
