/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <thpp/Tensor.h>

#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <folly/io/IOBuf.h>
#include <folly/io/TypedIOBuf.h>

namespace thpp {
namespace test {

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

  ThriftTensor serialized;
  src.serialize(serialized);

  src.force(Tensor<float>::CONTIGUOUS);
  Tensor<float> deserialized(std::move(serialized));
  EXPECT_TRUE(src.sizes() == deserialized.sizes());
  EXPECT_TRUE(src.strides() == deserialized.strides());
  EXPECT_EQ(0, memcmp(src.data(), deserialized.data(),
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

TEST(SerializationTest, SmallerThanStorage) {
  Tensor<long> t({10L});
  for (long i = 0; i < 10L; ++i) {
    t.at(i) = i;
  }
  t.resize(LongStorage{5L});

  ThriftTensor out;
  t.serialize(out);

  Tensor<long> t1(std::move(out));
  EXPECT_EQ(1, t1.ndims());
  EXPECT_EQ(5, t1.size());
  for (long i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(i, t1.at(i));
  }
}

TEST(SerializationTest, StorageOffset) {
  Tensor<long> t({10L});
  for (long i = 0; i < 10L; ++i) {
    t.at(i) = i;
  }

  t.narrow(0, 1, 5);
  EXPECT_EQ(5, t.size());
  for (long i = 0; i < t.size(); ++i) {
    EXPECT_EQ(i + 1, t.at(i));
  }

  ThriftTensor out;
  t.serialize(out);

  Tensor<long> t1(std::move(out));
  EXPECT_EQ(1, t1.ndims());
  EXPECT_EQ(5, t1.size());
  for (long i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(i + 1, t1.at(i));
  }
}

TEST(SerializatioNTest, Empty0d) {
  Tensor<long> t;
  EXPECT_EQ(0, t.ndims());
  EXPECT_EQ(0, t.size());

  ThriftTensor out;
  t.serialize(out);

  Tensor<long> t1(std::move(out));
  EXPECT_EQ(0, t1.ndims());
  EXPECT_EQ(0, t1.size());
}

constexpr ThriftTensorEndianness nativeEndianness =
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    ThriftTensorEndianness::LITTLE;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    ThriftTensorEndianness::BIG;
#else
# error Weird endianness!
#endif

TEST(SerializationTest, IOBufStorage) {
  ThriftTensor serialized;
  serialized.dataType = ThriftTensorDataType::FLOAT;
  serialized.endianness = nativeEndianness;
  constexpr size_t n = 10;
  serialized.sizes.push_back(n);
  serialized.data = folly::IOBuf(folly::IOBuf::CREATE, n * sizeof(float));
  folly::TypedIOBuf<float> buf(&serialized.data);
  buf.append(n);
  for (int i = 0; i < n; ++i) {
    buf[i] = float(i);
  }
  const void* ptr = serialized.data.data();

  Tensor<float> deserialized(std::move(serialized));
  EXPECT_TRUE(deserialized.data() == ptr);  // actually sharing memory
  EXPECT_EQ(1, deserialized.sizes().size());
  EXPECT_EQ(n, deserialized.sizes()[0]);
  EXPECT_EQ(1, deserialized.strides().size());
  EXPECT_EQ(1, deserialized.strides()[0]);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(float(i), deserialized[i].front());
  }

  // Now resize to something big enough that it won't share memory any more
  deserialized.resize(LongStorage({1 << 20}));
  EXPECT_FALSE(deserialized.data() == ptr);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(float(i), deserialized[i].front());
  }
}

TEST(SerializationTest, ThriftStorageShare) {
  Storage<long> storage(size_t(1000), long(42));
  ThriftStorage serialized;
  storage.serialize(serialized);
  auto ptr = storage.data();
  auto size = storage.size();
  EXPECT_TRUE(static_cast<const void*>(serialized.data.data()) == ptr);
  storage = Storage<long>();  // lose the reference from storage

  Storage<long> deserialized(std::move(serialized));
  EXPECT_EQ(size, deserialized.size());
  EXPECT_TRUE(deserialized.data() == ptr);  // shares memory
}

TEST(SerializationTest, ThriftStorageNoShare) {
  Storage<long> storage(size_t(1000), long(42));
  ThriftStorage serialized;
  storage.serialize(serialized);
  auto ptr = storage.data();
  EXPECT_TRUE(static_cast<const void*>(serialized.data.data()) == ptr);

  Storage<long> deserialized(std::move(serialized));
  EXPECT_EQ(storage.size(), deserialized.size());
  EXPECT_FALSE(deserialized.data() == storage.data());  // doesn't share
}

TEST(SerializationTest, ThriftStorageRefs) {
  folly::IOBuf buf2;
  Storage<long> s1({1000L});
  folly::IOBuf buf1 = s1.getIOBuf();
  buf2 = s1.getIOBuf();
}

}}  // namespaces
