/*
 *  Copyright (c) 2014, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */

#include <thpp/Storage.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace thpp {
namespace test {

typedef Storage<float> FloatStorage;
TEST(Storage, Simple) {
  FloatStorage s({2, 3, 4});
  EXPECT_EQ(3, s.size());
  EXPECT_EQ(2, s.at(0));
  EXPECT_EQ(3, s.at(1));
  EXPECT_EQ(4, s.at(2));
}

void* g_ptr = nullptr;
int g_context = 0;
int g_nMalloc = 0;
int g_nFree = 0;

struct TestContext {
  int nMalloc, nFree;
  TestContext() : nMalloc(0), nFree(0) {}
};

void* test_malloc(void* ctx, long size) {
  auto myCtx = (TestContext*) ctx;
  myCtx->nMalloc++;
  return malloc(4*size);
}
void* test_realloc(void* /*ctx*/, void* /*ptr*/, long /*size*/) {
  ADD_FAILURE() << "realloc should not be called";
  return nullptr;
}
void test_free(void* ctx, void* /*ptr*/) {
  auto myCtx = (TestContext*) ctx;
  myCtx->nFree++;
}

TEST(Storage, CustomAllocator) {
  THAllocator testAlloc = {
    &test_malloc, &test_realloc, &test_free
  };

  // 1. delete the storage first, then the IO buf
  auto ctx = TestContext();
  auto thStorage = THFloatStorage_newWithAllocator(42, &testAlloc, &ctx);
  EXPECT_EQ(ctx.nMalloc, 1);
  {
    auto storage = FloatStorage(thStorage);
    g_ptr = thStorage->data;
    auto buf = storage.getIOBuf();
    THFloatStorage_free(thStorage);
    EXPECT_EQ(ctx.nFree, 0);
  }
  EXPECT_EQ(ctx.nMalloc, 1);
  EXPECT_EQ(ctx.nFree, 1);

  // 2. delete the IO buf first, then the storage
  ctx = TestContext();
  thStorage = THFloatStorage_newWithAllocator(42, &testAlloc, &ctx);
  EXPECT_EQ(ctx.nMalloc, 1);
  {
    auto storage = FloatStorage(thStorage);
    g_ptr = thStorage->data;
    auto buf = storage.getIOBuf();
  }
  EXPECT_EQ(ctx.nFree, 0);
  THFloatStorage_free(thStorage);
  EXPECT_EQ(ctx.nFree, 1);
  EXPECT_EQ(ctx.nMalloc, 1);

}

}}  // namespaces
