/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <thpp/cuda/State.h>
#include <folly/ThreadLocal.h>

namespace thpp {

namespace {

class THCStateHolder {
 public:
  THCStateHolder();
  ~THCStateHolder();

  THCState* state() { return &state_; }

 private:
  void* tmp_;
  THCState state_;
};

THCStateHolder::THCStateHolder() {
  memset(&state_, 0, sizeof(THCState));
  THCudaInit(&state_);
  // TODO(tudorb): There must be a better way of doing this. We need to check
  // that we're not unloading the driver (running during process exit); we
  // do that by checking at destruction time if freeing fails with a
  // specific error.
  CHECK_EQ(cudaMalloc(&tmp_, 1), cudaSuccess);
}

THCStateHolder::~THCStateHolder() {
  cudaError_t err = cudaFree(tmp_);
  if (err == cudaSuccess) {
    THCudaShutdown(&state_);
  } else {
    CHECK_EQ(err, cudaErrorCudartUnloading);
  }
}

folly::ThreadLocal<THCStateHolder> gDefaultTHCState;

}  // namespace

namespace detail {
folly::ThreadLocal<THCState*> gCurrentTHCState;
}  // namespace detail

void setDefaultTHCState() {
  setTHCState(gDefaultTHCState->state());
}
}  // namespaces
