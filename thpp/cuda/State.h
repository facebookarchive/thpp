/**
 * Copyright 2015 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#ifndef THPP_CUDA_STATE_H_
#define THPP_CUDA_STATE_H_

#include <THC.h>
#include <folly/Exception.h>
#include <folly/ThreadLocal.h>

namespace thpp {

// TODO(tudorb): Nice C++ interface around this.

// By default, we use one THCState per thread, created on first use.
//
// You may associate the current thread with a different THCState object if
// you wish (for example, so that Lua code using cutorch will use the same
// state)

namespace detail {
extern folly::ThreadLocalPtr<THCState> gCurrentTHCState;
THCState* doSetDefaultTHCState();
}  // namespace

inline THCState* getTHCState() {
  auto state = detail::gCurrentTHCState.get();
  return state ? state : detail::doSetDefaultTHCState();
}

inline void setTHCState(THCState* state) {
  DCHECK(state);
  detail::gCurrentTHCState.reset(state);
}

inline void setDefaultTHCState() {
  detail::doSetDefaultTHCState();
}

namespace cuda {

inline void check(cudaError_t err) {
  folly::throwOnFail<std::runtime_error>(
      err == cudaSuccess,
      folly::to<std::string>("CUDA error ", err));
}

inline int getDevice() {
  int device;
  check(cudaGetDevice(&device));
  return device;
}

inline void setDevice(int dev) {
  check(cudaSetDevice(dev));
}

class DeviceGuard {
 public:
  explicit DeviceGuard() : device_(getDevice()) { }
  ~DeviceGuard() { setDevice(device_); }

 private:
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&& other) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard& operator=(DeviceGuard&& other) = delete;

  int device_;
};

}  // namespace cuda

}  // namespaces

#endif /* THPP_CUDA_STATE_H_ */
