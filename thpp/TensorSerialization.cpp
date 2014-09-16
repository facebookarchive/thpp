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
#include <folly/Format.h>

namespace thpp {
namespace detail {

namespace {

std::unique_ptr<folly::IOBuf> partialCloneOne(const folly::IOBuf& buf,
                                              uint64_t offset,
                                              uint64_t length) {
  DCHECK_LE(offset + length, buf.length());
  auto cloned = buf.cloneOne();
  cloned->trimStart(offset);
  cloned->trimEnd(length - cloned->length());
  return cloned;
}

}  // namespace

void serialize(
    ThriftTensor& out,
    LongRange sizes,
    LongRange strides,
    folly::IOBuf&& data,
    ThriftTensorDataType dtype,
    size_t elementSize,
    ThriftTensorEndianness endianness,
    bool mayShare) {
  DCHECK(!data.isChained());
  if (endianness == ThriftTensorEndianness::NATIVE) {
    endianness = gMachineEndianness;
  } else {
    CHECK(endianness == gMachineEndianness)
      << "Non-native endianness not yet implemented";
  }

  int ndims = sizes.size();
  uint64_t dataSize = 1;
  uint64_t contiguousSize = 1;
  int firstContiguousDim = ndims - 1;

  if (!strides.empty()) {
    DCHECK_EQ(strides.size(), ndims);
    while (firstContiguousDim >= 0) {
      if (strides[firstContiguousDim] != contiguousSize) {
        break;
      }
      contiguousSize *= sizes[firstContiguousDim];
      --firstContiguousDim;
    }
    ++firstContiguousDim;
    dataSize = contiguousSize;
    for (int i = 0; i < firstContiguousDim; ++i) {
      dataSize *= sizes[i];
    }
  } else {
    for (auto s : sizes) {
      dataSize *= s;
    }
    contiguousSize = dataSize;
    firstContiguousDim = 0;
  }

  // Dimensions from firstContiguousDim till the last form a contiguous range
  // of contiguousSize elements; we'll copy / clone that in one go rather
  // than iterating through all elements.

  // We want bytes.
  dataSize *= elementSize;
  contiguousSize *= elementSize;

  DCHECK_LE(contiguousSize, dataSize);

  out.dataType = dtype;
  out.endianness = endianness;
  out.sizes.assign(sizes.begin(), sizes.end());

  if (ndims == 0) {
    // Empty tensor, nothing to do.
    out.data = folly::IOBuf();
    data = folly::IOBuf();
    return;
  }

  if (firstContiguousDim == 0 && mayShare) {
    // We're done.
    out.data = std::move(data);
    DCHECK_GE(out.data.length(), dataSize);
    out.data.trimEnd(out.data.length() - dataSize);
    return;
  }

  // We have to do this the hard way...
  folly::IOBufQueue outQueue;

  // If the contiguous chunk size is >= kMinCloneSize, we clone rather
  // than copying
  static constexpr uint64_t kMinCloneSize = 4 << 10;

  // Don't allocate huge contiguous buffers.
  // jemalloc defers to mmap() for buffers of 4MiB or more.
  static constexpr uint64_t kMaxBlockSize = 2 << 20;
  folly::io::QueueAppender appender(&outQueue,
                                    std::min(dataSize, kMaxBlockSize));

  std::vector<uint64_t> counter;
  counter.resize(firstContiguousDim);
  int idx = firstContiguousDim;
  const uint8_t* src = data.data();
  while (idx >= 0) {
    if (idx == firstContiguousDim) {
      if (contiguousSize >= kMinCloneSize) {
        appender.insert(partialCloneOne(data, src - data.data(),
                                        contiguousSize));
      } else {
        appender.push(src, contiguousSize);
      }
      --idx;
      continue;
    }
    src += strides[idx] * elementSize;
    if (++counter[idx] == sizes[idx]) {
      src -= sizes[idx] * strides[idx] * elementSize;
      counter[idx] = 0;
      --idx;
    } else {
      idx = firstContiguousDim;
    }
  }

  outQueue.move()->cloneInto(out.data);
}

template folly::IOBuf deserialize(ThriftTensor& in,
                                  ThriftTensorDataType dtype);

}}  // namespaces
