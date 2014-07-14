#!/bin/bash -e
#
#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#
set -o pipefail

if [[ ! -r ./Tensor.h ]]; then
  echo "Please run from the thpp subdirectory." >&2
  exit 1
fi

rm -rf gtest-1.7.0 gtest-1.7.0.zip
curl -JLO https://googletest.googlecode.com/files/gtest-1.7.0.zip
if [[ $(sha1sum -b gtest-1.7.0.zip | cut -d' ' -f1) != \
      'f85f6d2481e2c6c4a18539e391aa4ea8ab0394af' ]]; then
  echo "Invalid gtest-1.7.0.zip file" >&2
  exit 1
fi
unzip gtest-1.7.0.zip

# Build in a separate directory
mkdir -p build
cd build

# Configure
cmake ..

# Make
make

# Run tests
ctest

# Install
sudo make install
