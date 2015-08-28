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

sudo rm -rf /usr/local/include/folly
sudo rm -rf /usr/local/include/wangle

sudo rm -rf /usr/local/lib/libfolly*
sudo rm -rf /usr/local/lib/libwangle*


cd /tmp

dir=$(mktemp --tmpdir -d follythrift-build.XXXXXX)

cd $dir

#git clone -b v0.54.0  --depth 1 https://github.com/facebook/folly.git
git clone -b v0.30.0  --depth 1 https://github.com/facebook/fbthrift.git
git clone -b v0.12.0  --depth 1 https://github.com/facebook/wangle

git clone https://github.com/facebook/folly.git
cd folly
git reset --hard 0fdbb61ecd5679f0cd2bf13f867e9b72212ec371
cd ..
# git clone  --depth 1 https://github.com/facebook/fbthrift.git
# git clone  --depth 1 https://github.com/facebook/wangle

echo
echo Building folly
echo

cd $dir/folly/folly
autoreconf -ivf
./configure
make
sudo make install
sudo ldconfig # reload the lib paths after freshly installed folly. fbthrift needs it.


echo
echo Building wangle
echo

cd $dir/wangle/wangle

cmake .
make
sudo make install
sudo ldconfig # reload the lib paths after freshly installed wangle. fbthrift needs it.

echo
echo Building fbthrift
echo

cd $dir/fbthrift/thrift
autoreconf -ivf
./configure
make
sudo make install
