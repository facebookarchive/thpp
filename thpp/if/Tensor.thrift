// Copyright 2014 Facebook

namespace cpp2 thpp

typedef binary (cpp2.type = "folly::IOBuf") IOBuf

enum ThriftTensorDataType {
  BYTE = 1,
  INT32 = 2,
  INT64 = 3,
  FLOAT = 4,   // IEEE-754 "binary32"
  DOUBLE = 5,  // IEEE-754 "binary64"
}

enum ThriftTensorEndianness {
  LITTLE = 1,
  BIG = 2,

  // Native is never used on the wire, just as argument to serialization /
  // deserialization functions
  NATIVE = 3,
}

struct ThriftTensor {
  1: required ThriftTensorDataType dataType,
  2: required ThriftTensorEndianness endianness,
  3: required list<i64> sizes,
  4: IOBuf data,
}

struct ThriftStorage {
  1: required ThriftTensorDataType dataType,
  2: required ThriftTensorEndianness endianness,
  3: IOBuf data,
}
