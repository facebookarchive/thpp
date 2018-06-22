#pragma once

namespace thpp {
template<typename T> class Tensor;
template<typename T> class TensorPtr;
template<typename T> class Storage;
template<typename T, typename Enable=void> class IsTensor;
template<typename T, typename Enable=void> class IsTensorPtr;
template<typename T, typename Enable=void> class IsStorage;
}
