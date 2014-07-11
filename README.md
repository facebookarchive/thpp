# TH++: A C++ tensor library

TH++ is a C++ tensor library, implemented as a wrapper around the
[TH library](https://github.com/torch/torch7/tree/master/lib/TH) (the low-level
tensor library in [Torch](http://torch.ch/)). There is unfortunately little
documentation about TH, but the interface mimics the Lua
[Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md) interface.

The core of the library is the `Tensor<T>` class template, where `T` is a
numeric type (usually floating point, `float` or `double`). A tensor is
a multi-dimensional array, usually in C (row-major) order, but many
operations (transpose, slice, etc) are performed by permuting indexes and
changing offsets, so the data is no longer contiguous / in row-major order.
Read the [numpy.ndarray
documentation](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html)
for more details about the strided indexing scheme.

Tensors may also share memory with other tensors; operations that manipulate
metadata (select, slice, transpose, etc) will make the destination tensor
share memory with the source. To ensure you have a unique copy, call
`force(Tensor<T>::UNIQUE)` on the tensor. Similarly, to ensure you have
a contiguous C (row-major) tensor, call `force(Tensor<T>::CONTIGUOUS)`, which
may also create a unique copy.

Please see the header file `<thpp/Tensor.h>` for more details.
