#ifndef ARRAY_SUPPORT_HEADER_H
#define ARRAY_SUPPORT_HEADER_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// C-contiguous, NumPy-shaped, fixed-rank ndarray alias used throughout
// the parser. Concrete buffers are wrapped via the local
// wrap_malloc_buffer helper in stlfile.cpp so the capsule deleter
// matches the allocator.
template <typename T, size_t N>
using NDArray = nb::ndarray<nb::numpy, T, nb::ndim<N>, nb::c_contig>;

#endif // ARRAY_SUPPORT_HEADER_H
