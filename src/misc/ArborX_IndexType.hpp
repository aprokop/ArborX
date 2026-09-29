/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INDEX_TYPE_HPP
#define ARBORX_INDEX_TYPE_HPP

#include <ArborX_Config.hpp> // ARBORX_ENABLE_LARGE_INDEX

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace ArborX::Details
{

// Signed integer type used for indexing nodes and values of a hierarchy. It is
// also used as the Kokkos::IndexType of the parallel kernels.
#ifdef ARBORX_ENABLE_LARGE_INDEX
using index_type = std::int64_t;
#else
using index_type = int;
#endif

template <typename ExecutionSpace, typename... Args>
using IndexRangePolicy =
    Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<index_type>, Args...>;

} // namespace ArborX::Details

#endif
