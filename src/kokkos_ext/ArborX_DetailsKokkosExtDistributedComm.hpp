/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_KOKKOS_EXT_DISTRIBUTED_COMM_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_DISTRIBUTED_COMM_HPP

#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>

#include <mpi.h>

namespace ArborX::Details::KokkosExt
{

template <typename View>
void isend(View const &view, int destination, int tag, MPI_Comm comm,
           MPI_Request &request)
{
  using ValueType = typename View::value_type;
  using Layout = typename View::array_layout;
  static_assert(Kokkos::is_view_v<View>);
  static_assert(View::rank == 1 &&
                (std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                 std::is_same_v<Layout, Kokkos::LayoutRight>));

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  MPI_Isend(view.data(), n * sizeof(ValueType), MPI_BYTE, destination, tag,
            comm, &request);
}

template <typename View>
void irecv(View const &view, int source, int tag, MPI_Comm comm,
           MPI_Request &request)
{
  using ValueType = typename View::value_type;
  using Layout = typename View::array_layout;
  static_assert(Kokkos::is_view_v<View>);
  static_assert(View::rank == 1 &&
                (std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                 std::is_same_v<Layout, Kokkos::LayoutRight>));

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  MPI_Irecv(view.data(), n * sizeof(ValueType), MPI_BYTE, source, tag, comm,
            &request);
}

} // namespace ArborX::Details::KokkosExt

#endif
