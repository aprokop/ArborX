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

#ifndef ARBORX_ENABLE_KOKKOS_COMM
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>
#else
#include <KokkosComm.hpp>
#include <KokkosComm_alltoall.hpp>
#include <KokkosComm_irecv.hpp>
#endif

#include <mpi.h>

namespace ArborX::Details::KokkosExt
{

#ifndef ARBORX_ENABLE_KOKKOS_COMM
template <typename View>
inline constexpr bool is_valid_mpi_view_v =
    (View::rank == 1 &&
     (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
      std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>));

template <typename View>
void isend(View const &view, int destination, int tag, MPI_Comm comm,
           MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Isend(view.data(), n * sizeof(ValueType), MPI_BYTE, destination, tag,
            comm, &request);
}

template <typename View>
void irecv(View const &view, int source, int tag, MPI_Comm comm,
           MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Irecv(view.data(), n * sizeof(ValueType), MPI_BYTE, source, tag, comm,
            &request);
}

template <typename View>
void allgather(View const &view, MPI_Comm comm)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
                sizeof(ValueType), MPI_BYTE, comm);
}

template <typename View>
void alltoall(View const &view, MPI_Comm comm)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
               sizeof(ValueType), MPI_BYTE, comm);
}

#else
using KokkosComm::allgather;
using KokkosComm::isend;
using KokkosComm::Impl::alltoall;
using KokkosComm::Impl::irecv;
#endif

} // namespace ArborX::Details::KokkosExt

#endif
