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

#ifndef ARBORX_KOKKOS_EXT_DISTRIBUTED_COMM_HPP
#define ARBORX_KOKKOS_EXT_DISTRIBUTED_COMM_HPP

#include <ArborX_Config.hpp>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>

#include <mpi.h>

namespace ArborX::Details::KokkosExt
{

template <typename View>
inline constexpr bool is_valid_mpi_view_v =
    (View::rank == 1 &&
     (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
      std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>));

template <typename ExecutionSpace, typename View>
void mpi_isend(MPI_Comm comm, ExecutionSpace const &space, View const &view,
               int destination, int tag, MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  KOKKOS_ASSERT(n > 0);

#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  auto send_view = view;
#else
  auto send_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::view_alloc(Kokkos::HostSpace{}, space,
                         Kokkos::WithoutInitializing),
      view);
#endif
  space.fence("ArborX::KokkosExt::mpi_isend");

  using ValueType = typename View::value_type;
  MPI_Isend(view.data(), n * sizeof(ValueType), MPI_BYTE, destination, tag,
            comm, &request);
}

template <typename ExecutionSpace, typename View>
void mpi_irecv(MPI_Comm comm, ExecutionSpace const &space, View const &view,
               int source, int tag, MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  KOKKOS_ASSERT(n > 0);

#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  auto recv_view = view;
#else
  auto recv_view = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::HostSpace{}, space,
                         Kokkos::WithoutInitializing),
      view);
#endif

  using ValueType = typename View::value_type;
  MPI_Irecv(view.data(), n * sizeof(ValueType), MPI_BYTE, source, tag, comm,
            &request);

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(space, view, recv_view);
#endif
}

template <typename ExecutionSpace, typename View>
void mpi_alltoall(MPI_Comm comm, ExecutionSpace const &space, View const &view)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  using ValueType = typename View::value_type;

  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  auto const n = view.size();
  KOKKOS_ASSERT(n > 0);
  KOKKOS_ASSERT(n % comm_size == 0);

  auto per_rank = n / comm_size;

#ifdef ARBORX_ENABLE_GPU_MPI
  auto send_view = view;
#else
  auto send_view = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::HostSpace{}, space,
                         Kokkos::WithoutInitializing),
      view);
  Kokkos::deep_copy(space, send_view, view);
#endif
  space.fence("ArborX::KokkosExt::mpi_alltoall");

  using ValueType = typename View::value_type;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
                sizeof(ValueType), MPI_BYTE, comm);

#ifndef ARBORX_ENABLE_GPU_MPI
  Kokkos::deep_copy(space, view, send_view);
#endif
}

template <typename ExecutionSpace, typename View>
void mpi_allgather(MPI_Comm comm, ExecutionSpace const &space, View const &view)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  auto const n = view.size();
  KOKKOS_ASSERT(n > 0);
  KOKKOS_ASSERT(n % comm_size == 0);

  auto per_rank = n / comm_size;

#ifdef ARBORX_ENABLE_GPU_MPI
  auto send_view = view;
#else
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  auto send_view = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::HostSpace{}, space,
                         Kokkos::WithoutInitializing),
      view);
  auto slice =
      Kokkos::make_pair(per_rank * comm_rank, (per_rank + 1) * comm_rank);
  Kokkos::deep_copy(space, Kokkos::subview(send_view, slice),
                    Kokkos::subview(view, slice));
#endif
  space.fence("ArborX::KokkosExt::mpi_allgather");

  using ValueType = typename View::value_type;
  MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
               sizeof(ValueType), MPI_BYTE, comm);

#ifndef ARBORX_ENABLE_GPU_MPI
  Kokkos::deep_copy(space, view, send_view);
#endif
}

} // namespace ArborX::Details::KokkosExt

#endif
