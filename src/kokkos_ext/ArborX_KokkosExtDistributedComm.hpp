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

template <typename Scalar>
MPI_Datatype mpi_type()
{
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, std::byte>)
    return MPI_BYTE;

  else if constexpr (std::is_same_v<T, char>)
    return MPI_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return MPI_UNSIGNED_CHAR;

  else if constexpr (std::is_same_v<T, short>)
    return MPI_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>)
    return MPI_UNSIGNED_SHORT;

  else if constexpr (std::is_same_v<T, int>)
    return MPI_INT;
  else if constexpr (std::is_same_v<T, unsigned>)
    return MPI_UNSIGNED;

  else if constexpr (std::is_same_v<T, long>)
    return MPI_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>)
    return MPI_UNSIGNED_LONG;

  else if constexpr (std::is_same_v<T, long long>)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<T, unsigned long long>)
    return MPI_UNSIGNED_LONG_LONG;

  else if constexpr (std::is_same_v<T, std::int8_t>)
    return MPI_INT8_T;
  else if constexpr (std::is_same_v<T, std::uint8_t>)
    return MPI_UINT8_T;

  else if constexpr (std::is_same_v<T, std::int16_t>)
    return MPI_INT16_T;
  else if constexpr (std::is_same_v<T, std::uint16_t>)
    return MPI_UINT16_T;

  else if constexpr (std::is_same_v<T, std::int32_t>)
    return MPI_INT32_T;
  else if constexpr (std::is_same_v<T, std::uint32_t>)
    return MPI_UINT32_T;

  else if constexpr (std::is_same_v<T, std::int64_t>)
    return MPI_INT64_T;
  else if constexpr (std::is_same_v<T, std::uint64_t>)
    return MPI_UINT64_T;

  else if constexpr (std::is_same_v<T, std::size_t>)
  {
    if constexpr (sizeof(std::size_t) == 1)
      return MPI_UINT8_T;
    if constexpr (sizeof(std::size_t) == 2)
      return MPI_UINT16_T;
    if constexpr (sizeof(std::size_t) == 4)
      return MPI_UINT32_T;
    if constexpr (sizeof(std::size_t) == 8)
      return MPI_UINT64_T;
  }

  else if constexpr (std::is_same_v<T, std::ptrdiff_t>)
  {
    if constexpr (sizeof(std::ptrdiff_t) == 1)
      return MPI_INT8_T;
    if constexpr (sizeof(std::ptrdiff_t) == 2)
      return MPI_INT16_T;
    if constexpr (sizeof(std::ptrdiff_t) == 4)
      return MPI_INT32_T;
    if constexpr (sizeof(std::ptrdiff_t) == 8)
      return MPI_INT64_T;
  }

  else if constexpr (std::is_same_v<T, float>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, long double>)
    return MPI_LONG_DOUBLE;

  else if constexpr (std::is_same_v<T, Kokkos::complex<float>>)
    return MPI_COMPLEX;
  else if constexpr (std::is_same_v<T, Kokkos::complex<double>>)
    return MPI_DOUBLE_COMPLEX;

  return nullptr;
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

template <typename Scalar>
inline bool is_known_mpi_type = (mpi_type_v<Scalar> != nullptr);

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
  if (is_known_mpi_type<ValueType>)
  {
    auto mpi_t = mpi_type_v<ValueType>;
    MPI_Isend(send_view.data(), n, mpi_t, destination, tag, comm, &request);
    return;
  }
  MPI_Isend(send_view.data(), n * sizeof(ValueType), MPI_BYTE, destination, tag,
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
  if (is_known_mpi_type<ValueType>)
  {
    auto mpi_t = mpi_type_v<ValueType>;
    MPI_Irecv(recv_view.data(), n, mpi_t, source, tag, comm, &request);
  }
  else
  {
    MPI_Irecv(recv_view.data(), n * sizeof(ValueType), MPI_BYTE, source, tag,
              comm, &request);
  }

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(space, view, recv_view);
#endif
}

template <typename ExecutionSpace, typename View>
void mpi_alltoall_inplace(MPI_Comm comm, ExecutionSpace const &space,
                          View const &view)
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
  space.fence("ArborX::KokkosExt::mpi_alltoall_inplace");

  using ValueType = typename View::value_type;
  if (is_known_mpi_type<ValueType>)
  {
    auto mpi_t = mpi_type_v<ValueType>;
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(), per_rank,
                 mpi_t, comm);
  }
  else
  {
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
                 per_rank * sizeof(ValueType), MPI_BYTE, comm);
  }

#ifndef ARBORX_ENABLE_GPU_MPI
  Kokkos::deep_copy(space, view, send_view);
#endif
}

template <typename ExecutionSpace, typename View>
void mpi_allgather_inplace(MPI_Comm comm, ExecutionSpace const &space,
                           View const &view)
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
  space.fence("ArborX::KokkosExt::mpi_allgather_inplace");

  using ValueType = typename View::value_type;

  if (is_known_mpi_type<ValueType>)
  {
    auto mpi_t = mpi_type_v<ValueType>;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, send_view.data(),
                  per_rank, mpi_t, comm);
  }
  else
  {
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, send_view.data(),
                  per_rank * sizeof(ValueType), MPI_BYTE, comm);
  }

#ifndef ARBORX_ENABLE_GPU_MPI
  Kokkos::deep_copy(space, view, send_view);
#endif
}

} // namespace ArborX::Details::KokkosExt

#endif
