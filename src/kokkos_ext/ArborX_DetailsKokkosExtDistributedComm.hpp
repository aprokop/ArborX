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

template <typename Scalar>
MPI_Datatype mpi_type()
{
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, char>)
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

  else
    return MPI_BYTE;
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

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

  auto const n = view.extent_int(0);
  ARBORX_ASSERT(n > 0);

  auto type = mpi_type_v<ValueType>;
  auto count = (type == MPI_BYTE ? n * sizeof(ValueType) : n);
  MPI_Isend(view.data(), count, type, destination, tag, comm, &request);
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

  auto const n = view.extent_int(0);
  ARBORX_ASSERT(n > 0);

  auto type = mpi_type_v<ValueType>;
  auto count = (type == MPI_BYTE ? n * sizeof(ValueType) : n);
  MPI_Irecv(view.data(), count, type, source, tag, comm, &request);
}

} // namespace ArborX::Details::KokkosExt

#endif
