/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DBSCAN_HPP
#define ARBORX_DBSCAN_HPP

#include <ArborX_LinearBVH.hpp>

#include <map>

namespace ArborX
{

template <typename View>
struct PrimitivesWithRadius
{
  View _view;
  double _r;
};

template <typename View>
auto buildPredicates(View v, double r)
{
  return PrimitivesWithRadius<View>{v, r};
}

template <typename View>
struct AccessTraits<PrimitivesWithRadius<View>, PredicatesTag>
{
  using memory_space = typename View::memory_space;
  using Predicates = PrimitivesWithRadius<View>;
  static size_t size(Predicates const &w) { return w._view.extent(0); }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(Sphere{w._view(i), w._r}), (int)i);
  }
};

namespace NPCF
{

template <typename MemorySpace>
struct ExactCountCallback
{
  Kokkos::View<ArborX::Point *, MemorySpace> _points;
  Kokkos::View<int, MemorySpace> _count;
  float a;
  float b;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    auto i = getData(query);
    auto dist = Details::distance(_points(i), _points(j));
    if (dist >= a && dist <= b)
      Kokkos::atomic_fetch_add(&_count(), 1);
  }
};

} // namespace NPCF

template <typename ExecutionSpace, typename Points>
int countExact(ExecutionSpace const &exec_space, Points const &points, float a,
               float b)
{
  using MemorySpace = typename Points::memory_space;

  ArborX::BVH<MemorySpace> bvh(exec_space, points);

  auto const predicates = buildPredicates(points, b);

  Kokkos::View<int, MemorySpace> count("ArborX::npcf::count");
  bvh.query(exec_space, predicates,
            NPCF::ExactCountCallback<MemorySpace>{points, count, a, b});

  auto count_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);
  return count_host();
}

#if 0
template <typename ExecutionSpace, typename Points>
int countRandomized(ExecutionSpace const &exec_space, Points const &points, float a,
               float b, float eps)
{
  using MemorySpace = typename Points::memory_space;

  ArborX::BVH<MemorySpace> bvh(exec_space, points);

  auto const predicates = buildPredicates(points, b);

  Kokkos::View<int, MemorySpace> count("ArborX::npcf::count");
  bvh.query(exec_space, predicates,
            NPCF::ExactCountCallback<MemorySpace>{points, count, a, b});

  auto count_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);
  return count_host();
}
#endif

} // namespace ArborX

#endif
