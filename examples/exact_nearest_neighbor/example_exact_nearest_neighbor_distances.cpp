/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_Segment.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

using Point = ArborX::Point<2>;
using Segment = ArborX::Experimental::Segment<2>;

struct DistanceCallback
{
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Segment const &segment,
                                  OutputFunctor const &out) const
  {
    using ArborX::Details::distance;
    Point const &point = ArborX::getGeometry(predicate);
    out(distance(point, segment));
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::View<Segment *, MemorySpace> segments("segments", 2);
  auto segments_host = Kokkos::create_mirror_view(segments);

  // Two crossed segments. The bounding box of the larger one fully encompases
  // the bounding box of the smaller one. The point is in the left top corner.
  // x    /
  //    -/
  //    /-
  //   /
  Point point{0.f, 1.f};
  segments_host[0] = {{0.f, 0.f}, {1.f, 1.f}};
  segments_host[1] = {{0.4f, 0.6f}, {0.6f, 0.4f}};
  Kokkos::deep_copy(segments, segments_host);

  Kokkos::View<decltype(ArborX::nearest(Point())) *, MemorySpace> queries(
      "Example::queries", 1);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host[0] = ArborX::nearest(point, 1);
  Kokkos::deep_copy(queries, queries_host);

  ExecutionSpace space;

  ArborX::BoundingVolumeHierarchy const tree(space, segments);

  // The query will resize indices and offsets accordingly
  Kokkos::View<float *, MemorySpace> distances("Example::distances", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  tree.query(space, queries, DistanceCallback{}, distances, offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto distances_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, distances);

  // Expected output:
  //   offsets: 0 1
  //   distances: 0.565685
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\ndistances: ";
  std::copy(distances_host.data(), distances_host.data() + distances.size(),
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << "\n";

  return 0;
}
