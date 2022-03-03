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

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  using Point = ArborX::Point<2>;
  using Segment = ArborX::Experimental::Segment<2>;

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

  ArborX::BoundingVolumeHierarchy const tree(
      space, ArborX::Experimental::attach_indices(segments));

  // The query will resize indices and offsets accordingly
  Kokkos::View<unsigned *, MemorySpace> indices("Example::indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  tree.query(space, queries, ArborX::Details::LegacyDefaultCallback{}, indices,
             offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  // Expected output:
  //   offsets: 0 1
  //   indices: 1
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\nindices: ";
  std::copy(indices_host.data(), indices_host.data() + indices.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";

  return 0;
}
