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
#ifndef ARBORX_HDBSCAN_HPP
#define ARBORX_HDBSCAN_HPP

#include <ArborX_Dendrogram.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_MinimumSpanningTree.hpp>

#include <Kokkos_Profiling_ProfileSection.hpp>

#include <iomanip>
#include <limits>

namespace ArborX
{

namespace HDBSCAN
{

struct Parameters
{
  // Print timers to the standard output
  bool _print_timers = false;
  // Print MST to the standard output
  bool _print_mst = false;
  // Verify dendrogram
  bool _verify_dendrogram = false;
  // Dendrogram implementation
  DendrogramImplementation _dendrogram = DendrogramImplementation::UNION_FIND;

  Parameters &setPrintTimers(bool print_timers)
  {
    _print_timers = print_timers;
    return *this;
  }
  Parameters &setPrintMST(bool print_mst)
  {
    _print_mst = print_mst;
    return *this;
  }
  Parameters &setVerifyDendrogram(bool verify_dendrogram)
  {
    _verify_dendrogram = verify_dendrogram;
    return *this;
  }
  Parameters &setDendrogram(DendrogramImplementation dendrogram)
  {
    _dendrogram = dendrogram;
    return *this;
  }
};
} // namespace HDBSCAN

template <typename ExecutionSpace, typename Primitives>
Kokkos::View<int *, typename Primitives::memory_space>
hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
        int core_min_size,
        HDBSCAN::Parameters const &parameters = HDBSCAN::Parameters())
{
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN");

  using MemorySpace = typename Primitives::memory_space;

  // Right now, we use the same minpts for computing core distances as well as
  // minimum cluster size. For the latter, minpts = 2 is special in that it
  // requires introducing a self-loops at the MST level to compute cluster
  // stability. To simplify our life, we disallow this case, and require
  // minpts > 2.
  // ARBORX_ASSERT(core_min_size > 2);

  int const n = primitives.extent_int(0);

  Kokkos::Profiling::ProfilingSection profile_mst("ArborX::HDBSCAN::mst");
  profile_mst.start();
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::mst");

  Details::MinimumSpanningTree<MemorySpace> mst(exec_space, primitives,
                                                core_min_size);

  Kokkos::Profiling::popRegion();
  profile_mst.stop();

  // Print MST
  if (parameters._print_mst)
  {
    auto mst_edges_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mst.edges);
    std::cout << "=== MST ===" << std::endl;
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int k = 0; k < n - 1; ++k)
    {
      int i = std::min(mst_edges_host(k).source, mst_edges_host(k).target);
      int j = std::max(mst_edges_host(k).source, mst_edges_host(k).target);
      float d = mst_edges_host(k).weight;
      std::cout << i << " " << j << " " << d << std::endl;
    }
  }

  Kokkos::Profiling::ProfilingSection profile_dendrogram(
      "ArborX::HDBSCAN::dendrogram");
  profile_dendrogram.start();
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::dendrogram");

  if (parameters._dendrogram != DendrogramImplementation::NONE)
  {
    Dendrogram<MemorySpace> dendrogram(exec_space, mst.edges,
                                       parameters._dendrogram);

    if (parameters._verify_dendrogram)
    {
      auto success = ArborX::Details::verifyDendrogram(
          exec_space, mst.edges, dendrogram._edge_parents);
      printf("Verification %s\n", (success ? "passed" : "failed"));
    }
  }
  Kokkos::Profiling::popRegion();
  profile_dendrogram.stop();

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::HDBSCAN::labels"),
      n);
  Kokkos::deep_copy(exec_space, labels, 0);

  Kokkos::Profiling::popRegion();

  return labels;
}

} // namespace ArborX

#endif
