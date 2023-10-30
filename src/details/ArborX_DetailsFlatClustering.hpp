/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_FLAT_CLUSTERING_HPP
#define ARBORX_DETAILS_FLAT_CLUSTERING_HPP

#include <Kokkos_View.hpp>

namespace ArborX::Details
{

template <class ExecutionSpace, typename Parents, typename Heights,
          typename ChainOffsets>
void computeFlatClustering(ExecutionSpace const &space, Parents const &parents,
                           Heights const &heights,
                           ChainOffsets const &chain_offsets)
{
  using MemorySpace = typename Parents::memory_space;

  KokkosExt::ScopedProfileRegion guard("ArborX::HDBSCAN::flat_clustering");
  auto const n = heights.size() + 1;

  Kokkos::View<int *, MemorySpace> counts_naive(
      Kokkos::view_alloc(space,
                         "ArborX::HDBSCAN::flat_clustering::counts_naive"),
      n - 1);
  Kokkos::View<int *, MemorySpace> counts_hierarchical(
      Kokkos::view_alloc(
          space, "ArborX::HDBSCAN::flat_clustering::counts_hierarchical"),
      n - 1);

  // Naive approach
  {
    auto &counts = counts_naive;

    KokkosExt::ScopedProfileRegion guard(
        "ArborX::HDBSCAN::flat_clustering::counts_naive");

    Kokkos::parallel_for(
        "ArborX::HDBSCAN::flat_clustering::compute_counts",
        Kokkos::RangePolicy<ExecutionSpace>(space, n - 1, 2 * n - 1),
        KOKKOS_LAMBDA(int i) {
          int count = 1;
          int parent = parents(i);
          do
          {
            auto stored_count =
                Kokkos::atomic_fetch_add(&counts(parent), count);

            // Terminate the first thread up
            if (stored_count == 0)
              break;

            // Update the count using local variable, instead of reading
            // counts
            count += stored_count;

            parent = parents(parent);
          } while (parent != -1);
        });
  }

  // Hierarchical approach
  {
    auto &counts = counts_hierarchical;

    KokkosExt::ScopedProfileRegion guard(
        "ArborX::HDBSCAN::flat_clustering::counts_hierarchical_1");
  }

  // Check
  int wrong = 0;
  Kokkos::parallel_reduce(
      "ArborX::HDBSCAN::flat_clustering::counts_check",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i, int &update) {
        if (counts_naive(i) != counts_hierarchical(i))
        {
          ++update;
          printf("[%d]: naive = %d, hier = %d\n", i, counts_naive(i),
                 counts_hierarchical(i));
        }
      },
      wrong);
  std::cout << "Counts check: " << (wrong ? "failed" : "succeeded")
            << std::endl;
}

} // namespace ArborX::Details

#endif
