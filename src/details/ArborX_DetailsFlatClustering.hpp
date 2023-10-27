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
        KOKKOS_CLASS_LAMBDA(int i) {
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

    Kokkos::View<int *, MemorySpace> chain_levels("blah",
                                                  chain_offsets.size() - 1);
    Kokkos::parallel_for(
        "ArborX::HDBSCAN::flat_clustering::compute_chain_levels",
        Kokkos::RangePolicy<ExecutionSpace>(space, num_edges,
                                            2 * num_edges - 1),
        KOKKOS_LAMBDA(int i) {
      auto upper_bound = [&chain_offsets](int x) {
        int first = 0;
        int last = v.extent_int(0);
        int count = last - first;
        while (count > 0)
        {
          int step = count / 2;
          int mid = first + step;
          if (!(x < v(mid)))
          {
            first = ++mid;
            count -= step + 1;
          }
          else
          {
            count = step;
          }
        }
        return first;
      };

      int count = 1;
      int next_chain = upper_bound(parents(i));
      while
        do
        {
          count = Kokkos::atomic_fetch_add(
              &chain_levels(chain_levels(next_chain)), count);

          if (!count)
          {
            // First thread
            break;
          }

          count = chain_levels(next_chain);

          while (i != -1)
          {
            int next_chain = upper_bound(
          }
        });
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
