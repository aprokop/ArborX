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

#ifndef ARBORX_DENDROGRAM_TRAVERSAL_HPP
#define ARBORX_DENDROGRAM_TRAVERSAL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX::Details
{

template <class ExecutionSpace, typename Parents, typename Heights,
          typename ChainOffsets, typename ChainLevels, typename Functor>
void traverseDendrogram(ExecutionSpace const &space, Parents const &parents,
                        Heights const &heights,
                        ChainOffsets const &chain_offsets,
                        ChainLevels const &chain_levels, Functor const &functor)
{
  auto &counts = counts_hierarchical;

  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::HDBSCAN::flat_clustering::counts_hierarchical_1");

  Kokkos::parallel_for(
      "ArborX::HDBSCAN::flat_clustering::compute_counts_hier_vertices",
      Kokkos::RangePolicy<ExecutionSpace>(space, n - 1, 2 * n - 1),
      KOKKOS_LAMBDA(int i) { Kokkos::atomic_add(&counts(parents(i)), 1); });

  auto chain_levels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chain_levels);
  using TeamPolicy =
      Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
  for (int level = 0; level < (int)chain_levels.size() - 1; ++level)
    Kokkos::parallel_for(
        "ArborX::HDBSCAN::flat_clustering::compute_counts_hier_level_" +
            std::to_string(level),
        TeamPolicy(space,
                   chain_levels_host(level + 1) - chain_levels_host(level),
                   Kokkos::AUTO),
        KOKKOS_LAMBDA(typename TeamPolicy::member_type const &team) {
          int chain = chain_levels(level) + team.league_rank();

          int begin = chain_offsets(chain);
          int end = chain_offsets(chain + 1);

          Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, begin, end),
                                [=](int i, int &update, bool final_pass) {
                                  update += counts(i);
                                  if (final_pass)
                                    counts(i) = update;
                                });
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            if (parents(end - 1) != -1)
              Kokkos::atomic_add(&counts(parents(end - 1)), counts(end - 1));
          });
        });
}

} // namespace ArborX::Details

#endif
