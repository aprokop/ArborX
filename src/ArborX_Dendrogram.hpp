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
#ifndef ARBORX_DENDROGRAM_HPP
#define ARBORX_DENDROGRAM_HPP

#include <ArborX_DetailsDendrogramAlpha.hpp>
#include <ArborX_DetailsDendrogramUnionFind.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>

namespace ArborX
{

enum class DendrogramImplementation
{
  UNION_FIND,
  ALPHA,
  NONE // FIXME to remove
};

template <typename MemorySpace>
struct Dendrogram
{
  using WeightedEdge = Details::WeightedEdge;

  Kokkos::View<WeightedEdge *, MemorySpace> _edges;
  Kokkos::View<int *, MemorySpace> _edge_parents;

  template <typename ExecutionSpace>
  Dendrogram(ExecutionSpace const &exec_space,
             Kokkos::View<WeightedEdge *, MemorySpace> edges,
             DendrogramImplementation impl = DendrogramImplementation::ALPHA)
      : _edges(edges)
      , _edge_parents("ArborX::Dendrogram::edge_parents", 0)

  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram");

    bool const are_edges_sorted = checkEdgesSorted(exec_space, edges);
    printf("Edges are sorted: %s\n", (are_edges_sorted ? "yes" : "no"));

    Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges;
    Kokkos::View<unsigned int *, MemorySpace> permute;
    if (are_edges_sorted == true)
    {
      sorted_edges = edges;
    }
    else
    {
      Kokkos::Profiling::ProfilingSection profile_edge_sort(
          "ArborX::Dendrogram::edge_sort");
      profile_edge_sort.start();

      Kokkos::Profiling::pushRegion("ArborX::Dendrogram::edge_sort");
      sorted_edges = KokkosExt::clone(exec_space, edges);
      permute = Details::sortEdges(exec_space, sorted_edges);
      Kokkos::Profiling::popRegion();

      profile_edge_sort.stop();
    }

    Kokkos::View<int *, MemorySpace> edge_parents;
    switch (impl)
    {
    case DendrogramImplementation::UNION_FIND:
      edge_parents = unionFind(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::ALPHA:
      edge_parents = alpha(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::NONE:
      ARBORX_ASSERT(false);
    }

    if (are_edges_sorted == true)
      _edge_parents = edge_parents;
    else
      assignEdgeParents(exec_space, edge_parents, permute);

    Kokkos::Profiling::popRegion();
  }

  template <typename ExecutionSpace, typename Permute>
  void assignEdgeParents(ExecutionSpace const &exec_space,
                         Kokkos::View<int *, MemorySpace> permuted_edge_parents,
                         Permute permute)
  {
    auto &edge_parents = _edge_parents; // FIXME avoid capturing *this
    KokkosExt::reallocWithoutInitializing(exec_space, edge_parents,
                                          permuted_edge_parents.size());
    Kokkos::parallel_for(
        "ArborX::Dendrogram::permute_parents",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edge_parents.size()),
        KOKKOS_LAMBDA(int i) {
          edge_parents(permute(i)) = permuted_edge_parents(i);
        });
  }

  template <typename ExecutionSpace>
  bool checkEdgesSorted(ExecutionSpace const &exec_space,
                        Kokkos::View<WeightedEdge *, MemorySpace> edges)
  {
    int count = 0;
    Kokkos::parallel_reduce(
        "ArborX::Dendrogram::check_edges_sorted",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size() - 1),
        KOKKOS_LAMBDA(int i, int &partial_sum) {
          if (edges(i).weight < edges(i + 1).weight)
            ++partial_sum;
        },
        count);
    return (count == 0);
  }

  template <typename ExecutionSpace, typename Edges>
  Kokkos::View<int *, MemorySpace> unionFind(ExecutionSpace const &exec_space,
                                             Edges sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");
    auto edge_parents = Details::dendrogramUnionFind(exec_space, sorted_edges);
    Kokkos::Profiling::popRegion();

    return edge_parents;
  }

  template <typename ExecutionSpace>
  Kokkos::View<int *, MemorySpace>
  alpha(ExecutionSpace const &exec_space,
        Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha");

    using Details::ROOT_CHAIN_VALUE;
    using Details::UNDEFINED_CHAIN_VALUE;

    Kokkos::View<int *, MemorySpace> sided_level_parents(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::sided_parents"),
        sorted_edges.size());
    Kokkos::deep_copy(exec_space, sided_level_parents, UNDEFINED_CHAIN_VALUE);

    Kokkos::View<int *, MemorySpace> global_map(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::sided_parents"),
        sorted_edges.size());
    iota(exec_space, global_map);

    auto edges = sorted_edges;

#ifdef VERBOSE
    int global_num_edges = sorted_edges.size();
#endif

    Kokkos::Profiling::ProfilingSection profile_compute_alpha_edges(
        "ArborX::Dendrogram::alpha_edges");
    Kokkos::Profiling::ProfilingSection profile_alpha_vertices(
        "ArborX::Dendrogram::alpha_vertices");
    Kokkos::Profiling::ProfilingSection profile_build_alpha_incidence_matrix(
        "ArborX::Dendrogram::alpha_incidence_matrix");
    Kokkos::Profiling::ProfilingSection profile_update_sided_parents(
        "ArborX::Dendrogram::sided_parents");
    Kokkos::Profiling::ProfilingSection profile_compress_edges(
        "ArborX::Dendrogram::compress_edges");

    int level = 0;
    do
    {
      Kokkos::Profiling::pushRegion("ArborX::Dendrogram::level_" +
                                    std::to_string(level));
      int num_edges = edges.size();

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] edges:\n", level);
      for (int i = 0; i < num_edges; ++i)
        printf("[%5d] %5d %5d %10.2f\n", i, edges(i).source, edges(i).target,
               edges(i).weight);
      fflush(stdout);
#endif

      // Step 1: find alpha edges of the current MST
      profile_compute_alpha_edges.start();
      auto alpha_edge_indices = Details::findAlphaEdges(exec_space, edges);
      profile_compute_alpha_edges.stop();

      auto num_alpha_edges = (int)alpha_edge_indices.size();
      printf("[%d] #alpha edges: %d / %d [%.2f%%]\n", level, num_alpha_edges,
             num_edges, (100.f * num_alpha_edges) / num_edges);

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] Alpha edge indices:\n", level);
      for (int i = 0; i < (int)alpha_edge_indices.size(); ++i)
        printf(" %d", alpha_edge_indices(i));
      printf("\n");
      fflush(stdout);
#endif

      if (num_alpha_edges == 0)
      {
        // Done with recursion. The edges that haven't been assigned yet
        // automatically have ROOT_CHAIN_VALUE from the initialization.
        Kokkos::parallel_for(
            "ArborX::Dendrogram::assign_remaining_side_parents",
            Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
            KOKKOS_LAMBDA(int const e) {
              int &sided_level_parent = sided_level_parents(global_map(e));
              if (sided_level_parent == UNDEFINED_CHAIN_VALUE)
                sided_level_parent = ROOT_CHAIN_VALUE;
            });

#ifdef VERBOSE
        printf("-------------------------------------\n");
        printf("[%d] sided parents:\n", level);
        for (int i = 0; i < (int)global_num_edges; ++i)
          printf("%5d -> %5d\n", i, sided_level_parents(i));
        fflush(stdout);
#endif

        Kokkos::Profiling::popRegion();
        break;
      }

      auto const largest_alpha_index =
          KokkosExt::lastElement(exec_space, alpha_edge_indices);

      // Step 2: construct virtual alpha-vertices
      profile_alpha_vertices.start();
      auto alpha_vertices =
          Details::assignAlphaVertices(exec_space, edges, alpha_edge_indices);
      profile_alpha_vertices.stop();

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] Alpha vertices:\n", level);
      for (int i = 0; i < (int)alpha_vertices.size(); ++i)
        printf(" %d", alpha_vertices(i));
      printf("\n");
      fflush(stdout);
#endif

      // Step 3: build alpha incidence matrix
      profile_build_alpha_incidence_matrix.start();
      Kokkos::View<int *, MemorySpace> alpha_mat_offsets(
          "ArborX::Dendrogram::alpha_mat_offsets", 0);
      Kokkos::View<int *, MemorySpace> alpha_mat_edges(
          "ArborX::Dendrogram::alpha_mat_edges", 0);
      Details::buildAlphaIncidenceMatrix(exec_space, edges, alpha_edge_indices,
                                         alpha_vertices, alpha_mat_offsets,
                                         alpha_mat_edges);
      profile_build_alpha_incidence_matrix.stop();

      Kokkos::resize(alpha_edge_indices, 0); // deallocate

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] Alpha incidence matrix:\n", level);
      for (int i = 0; i < (int)alpha_mat_offsets.size() - 1; ++i)
      {
        printf("%5d:", i);
        for (int j = alpha_mat_offsets(i); j < alpha_mat_offsets(i + 1); ++j)
          printf(" %5d", alpha_mat_edges(j));
        printf("\n");
      }
      fflush(stdout);
#endif

      // Step 4: update sided parents
      profile_update_sided_parents.start();
      Details::updateSidedParents(
          exec_space, edges, largest_alpha_index, alpha_vertices,
          alpha_mat_offsets, alpha_mat_edges, global_map, sided_level_parents);
      profile_update_sided_parents.stop();

      Kokkos::resize(alpha_vertices, 0);    // deallocate
      Kokkos::resize(alpha_mat_offsets, 0); // deallocate
      Kokkos::resize(alpha_mat_edges, 0);   // deallocate

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] sided parents:\n", level);
      for (int i = 0; i < (int)global_num_edges; ++i)
        printf("%5d -> %5d\n", i, sided_level_parents(i));
      fflush(stdout);
#endif

      // Step 5: compress edges
      profile_compress_edges.start();
      auto [compressed_edges, compressed_global_map] =
          Details::compressEdgesAndGlobalMap(exec_space, edges,
                                             sided_level_parents, global_map);
      profile_compress_edges.stop();

      auto num_compressed_edges = (int)compressed_edges.size();
      printf("[%d] #compressed edges: %d / %d [%.2f%%]\n", level,
             num_compressed_edges, num_edges,
             (100.f * num_compressed_edges) / num_edges);

#ifdef VERBOSE
      printf("-------------------------------------\n");
      printf("[%d] global map:\n", level);
      for (int i = 0; i < (int)num_alpha_edges; ++i)
        printf(" %5d", global_map(i));
      printf("\n");
      fflush(stdout);
#endif

      // Prepare for the next iteration
      global_map = compressed_global_map;
      edges = compressed_edges;

      Kokkos::Profiling::popRegion();

    } while (true);

    // Step 6: build full dendrogram
    Kokkos::Profiling::ProfilingSection profile_compute_parents(
        "ArborX::Dendrogram::parents");
    profile_compute_parents.start();
    auto parents = Details::computeParents(exec_space, sided_level_parents);
    profile_compute_parents.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("parents:\n");
    for (int i = 0; i < (int)parents.size(); ++i)
      printf("%5d -> %5d\n", i, parents(i));
    fflush(stdout);
#endif

#ifdef VERBOSE
    bool success = verifyDendrogram(exec_space, sorted_edges, parents);
    printf("Verification %s\n", (success ? "passed" : "failed"));
#endif

    Kokkos::Profiling::popRegion();

    return parents;
  }
};

} // namespace ArborX

#endif
