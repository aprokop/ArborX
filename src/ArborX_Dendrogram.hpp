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
        Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges, int level = 0)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha_" +
                                  std::to_string(level));

    auto const num_edges = sorted_edges.size();

    // Step 1: find alpha edges of the original MST
    auto alpha_edge_indices = Details::findAlphaEdges(exec_space, sorted_edges);

    auto num_alpha_edges = (int)alpha_edge_indices.size();
    printf("#alpha edges: %d [%.2f%%]\n", num_alpha_edges,
           (100.f * num_alpha_edges) / num_edges);

    // Step 2: construct virtual alpha-vertices
    auto alpha_vertices = Details::assignAlphaVertices(exec_space, sorted_edges,
                                                       alpha_edge_indices);

    // Step 3: construct alpha-MST
    auto alpha_edges = Details::buildAlphaMST(
        exec_space, sorted_edges, alpha_edge_indices, alpha_vertices);

    // Step 4: build dendrogram of the alpha-tree
    auto alpha_parents_of_alpha =
        (level < 5 && num_alpha_edges > 100
             ? alpha(exec_space, alpha_edges, level + 1)
             : Details::dendrogramUnionFind(exec_space, alpha_edges));
    {
      auto alpha_parents_of_alpha_copy =
          KokkosExt::clone(exec_space, alpha_parents_of_alpha);
      Kokkos::parallel_for(
          "ArborX::Dendrogram::transform_alpha_dendrogram",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
          KOKKOS_LAMBDA(int const e) {
            alpha_parents_of_alpha(e) =
                (alpha_parents_of_alpha_copy(e) != -1
                     ? alpha_edge_indices(alpha_parents_of_alpha_copy(e))
                     : -1);
          });
    }

    Kokkos::resize(alpha_edges, 0); // deallocate

    // Step 5: build alpha incidence matrix
    Kokkos::View<int *, MemorySpace> alpha_mat_offsets(
        "ArborX::Dendrogram::alpha_mat_offsets", 0);
    Kokkos::View<int *, MemorySpace> alpha_mat_edges(
        "ArborX::Dendrogram::alpha_mat_edges", 0);
    Details::buildAlphaIncidenceMatrix(exec_space, sorted_edges,
                                       alpha_edge_indices, alpha_vertices,
                                       alpha_mat_offsets, alpha_mat_edges);

    // Step 6: compute alpha_parents
    auto alpha_parents = computeAlphaParents(
        exec_space, sorted_edges, alpha_edge_indices, alpha_vertices,
        alpha_parents_of_alpha, alpha_mat_offsets, alpha_mat_edges);

    Kokkos::resize(alpha_edge_indices, 0);     // deallocate
    Kokkos::resize(alpha_vertices, 0);         // deallocate
    Kokkos::resize(alpha_parents_of_alpha, 0); // deallocate
    Kokkos::resize(alpha_mat_offsets, 0);      // deallocate
    Kokkos::resize(alpha_mat_edges, 0);        // deallocate

    // Step 7: build full dendrogram
    auto parents =
        Details::computeParents(exec_space, sorted_edges, alpha_parents);

    Kokkos::Profiling::popRegion();

    return parents;
  }
};

} // namespace ArborX

#endif
