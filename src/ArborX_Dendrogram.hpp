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
      , _edge_parents(Kokkos::view_alloc(exec_space,
                                         Kokkos::WithoutInitializing,
                                         "ArborX::Dendrogram::edge_parents"),
                      edges.size())

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

    Kokkos::View<int *, MemorySpace> sided_edge_parents;
    switch (impl)
    {
    case DendrogramImplementation::UNION_FIND:
      sided_edge_parents = unionFind(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::ALPHA:
      sided_edge_parents = alpha(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::NONE:
      ARBORX_ASSERT(false);
    }

    // TODO: convert sided parents to unsided
    auto &edge_parents = sided_edge_parents;

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
  auto unionFind(ExecutionSpace const &exec_space, Edges sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");
    auto edge_parents = Details::dendrogramUnionFind(exec_space, sorted_edges);
    Kokkos::Profiling::popRegion();

    return edge_parents;
  }

  template <typename ExecutionSpace>
  auto alpha(ExecutionSpace const &exec_space,
             Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha");

    auto const num_edges = sorted_edges.extent_int(0);

#define VERBOSE

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[0] edges:\n");
    for (int i = 0; i < num_edges; ++i)
      printf("%5d %5d %10.2f\n", sorted_edges(i).source, sorted_edges(i).target,
             sorted_edges(i).weight);
#endif

    // Step 1: find alpha edges of the original MST
    Kokkos::Profiling::ProfilingSection profile_compute_alpha_edges(
        "ArborX::Dendrogram::compute_alpha_edges");
    profile_compute_alpha_edges.start();
    auto alpha_edge_indices = Details::findAlphaEdges(exec_space, sorted_edges);
    profile_compute_alpha_edges.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    auto num_alpha_edges = (int)alpha_edge_indices.size();
    printf("#alpha edges: %d [%.2f%%]\n", num_alpha_edges,
           (100.f * num_alpha_edges) / num_edges);
    printf("[1] Alpha edge indices:\n");
    for (int i = 0; i < (int)alpha_edge_indices.size(); ++i)
      printf(" %d", alpha_edge_indices(i));
    printf("\n");
#endif

    // Step 2: construct virtual alpha-vertices
    Kokkos::Profiling::ProfilingSection profile_alpha_vertices(
        "ArborX::Dendrogram::alpha_vertices");
    profile_alpha_vertices.start();
    auto alpha_vertices = Details::assignAlphaVertices(exec_space, sorted_edges,
                                                       alpha_edge_indices);
    profile_alpha_vertices.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[2] Alpha vertices:\n");
    for (int i = 0; i < (int)alpha_vertices.size(); ++i)
      printf(" %d", alpha_vertices(i));
    printf("\n");
#endif

    // Step 3: construct alpha-MST
    Kokkos::Profiling::ProfilingSection profile_alpha_mst(
        "ArborX::Dendrogram::alpha_mst");
    profile_alpha_mst.start();
    auto alpha_edges = Details::buildAlphaMST(
        exec_space, sorted_edges, alpha_edge_indices, alpha_vertices);
    profile_alpha_mst.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[3] Alpha edges:\n");
    for (int i = 0; i < (int)alpha_edges.size(); ++i)
      printf("%5d %5d %10.2f\n", alpha_edges(i).source, alpha_edges(i).target,
             alpha_edges(i).weight);
#endif

    // Step 4: build dendrogram of the alpha-tree
    Kokkos::Profiling::ProfilingSection profile_dendrogram_alpha(
        "ArborX::Dendrogram::dendrogram_alpha");
    profile_dendrogram_alpha.start();
    auto sided_alpha_parents_of_alpha = Details::dendrogramUnionFind(
        exec_space, alpha_edges, alpha_edge_indices);
    profile_dendrogram_alpha.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[4] Sided parents of alpha edges:\n");
    for (int i = 0; i < num_alpha_edges; ++i)
      printf("%5d [%5d] -> %5d\n", i, alpha_edge_indices(i),
             sided_alpha_parents_of_alpha(i));
#endif

    // Step 5: build alpha incidence matrix
    Kokkos::Profiling::ProfilingSection profile_build_alpha_incidence_matrix(
        "ArborX::Dendrogram::alpha_incidence_matrix");
    profile_build_alpha_incidence_matrix.start();
    Kokkos::View<int *, MemorySpace> alpha_mat_offsets(
        "ArborX::Dendrogram::alpha_mat_offsets", 0);
    Kokkos::View<int *, MemorySpace> alpha_mat_edges(
        "ArborX::Dendrogram::alpha_mat_edges", 0);
    Details::buildAlphaIncidenceMatrix(exec_space, sorted_edges,
                                       alpha_edge_indices, alpha_vertices,
                                       alpha_mat_offsets, alpha_mat_edges);
    profile_build_alpha_incidence_matrix.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[5] Alpha incidence matrix:\n");
    for (int i = 0; i < (int)alpha_mat_offsets.size() - 1; ++i)
    {
      printf("%5d:", i);
      for (int j = alpha_mat_offsets(i); j < alpha_mat_offsets(i + 1); ++j)
        printf(" %5d", alpha_mat_edges(j));
      printf("\n");
    }
#endif

    // Step 6: compute sided alpha_parents
    Kokkos::Profiling::ProfilingSection profile_compute_sided_alpha_parents(
        "ArborX::Dendrogram::sided_alpha_parents");
    profile_compute_sided_alpha_parents.start();
    auto sided_alpha_parents = computeSidedAlphaParents(
        exec_space, sorted_edges, alpha_edge_indices, alpha_vertices,
        sided_alpha_parents_of_alpha, alpha_mat_offsets, alpha_mat_edges);
    profile_compute_sided_alpha_parents.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[6] Sided alpha parents:\n");
    for (int i = 0; i < num_edges; ++i)
      printf("%5d -> %5d\n", i, sided_alpha_parents(i));
#endif

    // Step 7: build full dendrogram
    Kokkos::Profiling::ProfilingSection profile_compute_sided_parents(
        "ArborX::Dendrogram::sided_parents");
    profile_compute_sided_parents.start();
    auto sided_parents =
        Details::computeSidedParents(exec_space, sided_alpha_parents);
    profile_compute_sided_parents.stop();

#ifdef VERBOSE
    printf("-------------------------------------\n");
    printf("[7] Sided parents:\n");
    for (int i = 0; i < num_edges; ++i)
      printf("%5d -> %5d\n", i, sided_parents(i));
#endif

    return sided_parents;
  }
};

} // namespace ArborX

#endif
