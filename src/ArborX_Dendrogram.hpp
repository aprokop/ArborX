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

#include <ArborX_DetailsDendrogram.hpp>
#include <ArborX_DetailsEulerTour.hpp>
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
  ALPHA_NEW,
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

    Kokkos::View<int *, MemorySpace> edge_parents;
    switch (impl)
    {
    case DendrogramImplementation::UNION_FIND:
      edge_parents = unionFind(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::ALPHA:
      edge_parents = alpha(exec_space, sorted_edges);
      break;
    case DendrogramImplementation::ALPHA_NEW:
      edge_parents = alphaNew(exec_space, sorted_edges);
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

  template <typename ExecutionSpace, typename Edges>
  auto alpha(ExecutionSpace const &exec_space, Edges sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha");

    auto const num_edges = sorted_edges.extent_int(0);

    // Step 1: compute Euler tour for the original MST
    Kokkos::Profiling::ProfilingSection profile_euler_tour(
        "ArborX::Dendrogram::euler_tour");
    profile_euler_tour.start();
    // The returned Euler tour is of size twice the number of edges. Each pair
    // of entries {2*i, 2*i+1} correspond to the edge i, in two directions (one
    // going down, one up).
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::euler_tour");
    auto euler_tour = Details::eulerTour(exec_space, sorted_edges);
    Kokkos::Profiling::popRegion();

    // Steps 1.5: make sure the first entry for every edge is the start entry
    // (i.e., the smaller one)
    Kokkos::parallel_for(
        "ArborX::euler_tour::order_start_end",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                            euler_tour.extent(0) / 2),
        KOKKOS_LAMBDA(int k) {
          using KokkosExt::swap;
          int i = 2 * k;
          if (euler_tour(i) > euler_tour(i + 1))
            swap(euler_tour(i), euler_tour(i + 1));
        });
    profile_euler_tour.stop();

    // Step 2: construct edge incident matrix (vertex -> incident edges)
    Kokkos::Profiling::ProfilingSection profile_build_incidence_matrix(
        "ArborX::Dendrogram::build_incidence_matrix");
    profile_build_incidence_matrix.start();
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::build_incidence_matrix");
    Details::IncidenceMatrix<MemorySpace> incidence_matrix(exec_space,
                                                           sorted_edges);
    Kokkos::Profiling::popRegion();
    profile_build_incidence_matrix.stop();

    // Step 3: find alpha edges of the original MST
    Kokkos::Profiling::ProfilingSection profile_compute_alpha_edges(
        "ArborX::Dendrogram::compute_alpha_edges");
    profile_compute_alpha_edges.start();
    auto alpha_edge_indices =
        Details::findAlphaEdges(exec_space, incidence_matrix);
    profile_compute_alpha_edges.start();

    auto num_alpha_edges = alpha_edge_indices.extent_int(0);
    printf("#alpha edges: %d [%.2f%%]\n", num_alpha_edges,
           (100.f * num_alpha_edges) / num_edges);
#ifdef VERBOSE
    printf("alpha edges:\n");
    for (int i = 0; i < num_alpha_edges; ++i)
    {
      int e = alpha_edge_indices(i);
      printf("[%i] : %d (%d, %d)\n", i, e, sorted_edges(e).source,
             sorted_edges(e).target);
    }
    printf("\n");
#endif

    // Step 4: assign alpha-vertices
    Kokkos::Profiling::ProfilingSection profile_alpha_vertices(
        "ArborX::Dendrogram::alpha_vertices");
    profile_alpha_vertices.start();
    auto alpha_vertices = Details::assignAlphaVertices(exec_space, euler_tour,
                                                       alpha_edge_indices);
    profile_alpha_vertices.stop();

    // Step 5: construct alpha-MST
    Kokkos::Profiling::ProfilingSection profile_alpha_mst(
        "ArborX::Dendrogram::alpha_mst");
    profile_alpha_mst.start();
    auto alpha_mst_edges =
        Details::buildAlphaEdges(exec_space, sorted_edges, euler_tour,
                                 alpha_edge_indices, alpha_vertices);
    profile_alpha_mst.stop();

// #define VERBOSE
#ifdef VERBOSE
    printf("alpha mst:\n");
    for (int i = 0; i < num_alpha_edges; ++i)
    {
      printf("[%i] : (%d, %d)\n", i, alpha_mst_edges(i).source,
             alpha_mst_edges(i).target);
    }
    printf("\n");
#endif

    // Step 6: build dendrogram of the alpha-tree
    Kokkos::Profiling::ProfilingSection profile_dendrogram_alpha(
        "ArborX::Dendrogram::dendrogram_alpha");
    profile_dendrogram_alpha.start();
    Dendrogram<MemorySpace> dendrogram_alpha(
        exec_space, alpha_mst_edges, DendrogramImplementation::UNION_FIND);
    auto alpha_parents_of_alpha = dendrogram_alpha._edge_parents;
    profile_dendrogram_alpha.stop();

    // auto alpha_sided_parents =
    // findAlphaParents(exec_space, sorted_edges, alpha_parents_of_alpha);

    // auto permute = sortObjects(alpha_sided_parents);

    Kokkos::Profiling::popRegion();

    Kokkos::View<int *, MemorySpace> edge_parents(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::edge_parents"),
        num_edges);

    return edge_parents;
  }

  template <typename ExecutionSpace, typename Edges>
  auto alphaNew(ExecutionSpace const &exec_space, Edges sorted_edges)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha_new");

    auto const num_edges = sorted_edges.extent_int(0);

    // Step 1: construct edge incident matrix (vertex -> incident edges)
    Kokkos::Profiling::ProfilingSection profile_build_incidence_matrix(
        "ArborX::Dendrogram::build_incidence_matrix");
    profile_build_incidence_matrix.start();
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::build_incidence_matrix");
    Details::IncidenceMatrix<MemorySpace> incidence_matrix(exec_space,
                                                           sorted_edges);
    Kokkos::Profiling::popRegion();
    profile_build_incidence_matrix.stop();

    // Step 2: find alpha edges of the original MST
    Kokkos::Profiling::ProfilingSection profile_compute_alpha_edges(
        "ArborX::Dendrogram::compute_alpha_edges");
    profile_compute_alpha_edges.start();
    auto alpha_edge_indices =
        Details::findAlphaEdges(exec_space, incidence_matrix);
    profile_compute_alpha_edges.start();

    auto num_alpha_edges = alpha_edge_indices.extent_int(0);
    printf("#alpha edges: %d [%.2f%%]\n", num_alpha_edges,
           (100.f * num_alpha_edges) / num_edges);
#ifdef VERBOSE
    printf("alpha edges:\n");
    for (int i = 0; i < num_alpha_edges; ++i)
    {
      int e = alpha_edge_indices(i);
      printf("[%i] : %d (%d, %d)\n", i, e, sorted_edges(e).source,
             sorted_edges(e).target);
    }
    printf("\n");
#endif

    // Step 3: assign alpha-vertices through union-find
    Kokkos::Profiling::ProfilingSection profile_alpha_vertices(
        "ArborX::Dendrogram::alpha_vertices");
    profile_alpha_vertices.start();
    auto alpha_vertices = Details::assignAlphaVerticesNew(
        exec_space, sorted_edges, alpha_edge_indices);
    profile_alpha_vertices.stop();

    Kokkos::View<int *, MemorySpace> edge_parents(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::edge_parents"),
        num_edges);

    return edge_parents;
  }
};

} // namespace ArborX

#endif
