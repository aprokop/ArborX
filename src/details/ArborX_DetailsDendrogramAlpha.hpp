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

#ifndef ARBORX_DETAILS_DENDROGRAM_ALPHA_HPP
#define ARBORX_DETAILS_DENDROGRAM_ALPHA_HPP

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

// Sort edges in increasing order
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<unsigned int *, MemorySpace>
sortEdges(ExecutionSpace const &exec_space,
          Kokkos::View<WeightedEdge *, MemorySpace> &edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::sort_edges");

  int const num_edges = edges.size();

  Kokkos::View<float *, MemorySpace> weights(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::weights"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::copy_weights",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const edge_index) {
        weights(edge_index) = edges(edge_index).weight;
      });

  auto permute = Details::sortObjects(exec_space, weights);
  Details::applyPermutation(exec_space, permute, edges);

  Kokkos::Profiling::popRegion();

  return permute;
}

// Determine alpha edges
//
// An alpha-edge is an edge that has both children as edges in the dendrogram.
// In other words, an edge that is not an alpha edge has at most one child
// edge. Assuming the edges are sorted in the order of increasing weights, an
// edge e is an alpha-edge if both vertices have an incident edge that is
// smaller than e.
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
findAlphaEdges(ExecutionSpace const &exec_space,
               Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::find_alpha_edges");

  auto const num_edges = sorted_edges.size();
  auto const num_vertices = num_edges + 1;

  // Find smallest incident edge for each vertex
  Kokkos::View<int *, MemorySpace> smallest_vertex_incident_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::smallest_vertex_incident_edges"),
      num_vertices);
  Kokkos::deep_copy(exec_space, smallest_vertex_incident_edges, INT_MAX);

  Kokkos::parallel_for(
      "ArborX::Dendrogram::find_smallest_vertex_incident_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        auto &edge = sorted_edges(e);
        Kokkos::atomic_min(&smallest_vertex_incident_edges(edge.source), e);
        Kokkos::atomic_min(&smallest_vertex_incident_edges(edge.target), e);
      });

  Kokkos::View<int *, MemorySpace> alpha_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_edges"),
      num_edges);
  int num_alpha_edges;
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::determine_alpha_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e, int &update, bool final_pass) {
        auto &edge = sorted_edges(e);
        if (smallest_vertex_incident_edges(edge.source) < e &&
            smallest_vertex_incident_edges(edge.target) < e)
        {
          if (final_pass)
            alpha_edges(update) = e;
          ++update;
        }
      },
      num_alpha_edges);
  --num_alpha_edges;
  Kokkos::resize(alpha_edges, num_alpha_edges);

  Kokkos::Profiling::popRegion();

  return alpha_edges;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
assignAlphaVertices(ExecutionSpace const &exec_space,
                    Kokkos::View<WeightedEdge *, MemorySpace> edges,
                    Kokkos::View<int *, MemorySpace> alpha_edges)
{
  auto const num_edges = edges.size();
  auto num_vertices = num_edges + 1;
  auto num_alpha_edges = (int)alpha_edges.size();

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_vertices"),
      num_vertices);

  {
    // Do initial union-find on the subgraphs

    // TODO: may want to move this map to the higher level dendrom code
    Kokkos::View<int *, MemorySpace> marked_alpha_edges(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::marked_alpha_edges"),
        num_edges);
    Kokkos::deep_copy(exec_space, marked_alpha_edges, -1);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::mark_alpha_edges",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
        KOKKOS_LAMBDA(int i) { marked_alpha_edges(alpha_edges(i)) = i; });

    iota(exec_space, alpha_vertices);

    UnionFind<MemorySpace> union_find(alpha_vertices);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::alpha_vertices_union_find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
        KOKKOS_LAMBDA(int e) {
          if (marked_alpha_edges(e) == -1)
          {
            // Not an alpha edge, merge edge vertices
            auto &edge = edges(e);
            union_find.merge(edge.source, edge.target);
          }
        });
    // finalize union-find
    Kokkos::parallel_for(
        "ArborX::Dendrogram::finalize_union-find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int const i) {
          // ##### ECL license (see LICENSE.ECL) #####
          int next;
          int vstat = alpha_vertices(i);
          int const old = vstat;
          while (vstat > (next = alpha_vertices(vstat)))
          {
            vstat = next;
          }
          if (vstat != old)
            alpha_vertices(i) = vstat;
        });
  }

  {
    // Map found alpha-vertices back to [0, #alpha vertices) range

    Kokkos::View<int *, MemorySpace> map(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::map_back"),
        num_vertices);
    Kokkos::deep_copy(exec_space, map, -1);

    Kokkos::parallel_for(
        "ArborX::Dendrogram::find_unique_entries",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          map(alpha_vertices(i)) = 1;
        });
    int num_unique_entries = 0;
    Kokkos::parallel_scan(
        "ArborX::Dendrogram::map_scan",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
          if (map(i) != -1)
          {
            if (is_final)
              map(i) = partial_sum;
            ++partial_sum;
          }
        },
        num_unique_entries);
    ARBORX_ASSERT(num_unique_entries == num_alpha_edges + 1);

    Kokkos::parallel_for(
        "ArborX::Dendrogram::remap_alpha_vertices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          alpha_vertices(i) = map(alpha_vertices(i));
        });
  }

  return alpha_vertices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<WeightedEdge *, MemorySpace>
buildAlphaMST(ExecutionSpace const &exec_space,
              Kokkos::View<WeightedEdge *, MemorySpace> edges,
              Kokkos::View<int *, MemorySpace> alpha_edge_indices,
              Kokkos::View<int *, MemorySpace> alpha_vertices)
{
  auto const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<WeightedEdge *, MemorySpace> alpha_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_mst_edges"),
      num_alpha_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_alpha_mst_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int i) {
        int e = alpha_edge_indices(i);
        alpha_edges(i) = {alpha_vertices(edges(e).source),
                          alpha_vertices(edges(e).target), edges(i).weight};
      });
  return alpha_edges;
}

#if 0
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<WeightedEdge *, MemorySpace>
findAlphaParents(ExecutionSpace const &exec_space,
                 Kokkos::View<WeightedEdge *, MemorySpace> edges,
                 IncidenceMatrix<MemorySpace> const &alpha_indidence_matrix,
                 Kokkos::View<int *, MemorySpace> alpha_vertices,
                 Kokkos::View<int *, MemorySpace> alpha_parents_of_alpha)

{
  auto num_edges = edges.size();

  Kokkos::parallel_for(
      "ArborX::Dendrogram::insert_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);
        // Determine alpha-neighborhood
        auto alpha_vertex = alpha_vertices(edge.source);
        if (alpha_vertices(edge.target) == alpha)
        {
          // Edge is not an alpha-edge

          int largest_smaller = inf;
          int smallest_larger = -1;
          for (int k = alpha_incidence_matrix._incident_offsets(alpha_vertex);
               k < alpha_incidence_matrix._incident_offsets(alpha_vertex + 1);
               ++k)
          {
            auto alpha_e = alpha_incidence_matrix._incident_edge(k);
            if (alpha_e > e && alpha_e < largest_smaller)
              largest_smaller = alpha_e;
            if (alpha_e < e && alpha_e > smallest_larger)
              smallest_larger = alpha_e;
          }

          if (largest_smaller == inf)
          {
            if (smallest_larger >= 0)
              alpha_parents(e) = smallest_larger;
            else
              alpha_parents(e) = -1;
          }
          else
          {
            do
            {
              largest_smaller = alpha_parents(largest_smaller);
            } while (largest_smaller != -1 &&
                     largest_smaller < smallest_larger);
            // FIXME
          }
        }
      });
}
#endif

} // namespace ArborX::Details

#endif
