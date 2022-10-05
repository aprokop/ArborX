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

#include <ArborX_DetailsEulerTour.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>

#define VERBOSE

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
  Kokkos::resize(alpha_edges, num_alpha_edges);

  Kokkos::Profiling::popRegion();

  return alpha_edges;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
assignAlphaVertices(ExecutionSpace const &exec_space,
                    Kokkos::View<WeightedEdge *, MemorySpace> edges,
                    Kokkos::View<int *, MemorySpace> alpha_edge_indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::assign_alpha_vertices");

  Kokkos::View<int *, MemorySpace> alpha_inverse_map(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::inverse_map"),
      edges.size());
  Kokkos::deep_copy(exec_space, alpha_inverse_map, -1);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::update_inverse_map",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                          alpha_edge_indices.size()),
      KOKKOS_LAMBDA(int i) { alpha_inverse_map(alpha_edge_indices(i)) = i; });

  auto const num_edges = edges.size();
  auto num_vertices = num_edges + 1;
  auto num_alpha_edge_indices = (int)alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_vertices"),
      num_vertices);

  {
    // Do initial union-find on the subgraphs

    iota(exec_space, alpha_vertices);

    UnionFind<MemorySpace> union_find(alpha_vertices);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::alpha_vertices_union_find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
        KOKKOS_LAMBDA(int e) {
          if (alpha_inverse_map(e) == -1)
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
    ARBORX_ASSERT(num_unique_entries == num_alpha_edge_indices + 1);

    Kokkos::parallel_for(
        "ArborX::Dendrogram::remap_alpha_vertices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          alpha_vertices(i) = map(alpha_vertices(i));
        });
  }

  Kokkos::Profiling::popRegion();

  return alpha_vertices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<WeightedEdge *, MemorySpace>
buildAlphaMST(ExecutionSpace const &exec_space,
              Kokkos::View<WeightedEdge *, MemorySpace> edges,
              Kokkos::View<int *, MemorySpace> alpha_edge_indices,
              Kokkos::View<int *, MemorySpace> alpha_vertices)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::build_alpha_mst");

  auto const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<WeightedEdge *, MemorySpace> alpha_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_mst_edges"),
      num_alpha_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_alpha_mst_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int k) {
        auto &edge = edges(alpha_edge_indices(k));
        alpha_edges(k) = {alpha_vertices(edge.source),
                          alpha_vertices(edge.target), edge.weight};
      });

  Kokkos::Profiling::popRegion();

  return alpha_edges;
}

template <typename ExecutionSpace, typename MemorySpace>
void buildAlphaIncidenceMatrix(
    ExecutionSpace const &exec_space,
    Kokkos::View<WeightedEdge *, MemorySpace> edges,
    Kokkos::View<int *, MemorySpace> alpha_edge_indices,
    Kokkos::View<int *, MemorySpace> alpha_vertices,
    Kokkos::View<int *, MemorySpace> &alpha_mat_offsets,
    Kokkos::View<int *, MemorySpace> &alpha_mat_edges)
{
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::build_alpha_incidence_matrix");

  auto num_alpha_edges = alpha_edge_indices.size();
  auto num_vertices = num_alpha_edges + 1;

  Kokkos::realloc(alpha_mat_offsets, num_vertices + 1);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_alpha_mat_counts",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int const ee) {
        auto const &edge = edges(alpha_edge_indices(ee));

        auto const i = alpha_vertices(edge.source);
        auto const j = alpha_vertices(edge.target);

        Kokkos::atomic_increment(&alpha_mat_offsets(i));
        Kokkos::atomic_increment(&alpha_mat_offsets(j));
      });
  exclusivePrefixSum(exec_space, alpha_mat_offsets);

  ARBORX_ASSERT(KokkosExt::lastElement(exec_space, alpha_mat_offsets) ==
                2 * (int)num_alpha_edges);

  KokkosExt::reallocWithoutInitializing(
      exec_space, alpha_mat_edges,
      KokkosExt::lastElement(exec_space, alpha_mat_offsets));

  auto offsets = KokkosExt::clone(exec_space, alpha_mat_offsets);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_alpha_mat_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int const ee) {
        auto const e = alpha_edge_indices(ee); // original edge index
        auto const &edge = edges(e);

        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.source)), 1)) = e;
        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.target)), 1)) = e;
      });

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename MemorySpace>
void updateSidedParents(ExecutionSpace const &exec_space,
                        Kokkos::View<WeightedEdge *, MemorySpace> edges,
                        Kokkos::View<int *, MemorySpace> alpha_vertices,
                        Kokkos::View<int *, MemorySpace> &alpha_mat_offsets,
                        Kokkos::View<int *, MemorySpace> &alpha_mat_edges,
                        Kokkos::View<int *, MemorySpace> global_map,
                        Kokkos::View<int *, MemorySpace> &sided_level_parents,
                        Kokkos::View<int *, MemorySpace> &follow)
{
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::compute_alpha_inverse_map");

  Kokkos::parallel_for(
      "ArborX::Dendrogram::update_sided_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);

        auto alpha_vertex = alpha_vertices(edge.source);
        if (alpha_vertices(edge.target) != alpha_vertex)
        {
          // This is an alpha-edge, skip
          return;
        }

        int largest_smaller = -1;
        int smallest_larger = INT_MAX;
        for (int k = alpha_mat_offsets(alpha_vertex);
             k < alpha_mat_offsets(alpha_vertex + 1); ++k)
        {
          auto alpha_e = alpha_mat_edges(k);
          if (alpha_e < e && alpha_e > largest_smaller)
            largest_smaller = alpha_e;
          if (alpha_e > e && alpha_e < smallest_larger)
            smallest_larger = alpha_e;
        }
#ifdef VERBOSE
        printf("e = %d, largest_smaller = %d, smallest_larger = %d\n", e,
               largest_smaller, smallest_larger);
#endif
        assert(largest_smaller != INT_MAX || smallest_larger != -1);

        if (largest_smaller == -1 && smallest_larger != INT_MAX)
        {
          // No smaller incident alpha-edge.
          // Can immediately assign the parent.
          auto const &alpha_edge = edges(smallest_larger);
#ifdef VERBOSE
          printf("alpha vertex = %d, smallest_larger = (%d, %d)\n",
                 alpha_vertex, alpha_edge.source, alpha_edge.target);
#endif

          bool is_left_side =
              (alpha_vertices(alpha_edge.source) == alpha_vertex);
#ifdef VERBOSE
          printf("is left side = %s\n", (is_left_side ? "yes" : "no"));
#endif
          sided_level_parents(global_map(e)) =
              2 * global_map(smallest_larger) + static_cast<int>(is_left_side);
        }
        else if (largest_smaller != -1)
        {
#ifdef VERBOSE
          printf("%d => %d\n", global_map(e), global_map(largest_smaller));
#endif
          if (smallest_larger != INT_MAX)
          {
            // Store the current candidate and follow the other
            bool is_left_side =
                (alpha_vertices(edges(smallest_larger).source) == alpha_vertex);
            sided_level_parents(global_map(e)) =
                2 * global_map(smallest_larger) +
                static_cast<int>(is_left_side);
          }
          follow(global_map(e)) = global_map(largest_smaller);
        }
      });
  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
updateGlobalMap(ExecutionSpace const &exec_space,
                Kokkos::View<int *, MemorySpace> global_map,
                Kokkos::View<int *, MemorySpace> alpha_edge_indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::update_global_map");

  auto const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> new_global_map(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::global_map"),
      num_alpha_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::update_inverse_map",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int i) {
        new_global_map(i) = global_map(alpha_edge_indices(i));
      });

  Kokkos::Profiling::popRegion();

  return new_global_map;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeParents(ExecutionSpace const &exec_space,
               Kokkos::View<int *, MemorySpace> sided_level_parents)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::compute_parents");

  auto num_edges = sided_level_parents.size();

  // Encode both sided parent and edge into long long
  // This way, once we sort based on this value, edges with the same sided
  // parent will already be sorted in increasing order.
  Kokkos::View<long long *, MemorySpace> keys(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::keys"),
      num_edges);

  constexpr int shift = 32;

  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_sided_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        long long key = sided_level_parents(e);
        if (key == -1)
          key = INT_MAX;
        keys(e) = (key << shift) + e;
      });

  auto permute = sortObjects(exec_space, keys);

  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::parents"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const i) {
        int e = permute(i);
        if (i == (int)num_edges - 1)
          parents(e) = -1;
        else if ((keys(i) >> shift) == (keys(i + 1) >> shift))
          parents(e) = permute(i + 1);
        else
          parents(e) = (keys(i) >> shift) / 2;
      });

  Kokkos::Profiling::popRegion();

  return parents;
}

} // namespace ArborX::Details

#endif
