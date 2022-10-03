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
      KOKKOS_LAMBDA(int k) {
        using KokkosExt::swap;

        int e = alpha_edge_indices(k);
        auto &edge = edges(e);

        // Make sure that the smaller vertex is first, and the larger one is
        // second. This will be helpful when working with sideness.
        auto i = alpha_vertices(edge.source);
        auto j = alpha_vertices(edge.target);
        if (i > j)
          swap(i, j);

        alpha_edges(k) = {i, j, edge.weight};
      });
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
  auto num_alpha_edges = alpha_edge_indices.size();
  auto num_vertices = num_alpha_edges + 1;

  Kokkos::realloc(alpha_mat_offsets, num_vertices + 1);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_alpha_mat_counts",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int const ee) {
        auto const e = alpha_edge_indices(ee); // original edge index
        auto const &edge = edges(e);

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
        using KokkosExt::swap;

        auto const e = alpha_edge_indices(ee); // original edge index
        auto const &edge = edges(e);

        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.source)), 1)) = e;
        alpha_mat_edges(Kokkos::atomic_fetch_add(
            &offsets(alpha_vertices(edge.target)), 1)) = e;
      });
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeAlphaParents(ExecutionSpace const &exec_space,
                    Kokkos::View<WeightedEdge *, MemorySpace> edges,
                    Kokkos::View<int *, MemorySpace> alpha_edge_indices,
                    Kokkos::View<int *, MemorySpace> alpha_vertices,
                    Kokkos::View<int *, MemorySpace> alpha_parents_of_alpha,
                    Kokkos::View<int *, MemorySpace> alpha_mat_offsets,
                    Kokkos::View<int *, MemorySpace> alpha_mat_edges)

{
  auto num_edges = edges.size();
  auto num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> inverse_alpha_map(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DBSCAN::inverse_alpha_map"),
      num_edges);
  Kokkos::deep_copy(exec_space, inverse_alpha_map, -1);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_inverse_alpha_map",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int alpha_e) {
        inverse_alpha_map(alpha_edge_indices(alpha_e)) = alpha_e;
      });

  constexpr int COMPRESSION_LEVEL = 3;
  constexpr int COMPRESSION_STEP = 2;

  Kokkos::View<int *[COMPRESSION_LEVEL], MemorySpace>
      compressed_alpha_parents_of_alpha(
          Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                             "ArborX::Dendrogram::compressed_alpha_parents"),
          num_alpha_edges);

  Kokkos::parallel_for(
      "ArborX::Dendrogram::compress_alpha_tree_0",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int e) {
        compressed_alpha_parents_of_alpha(e, 0) = alpha_parents_of_alpha(e);
      });
  for (int level = 1; level < COMPRESSION_LEVEL; ++level)
  {
    Kokkos::parallel_for(
        "ArborX::Dendrogram::compress_alpha_tree_" + std::to_string(level),
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
        KOKKOS_LAMBDA(int e) {
          int alpha_parent;
          int k = 0;
          do
          {
            alpha_parent = compressed_alpha_parents_of_alpha(e, level - 1);
            if (alpha_parent != -1)
              alpha_parent = compressed_alpha_parents_of_alpha(
                  inverse_alpha_map(alpha_parent), level - 1);
            ++k;
          } while (alpha_parent != -1 && k < COMPRESSION_STEP);
          compressed_alpha_parents_of_alpha(e, level) = alpha_parent;
        });
#if 0
    printf("-------------------------------------\n");
    printf("[4] Parents of alpha edges (%d):\n",
           (int)std::pow(COMPRESSION_STEP, level));
    for (int i = 0; i < (int)num_alpha_edges; ++i)
      printf("%5d [%5d] -> %5d\n", i, alpha_edge_indices(i),
             compressed_alpha_parents_of_alpha(i, level));
    fflush(stdout);
#endif
  }

  Kokkos::View<int *, MemorySpace> alpha_parents(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DBSCAN::alpha_parents"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);

        // printf("e = %d: ", e);

        auto alpha_vertex = alpha_vertices(edge.source);
        if (alpha_vertices(edge.target) != alpha_vertex)
        {
          // This is an alpha-edge
          // printf("skipping alpha-edge\n");
          alpha_parents(e) = alpha_parents_of_alpha(inverse_alpha_map(e));
          return;
        }
        // printf("processing non alpha-edge\n");

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
        // printf("largest_smaller = %d, smallest_larger = %d\n",
        // largest_smaller, smallest_larger);

        assert(largest_smaller != INT_MAX || smallest_larger != -1);

        if (largest_smaller == -1)
        {
          // No smaller incident alpha-edge.
          // Can immediately assign the parent.

          if (smallest_larger == INT_MAX)
          {
            // No larger incident alpha-edge.
            // This edge will at the top chain of the dendrogram.
            alpha_parents(e) = -1;
          }
          else
          {
            alpha_parents(e) = smallest_larger;
          }
          return;
        }

        do
        {
          // std::cout << largest_smaller << " -> "
          // << alpha_parents_of_alpha(
          // inverse_alpha_map(largest_smaller))
          // << std::endl;
          auto a_e = inverse_alpha_map(largest_smaller);
          for (int level = COMPRESSION_LEVEL - 1; level >= 0; --level)
          {
            auto candidate = compressed_alpha_parents_of_alpha(a_e, level);
            if (level == 0 || (candidate != -1 && candidate < e))
            {
              largest_smaller = candidate;
              break;
            }
          }
        } while (largest_smaller < e && largest_smaller != -1);

        if (largest_smaller > smallest_larger)
          alpha_parents(e) = smallest_larger;
        else if (largest_smaller != -1)
          alpha_parents(e) = largest_smaller;
        else
          alpha_parents(e) =
              (smallest_larger == INT_MAX ? -1 : smallest_larger);
      });

  return alpha_parents;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
computeParents(ExecutionSpace const &exec_space,
               Kokkos::View<WeightedEdge *, MemorySpace> edges,
               Kokkos::View<int *, MemorySpace> alpha_parents)
{
  auto num_edges = alpha_parents.size();

  Kokkos::Profiling::ProfilingSection profile_euler_tour(
      "ArborX::Dendrogram::euler_tour");
  profile_euler_tour.start();
  // The returned Euler tour is of size twice the number of edges. Each pair
  // of entries {2*i, 2*i+1} correspond to the edge i, in two directions
  // (one going down, one up).
  auto euler_tour = eulerTour(exec_space, edges);

  // Make sure the first entry for every edge is the start entry (i.e., the
  // smaller one)
  Kokkos::parallel_for(
      "ArborX::Dendrogram::order_start_end_in_euler_tour",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                          euler_tour.extent(0) / 2),
      KOKKOS_LAMBDA(int k) {
        using KokkosExt::swap;
        int i = 2 * k;
        if (euler_tour(i) > euler_tour(i + 1))
          swap(euler_tour(i), euler_tour(i + 1));
      });

  profile_euler_tour.stop();

  auto sided_alpha_parents = KokkosExt::clone(exec_space, alpha_parents);

  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_sided_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        auto alpha_parent = alpha_parents(e);
        if (alpha_parent == -1)
        {
          sided_alpha_parents(e) = INT_MAX;
          return;
        }

        if (euler_tour(2 * e) > euler_tour(2 * alpha_parent) &&
            euler_tour(2 * e + 1) < euler_tour(2 * alpha_parent + 1))
          sided_alpha_parents(e) = 2 * alpha_parent + 1;
        else
          sided_alpha_parents(e) = 2 * alpha_parent + 0;
      });

#if 0
  printf("-------------------------------------\n");
  printf("Sided alpha parents:\n");
  for (int i = 0; i < (int)num_edges; ++i)
    printf("%5d -> %5d\n", i, sided_alpha_parents(i));
  fflush(stdout);
#endif

  auto permute = sortObjects(exec_space, sided_alpha_parents);

  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::sided_parents"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const i) {
        int e = permute(i);
        if (i == (int)num_edges - 1)
          parents(e) = -1;
        else if (sided_alpha_parents(i) == sided_alpha_parents(i + 1))
          parents(e) = permute(i + 1);
        else
          parents(e) = sided_alpha_parents(i) / 2;
      });
  return parents;
}

} // namespace ArborX::Details

#endif
