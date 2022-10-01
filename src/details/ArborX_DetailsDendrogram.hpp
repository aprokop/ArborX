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

#ifndef ARBORX_DETAILS_DENDROGRAM_HPP
#define ARBORX_DETAILS_DENDROGRAM_HPP

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <typename MemorySpace>
struct IncidenceMatrix
{
  Kokkos::View<WeightedEdge *, MemorySpace> _edges;
  Kokkos::View<int *, MemorySpace> _incident_offsets;
  Kokkos::View<int *, MemorySpace>
      _incident_edges; // edges incident to a specific vertex

  template <typename ExecutionSpace, typename Edges>
  IncidenceMatrix(ExecutionSpace const &exec_space, Edges const &edges)
      : _edges(edges)
  {
    buildIncidenceMatrix(exec_space, edges);
  }

  template <typename ExecutionSpace, typename Edges>
  void buildIncidenceMatrix(ExecutionSpace const &exec_space,
                            Edges const &edges)
  {
    int const n = edges.extent(0) + 1;

    Kokkos::realloc(_incident_offsets, n + 1);
    auto &incident_offsets = _incident_offsets; // FIXME avoid capture of *this
    Kokkos::parallel_for(
        "ArborX::Dendrogram::compute_incident_counts",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
        KOKKOS_LAMBDA(int const edge_index) {
          auto const &edge = edges(edge_index);
          Kokkos::atomic_increment(&incident_offsets(edge.source));
          Kokkos::atomic_increment(&incident_offsets(edge.target));
        });
    exclusivePrefixSum(exec_space, _incident_offsets);

    ARBORX_ASSERT(KokkosExt::lastElement(exec_space, _incident_offsets) ==
                  2 * (n - 1));

    KokkosExt::reallocWithoutInitializing(
        exec_space, _incident_edges,
        KokkosExt::lastElement(exec_space, _incident_offsets));

    auto offsets = KokkosExt::clone(exec_space, _incident_offsets);
    auto &incident_edges = _incident_edges; // FIXME avoid capture of *this
    Kokkos::parallel_for(
        "ArborX::Dendrogram::compute_incident_edges",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
        KOKKOS_LAMBDA(int const edge_index) {
          auto const &edge = edges(edge_index);
          incident_edges(Kokkos::atomic_fetch_add(&offsets(edge.source), 1)) =
              edge_index;
          incident_edges(Kokkos::atomic_fetch_add(&offsets(edge.target), 1)) =
              edge_index;
        });
  }

  template <typename ExecutionSpace>
  void degrees(ExecutionSpace const &exec_space)
  {
    int const n = _edges.extent(0) + 1;

    int max_degree = 0;
    auto &incident_offsets = _incident_offsets; // FIXME avoid capture of *this
    Kokkos::parallel_reduce(
        "ArborX::HDBSCA::max_offset",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          int degree = incident_offsets(i + 1) - incident_offsets(i);
          if (degree > update)
            update = degree;
        },
        Kokkos::Max<int>(max_degree));

    Kokkos::View<int *, MemorySpace> degrees_hist("ArborX::Dendrogram::degrees",
                                                  max_degree);
    Kokkos::parallel_for(
        "blah", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          int degree = incident_offsets(i + 1) - incident_offsets(i);
          Kokkos::atomic_fetch_add(&degrees_hist(degree), 1);
        });
    auto degrees_hist_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, degrees_hist);
    printf("Degrees distribution:\n");
    for (int i = 1; i < max_degree; ++i)
      printf("  %d: %8d [%.5f]\n", i, degrees_hist_host(i),
             (100.f * degrees_hist_host(i)) / n);
    return;
  }

  template <typename ExecutionSpace>
  void print(std::ostream &os)
  {
    int const n = _edges.extent(0) + 1;

    auto incident_offsets_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _incident_offsets);
    auto incident_edges_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, _incident_edges);

    os << "Incidence matrix:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
      os << i << ":";
      for (int j = incident_offsets_host(i); j < incident_offsets_host(i + 1);
           ++j)
        os << " " << incident_edges_host(j);
      os << std::endl;
    }
  }
};

template <typename ExecutionSpace, typename Edges>
Kokkos::View<unsigned int *, typename Edges::memory_space>
sortEdges(ExecutionSpace const &exec_space, Edges &edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::sort_edges");
  using MemorySpace = typename Edges::memory_space;

  // To sort in decreasing order, we use negative weights
  Kokkos::View<float *, MemorySpace> negative_weights(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::mst_distances"),
      edges.extent(0));
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_edges_distances",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
      KOKKOS_LAMBDA(int const edge_index) {
        negative_weights(edge_index) = -edges(edge_index).weight;
      });

  auto permute = Details::sortObjects(exec_space, negative_weights);
  Details::applyPermutation(exec_space, permute, edges);

  Kokkos::Profiling::popRegion();

  return permute;
}

struct WangUnionFind
{
  std::vector<int> _parents;

  // initialize n elements all as roots
  WangUnionFind(int n) { _parents.resize(n, -1); }

  int representative(int i)
  {
    if (is_root(i))
      return i;
    int p = _parents[i];
    if (is_root(p))
      return p;

    // find root, shortcutting along the way
    do
    {
      int gp = _parents[p];
      _parents[i] = gp;
      i = p;
      p = gp;
    } while (!is_root(p));
    return p;
  }

  bool is_root(int u) { return _parents[u] == -1; }

  // TODO: This is different from the original Wang's code. There, the code
  // simply did
  //    _parents[u] = v;
  // I don't understand the preconditions for this to work. Lets say we have 3
  // points, 0, 1, 2, and two edges, [0, 1] and [0, 2]. If we use the original
  // function, both 1 and 2 would be roots, and 0 would point to 2. So,
  // representative(1) != representative(2).
  //
  // They talk a bit about cycles or always linking from larger vertex id to a
  // smaller one. Don't understand what that means.
  void merge(int u, int v) { _parents[representative(u)] = v; }
};

template <typename ExecutionSpace, typename MST>
Kokkos::View<int *, typename MST::memory_space>
dendrogramUnionFind(ExecutionSpace const &exec_space, MST sorted_mst_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");

  using MemorySpace = typename MST::memory_space;

  int const n = sorted_mst_edges.extent_int(0) + 1;

  Kokkos::View<int *, MemorySpace> edge_parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::edge_parents"),
      n - 1);

  Kokkos::View<int *, MemorySpace> representative_edges(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::representative_edges"),
      n);
  Kokkos::deep_copy(representative_edges, -1);

  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::copy_to_host");

  auto sorted_mst_edges_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, sorted_mst_edges);
  auto edge_parents_host = Kokkos::create_mirror_view(edge_parents);
  auto representative_edges_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, representative_edges);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::union_find");

#if 1
  std::cout << "Running Wang's union-find" << std::endl;
  WangUnionFind union_find(n);
  std::ignore = exec_space;
#else
  std::cout << "Running ArborX's union-find" << std::endl;
  Kokkos::View<int *, MemorySpace> vertex_labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::vertex_labels"),
      n);
  iota(exec_space, vertex_labels);
  auto vertex_labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vertex_labels);
  Details::UnionFind<Kokkos::HostSpace> union_find(vertex_labels_host);
#endif
  for (int edge_index = n - 2; edge_index >= 0; --edge_index)
  {
    int i = sorted_mst_edges_host(edge_index).source;
    int j = sorted_mst_edges_host(edge_index).target;

    for (int k : {i, j})
    {
      auto edge_child = representative_edges_host(union_find.representative(k));
      if (edge_child != -1)
        edge_parents_host(edge_child) = edge_index;
    }

    union_find.merge(i, j);

    representative_edges_host(union_find.representative(i)) = edge_index;
  }
  edge_parents_host(0) = -1;

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram::copy_to_device");

  Kokkos::deep_copy(edge_parents, edge_parents_host);

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return edge_parents;
}

template <typename Permute, typename InvPermute>
void printSortedEuler(Permute permute, InvPermute inv_permute,
                      std::string const &label = "")
{
  printf("sorted euler tour (%s)\n", label.c_str());

  auto permute_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, permute);
  auto inv_permute_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inv_permute);

  auto is_start_edge = [&inv_permute_host, &permute_host](int k) {
    using KokkosExt::min;
    int e = permute_host(k) / 2;
    int m = min(inv_permute_host(2 * e), inv_permute_host(2 * e + 1));
    return k == m;
  };

  int shift = 0;
  bool first_close;
  for (int i = 0; i < permute.extent_int(0); ++i)
  {
    int e = permute_host(i) / 2;
    if (is_start_edge(i))
    {
      ++shift;
      printf("\n%3d %*c [", e, shift, ' ');
      first_close = true;
    }
    else
    {
      if (first_close)
      {
        printf("]");
        first_close = false;
      }
      else
      {
        printf("\n%3d %*c ]", e, shift, ' ');
      }
      --shift;
    }
  }
  printf("\n");
}

// Determine alpha edges
//
// An alpha-edge is an edge that has both children as edges in the
// dendrogram. In other words, an edge that is not an alpha edge has at most
// one child edge.
// Assuming the edges are sorted in the order of decreasing weights (i.e.,
// the first edge has the largest weight), an edge e is an alpha-edge if both
// vertices have an incident edge (different from e) that is larger (in
// index) than e.
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
findAlphaEdges(ExecutionSpace const &exec_space,
               IncidenceMatrix<MemorySpace> incidence_matrix)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::find_alpha_edges");

  auto const &incident_offsets = incidence_matrix._incident_offsets;
  auto const &incident_edges = incidence_matrix._incident_edges;
  auto const &sorted_edges = incidence_matrix._edges;

  auto const num_edges = sorted_edges.extent_int(0);

  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::alpha_edges");
  Kokkos::View<int *, MemorySpace> alpha_edge_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_edge_indices"),
      num_edges);
  int num_alpha_edges;
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::determine_alpha_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const edge, int &update, bool final_pass) {
        int vertices[2] = {sorted_edges(edge).source,
                           sorted_edges(edge).target};
        for (int k = 0; k < 2; ++k)
        {
          int v = vertices[k];
          bool found_larger_edge = false;
          for (int j = incident_offsets(v); j < incident_offsets(v + 1); ++j)
            if (incident_edges(j) > edge)
            {
              found_larger_edge = true;
              break;
            }
          if (!found_larger_edge)
            return;
        }

        if (final_pass)
          alpha_edge_indices(update) = edge;
        ++update;
      },
      num_alpha_edges);
  --num_alpha_edges;
  Kokkos::resize(alpha_edge_indices, num_alpha_edges);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return alpha_edge_indices;
}

enum Bracket
{
  NO_BRACKET = 0,
  OPENING_BRACKET = 1,
  CLOSING_BRACKET = 2
};

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
assignAlphaVertices(ExecutionSpace const &exec_space,
                    Kokkos::View<int *, MemorySpace> euler_tour,
                    Kokkos::View<int *, MemorySpace> alpha_edge_indices)
{
  ARBORX_ASSERT(euler_tour.size() % 2 == 0);

  int const num_edges = euler_tour.size() / 2;
  int const num_alpha_edges = alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> brackets(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::brackets"),
      euler_tour.size());
  Kokkos::View<int *, MemorySpace> matching_bracket(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::match"),
      euler_tour.size());
  Kokkos::deep_copy(exec_space, brackets, NO_BRACKET);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_bracket_array",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int k) {
        int alpha_edge = alpha_edge_indices(k);
        int open = euler_tour(2 * alpha_edge + 0);
        int close = euler_tour(2 * alpha_edge + 1);
        brackets(open) = OPENING_BRACKET;
        brackets(close) = CLOSING_BRACKET;
        matching_bracket(close) = open;
      });

  Kokkos::View<int *, MemorySpace> opening_bracket_counts(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::opening_bracket_counts"),
      euler_tour.size());
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::count_opening_brackets",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * num_edges),
      KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
        if (brackets(i) == OPENING_BRACKET)
          ++partial_sum;
        if (is_final)
          opening_bracket_counts(i) = partial_sum;
      });

  Kokkos::View<int *, MemorySpace> closing_bracket_counts(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::closing_bracket_counts"),
      euler_tour.size());
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::count_closing_brackets",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * num_edges),
      KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
        if (brackets(i) == CLOSING_BRACKET)
          ++partial_sum;
        if (is_final)
          closing_bracket_counts(i) = partial_sum;
      });

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_vertices"),
      euler_tour.size());
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::assign_alpha_vertices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * num_edges),
      KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
        if (is_final)
          alpha_vertices(i) = partial_sum;

        if (brackets(i) == OPENING_BRACKET)
          partial_sum = opening_bracket_counts(i);

        if (brackets(i) == CLOSING_BRACKET)
        {
          int match = matching_bracket(i);
          partial_sum = opening_bracket_counts(match) - 1;
        }
      });

#if 0
  auto brackets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, brackets);
  auto opening_brackets_count_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, opening_bracket_counts);
  auto closing_brackets_count_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, closing_bracket_counts);
  auto alpha_vertices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, alpha_vertices);

  printf("Brackets:\n");
  for (int i = 0; i < (int)euler_tour.size(); ++i)
  {
    if (brackets_host(i) == OPENING_BRACKET)
      printf(" [");
    else if (brackets_host(i) == CLOSING_BRACKET)
      printf(" ]");
    else
      printf(" .");
  }
  printf("\n#[:\n");
  for (int i = 0; i < (int)euler_tour.size(); ++i)
    printf("%2d", opening_brackets_count_host(i));
  // printf("\n#]:\n");
  // for (int i = 0; i < (int)euler_tour.size(); ++i)
  // printf("%2d", closing_brackets_count_host(i));
  printf("\nalpha vertices:\n");
  for (int i = 0; i < (int)euler_tour.size(); ++i)
    printf("%2d", alpha_vertices_host(i));
  printf("\n");
#endif

  return alpha_vertices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
assignAlphaVerticesNew(ExecutionSpace const &exec_space,
                       Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges,
                       Kokkos::View<int *, MemorySpace> alpha_edge_indices)
{
  auto n = sorted_edges.size() + 1;
  auto num_alpha_edges = (int)alpha_edge_indices.size();

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_vertices"),
      n);

  {
    // Do initial union-find on the subgraphs

    Kokkos::View<int *, MemorySpace> mark_alpha_edges(
        "ArborX::Dendrogram::alpha_vertices", n - 1);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::mark_alpha_edges",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
        KOKKOS_LAMBDA(int i) { mark_alpha_edges(alpha_edge_indices(i)) = 1; });

    iota(exec_space, alpha_vertices);

    UnionFind<MemorySpace> union_find(alpha_vertices);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::alpha_vertices_union_find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
        KOKKOS_LAMBDA(int e) {
          if (mark_alpha_edges(e) == 0)
          {
            // Not an alpha edge
            auto &edge = sorted_edges(e);
            union_find.merge(edge.source, edge.target);
          }
        });
    // finalize union-find
    Kokkos::parallel_for(
        "ArborX::Dendrogram::finalize_union-find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
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

#if 0
  printf("alpha vertices:\n");
  for (int i = 0; i < (int)n; ++i)
    printf(" %d", alpha_vertices(i));
  printf("\n");
#endif

  {
    // Map found alpha-vertices back to [0, #alpha vertices) range

    Kokkos::View<int *, MemorySpace> map(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::map_back"),
        n);
    Kokkos::deep_copy(exec_space, map, -1);

    Kokkos::parallel_for(
        "ArborX::Dendrogram::find_unique_entries",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          map(alpha_vertices(i)) = 1;
        });
#if 0
    printf("map:\n");
    for (int i = 0; i < (int)n; ++i)
      printf(" %d", map(i));
    printf("\n");
#endif
    int num_unique_entries = 0;
    Kokkos::parallel_scan(
        "ArborX::Dendrogram::map_scan",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
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
#if 0
    printf("map (scanned):\n");
    for (int i = 0; i < (int)n; ++i)
      printf(" %d", map(i));
    printf("\n");
#endif
    Kokkos::parallel_for(
        "ArborX::Dendrogram::remap_alpha_vertices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          // Assuming atomic store
          alpha_vertices(i) = map(alpha_vertices(i));
        });
  }

#if 0
  printf("alpha vertices (remapped):\n");
  for (int i = 0; i < (int)n; ++i)
    printf(" %d", alpha_vertices(i));
  printf("\n");
#endif

  return alpha_vertices;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<WeightedEdge *, MemorySpace>
buildAlphaEdges(ExecutionSpace const &exec_space,
                Kokkos::View<WeightedEdge *, MemorySpace> edges,
                Kokkos::View<int *, MemorySpace> euler_tour,
                Kokkos::View<int *, MemorySpace> alpha_edge_indices,
                Kokkos::View<int *, MemorySpace> alpha_vertices)
{

  Kokkos::View<WeightedEdge *, MemorySpace> alpha_mst_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_mst_edges"),
      alpha_edge_indices.size());
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_alpha_mst_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                          alpha_edge_indices.size()),
      KOKKOS_LAMBDA(int i) {
        int e = alpha_edge_indices(i);
        alpha_mst_edges(i) = {alpha_vertices(euler_tour(2 * e + 0)),
                              alpha_vertices(euler_tour(2 * e + 1)),
                              edges(i).weight};
      });
  return alpha_mst_edges;
}

#if 0
template <typename ExecutionSpace, typename MST, typename EdgeParents>
void computeFlatClustering(ExecutionSpace const &exec_space, int core_min_size,
                           MST sorted_mst_edges, EdgeParents edge_parents)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::compute_flat_clustering");

  using MemorySpace = typename MST::memory_space;

  int const n = sorted_mst_edges.extent_int(0) + 1;

  IncidenceMatrix<MemorySpace> incidence_matrix(exec_space, sorted_mst_edges);
  auto &incident_offsets = incidence_matrix._incident_offsets;
  auto &incident_edges = incidence_matrix._incident_edges;

  Kokkos::View<int *, MemorySpace> num_descendants(
      "ArborX::Dendrogram::num_descendants", n - 1);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_num_descendants",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        int parent = -1;

        // Parent of a vertex is the largest incident edge
        for (int j = incident_offsets(i); j < incident_offsets(i + 1); ++j)
          if (incident_edges(j) > parent)
            parent = incident_edges(j);

        int count = 1;

        do
        {
          // First thread up would encounter 0 at the parent,
          // while the second would see stored nonzero count
          // from the first thread
          int count_other_child = Kokkos::atomic_compare_exchange(
              &num_descendants(parent), 0, count);
          if (count_other_child == 0)
          {
            // First thread up
            break;
          }

          num_descendants(parent) += count;

          bool const is_valid_cluster =
              (num_descendants(parent) >= core_min_size);
          bool const is_true_cluster =
              is_valid_cluster &&
              (count >= core_min_size && count_other_child >= core_min_size);
          (void)is_valid_cluster;
          (void)is_true_cluster;

          count = num_descendants(parent);
          i = parent;
          parent = edge_parents(i);

        } while (i != 0);
      });

  Kokkos::Profiling::popRegion();
}
#endif

} // namespace ArborX::Details

#endif
