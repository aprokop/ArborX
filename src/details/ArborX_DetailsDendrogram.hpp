/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

#include <ArborX_DetailsEulerTour.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
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
    int const n = edges.extent(0) + 1;

    Kokkos::realloc(_incident_offsets, n + 1);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::compute_incident_counts",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
        KOKKOS_LAMBDA(int const edge_index) {
          auto const &edge = edges(edge_index);
          Kokkos::atomic_fetch_add(&_incident_offsets(edge.source), 1);
          Kokkos::atomic_fetch_add(&_incident_offsets(edge.target), 1);
        });
    exclusivePrefixSum(exec_space, _incident_offsets);

    ARBORX_ASSERT(KokkosExt::lastElement(exec_space, _incident_offsets) ==
                  2 * (n - 1));

    KokkosExt::reallocWithoutInitializing(
        exec_space, _incident_edges,
        KokkosExt::lastElement(exec_space, _incident_offsets));

    auto offsets = KokkosExt::clone(exec_space, _incident_offsets);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::compute_incident_counts",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
        KOKKOS_LAMBDA(int const edge_index) {
          auto const &edge = edges(edge_index);
          _incident_edges(Kokkos::atomic_fetch_add(&offsets(edge.source), 1)) =
              edge_index;
          _incident_edges(Kokkos::atomic_fetch_add(&offsets(edge.target), 1)) =
              edge_index;
        });
  }

  template <typename ExecutionSpace>
  void degrees(ExecutionSpace const &exec_space)
  {
    int const n = _edges.extent(0) + 1;

    int max_degree = 0;
    Kokkos::parallel_reduce(
        "ArborX::HDBSCA::max_offset",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          int degree = _incident_offsets(i + 1) - _incident_offsets(i);
          if (degree > update)
            update = degree;
        },
        Kokkos::Max<int>(max_degree));

    Kokkos::View<int *, MemorySpace> degrees_hist("ArborX::Dendrogram::degrees",
                                                  max_degree);
    Kokkos::parallel_for(
        "blah", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          int degree = _incident_offsets(i + 1) - _incident_offsets(i);
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
Edges sortEdges(ExecutionSpace const &exec_space, Edges edges)
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

  return edges;
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

template <typename ExecutionSpace, typename EulerTour, typename MST>
MST buildAlphaMST(
    ExecutionSpace const &exec_space, EulerTour euler_tour,
    Kokkos::View<int *, typename MST::memory_space> alpha_edge_indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::build_alpha_mst");

  int const num_alpha_edges = alpha_edge_indices.extent(0);

  EulerTour alpha_euler_tour(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::alpha_euler_tour"),
      2 * num_alpha_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::pick_alpha_euler_tour",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int k) {
        int i = alpha_edge_indices(k);
        alpha_euler_tour(2 * k + 0) = euler_tour(2 * i + 0);
        alpha_euler_tour(2 * k + 1) = euler_tour(2 * i + 1);
      });

  auto sorted_alpha_euler_tour = KokkosExt::clone(exec_space, alpha_euler_tour);
  auto permute = sortObjects(exec_space, sorted_alpha_euler_tour);
  auto inv_permute =
      KokkosExt::cloneWithoutInitializingNorCopying(exec_space, permute);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_inv_permute",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, permute.extent(0)),
      KOKKOS_LAMBDA(int i) { inv_permute(permute(i)) = i; });

#ifdef VERBOSE
  printSortedEuler(permute, inv_permute, "alpha");
#endif

  MST alpha_mst_edges(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         "ArborX::Dendrogram::alpha_mst_edges"),
                      num_alpha_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::compute_alpha_mst",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int i) {
        auto is_start_edge = [&inv_permute, &permute](int k) {
          using KokkosExt::min;
          int e = permute(k) / 2;
          int m = min(inv_permute(2 * e), inv_permute(2 * e + 1));
          return k == m;
        };

        // Go backwards to find a start edge this is nested in
        int p = inv_permute(2 * i) - 1;
        // printf("[%d] p = %d", i, p);
        while (p >= 0 && !is_start_edge(p))
        {
          p = inv_permute(permute(p) - 1) - 1;
          // printf("-> %d", p);
        }
        // printf("\n");

        alpha_mst_edges(i).source =
            (p >= 0 ? (permute(p) / 2) : num_alpha_edges);
        alpha_mst_edges(i).target = i;
      });
#ifdef VERBOSE
  printf("alpha MST graph\n");
  for (int i = 0; i < num_alpha_edges; ++i)
    printf("[%d]: (%d, %d)\n", i, alpha_mst_edges(i).source,
           alpha_mst_edges(i).target);
#endif

  Kokkos::Profiling::popRegion();

  return alpha_mst_edges;
}

enum class Bracket
{
  NO_BRACKET = 0,
  OPENING_BRACKET = 1,
  CLOSING_BRACKET = 2
};

template <typename ExecutionSpace, typename MemorySpace>
auto assignAlphaVertices(ExecutionSpace const &exec_space,
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
  Kokkos::deep_copy(exec_space, brackets, Bracket::NO_BRACKET);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::build_bracket_array",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_alpha_edges),
      KOKKOS_LAMBDA(int k) {
        int alpha_edge = alpha_edge_indices(k);
        euler_flat(euler_tour(2 * alpha_edge + 0)) = Bracket::OPENING_BRACKET;
        euler_flat(euler_tour(2 * alpha_edge + 1)) = Bracket::CLOSING_BRACKET;
      });

  Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> bracket_counts(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::bracket_counts"),
      euler_tour.size());
  Kokkos::parallel_scan(
      "ArborX::Dendrogram::count_alpha_[]",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * num_edges),
      KOKKOS_LAMBDA(int i, Kokkos::pair<int, int> &partial_sum, bool is_final) {
        if (brackets(i) == Bracket::OPENING_BRACKET)
          ++partial_sum.first;
        else if (brackets(i) == Bracket::CLOSING_BRACKET)
          ++partial_sum.second;

        if (is_final)
          bracket_counts[i] = partial_sum;
      });

  Kokkos::View<int *, MemorySpace> alpha_vertices(
      "ArborX::Dendrogram::alpha_vertices", euler_tour.size());
  Kokkos::parallel_for(
      "ArborX::Dendrogram::assign_alpha_vertices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * num_edges),
      KOKKOS_LAMBDA(int i, int &partial_sum, bool is_final) {
        if (brackets(i) == Bracket::OPENING_BRACKET)
          partial_sum = bracket_counts[i].first;

        if (is_final)
          alpha_vertices(i) = partial_sum;

        if (brackets(i) == Bracket::CLOSING_BRACKET)
          partial_sum -= (bracket_counts[i].first - bracket_counts[i].second);
      });
}

template <typename ExecutionSpace, typename MST>
void dendrogramAlphaTree(ExecutionSpace const &exec_space, MST sorted_mst_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_alpha");

  using MemorySpace = typename MST::memory_space;

  auto const num_edges = sorted_mst_edges.extent_int(0);

  // Step 1: compute Euler tour for the original MST
  Kokkos::Profiling::ProfilingSection profile_euler_tour(
      "ArborX::Dendrogram::euler_tour");
  profile_euler_tour.start();
  // The returned Euler tour is of size twice the number of edges. Each pair
  // of entries {2*i, 2*i+1} correspond to the edge i, in two directions (one
  // going down, one up).
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::euler_tour");
  auto euler_tour = eulerTour(exec_space, sorted_mst_edges);
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
  IncidenceMatrix<MemorySpace> incidence_matrix(exec_space, sorted_mst_edges);
  Kokkos::Profiling::popRegion();
  profile_build_incidence_matrix.stop();

  // Step 3: find alpha edges of the original MST
  Kokkos::Profiling::ProfilingSection profile_compute_alpha_edges(
      "ArborX::Dendrogram::compute_alpha_edges");
  profile_compute_alpha_edges.start();
  auto alpha_edge_indices = findAlphaEdges(exec_space, incidence_matrix);
  profile_compute_alpha_edges.start();

  auto num_alpha_edges = alpha_edge_indices.extent_int(0);
  printf("#alpha edges: %d [%.2f%%]\n", num_alpha_edges,
         (100.f * num_alpha_edges) / num_edges);
#ifdef VERBOSE
  printf("alpha edges:\n");
  for (int i = 0; i < num_alpha_edges; ++i)
  {
    int e = alpha_edge_indices(i);
    printf("[%i] : %d (%d, %d)\n", i, e, sorted_mst_edges(e).source,
           sorted_mst_edges(e).target);
  }
  printf("\n");
#endif

  // Step 4: assign alpha-vertices
  Kokkos::Profiling::ProfilingSection profile_alpha_vertices(
      "ArborX::Dendrogram::alpha_vertices");
  profile_alpha_vertices.start();
  auto alpha_vertices =
      assignAlphaVertices(exec_space, euler_tour, alpha_edge_indices);
  profile_alpha_vertices.stop();

#if 0
  // Step 5: construct alpha-MST
  Kokkos::Profiling::ProfilingSection profile_alpha_mst(
      "ArborX::Dendrogram::alpha_mst");
  profile_alpha_mst.start();
  auto alpha_mst_edges =
      buildAlphaMST(exec_space, alpha_edge_indices, alpha_vertices);
  profile_alpha_mst.stop();

  // Step 6: build dendrogram of the alpha-tree
  Kokkos::Profiling::ProfilingSection profile_dendrogram_alpha(
      "ArborX::Dendrogram::dendrogram_alpha");
  profile_dendrogram_alpha.start();
  auto alpha_parents_of_alpha =
      dendrogramUnionFind(exec_space, alpha_mst_edges);
  profile_dendrogram_alpha.stop();
#endif

  // auto alpha_sided_parents =
  // findAlphaParents(exec_space, sorted_mst_edges, alpha_parents_of_alpha);

  // auto permute = sortObjects(alpha_sided_parents);

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Details

#endif
