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

#ifndef ARBORX_DETAILS_DENDROGRAM_UNION_FIND_HPP
#define ARBORX_DETAILS_DENDROGRAM_UNION_FIND_HPP

#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

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

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
dendrogramUnionFind(ExecutionSpace const &exec_space,
                    Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");

  int const num_edges = sorted_edges.extent_int(0);
  int const num_vertices = num_edges + 1;

  Kokkos::View<int *, MemorySpace> edge_parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::edge_parents"),
      num_edges);

  constexpr int UNDEFINED = -1;
  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::labels"),
      num_vertices);
  Kokkos::deep_copy(exec_space, labels, UNDEFINED);

  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::copy_to_host");

  auto sorted_edges_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sorted_edges);
  auto edge_parents_host = Kokkos::create_mirror_view(edge_parents);
  auto labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram::union_find");

#if 1
  std::cout << "Running Wang's union-find" << std::endl;
  WangUnionFind union_find(num_vertices);
  std::ignore = exec_space;
#else
  std::cout << "Running ArborX's union-find" << std::endl;
  Kokkos::View<int *, MemorySpace> vertex_labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::vertex_labels"),
      num_vertices);
  iota(exec_space, vertex_labels);
  auto vertex_labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vertex_labels);
  Details::UnionFind<Kokkos::HostSpace> union_find(vertex_labels_host);
#endif
  for (int e = 0; e < num_edges; ++e)
  {
    int i = sorted_edges_host(e).source;
    int j = sorted_edges_host(e).target;

    for (int k : {i, j})
    {
      auto edge_child = labels_host(union_find.representative(k));
      if (edge_child != UNDEFINED)
        edge_parents_host(edge_child) = e;
    }

    union_find.merge(i, j);

    labels_host(union_find.representative(i)) = e;
  }
  edge_parents_host(num_edges - 1) = UNDEFINED; // root

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram::copy_to_device");

  Kokkos::deep_copy(exec_space, edge_parents, edge_parents_host);

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return edge_parents;
}

template <typename ExecutionSpace, typename MemorySpace>
bool verifyDendrogram(ExecutionSpace const &exec_space,
                      Kokkos::View<WeightedEdge *, MemorySpace> edges,
                      Kokkos::View<int *, MemorySpace> parents)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::verification");

  auto const num_edges = edges.size();

  Kokkos::View<float *, MemorySpace> weights(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::verification::weights"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::verification::copy_weights",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const edge_index) {
        weights(edge_index) = edges(edge_index).weight;
      });

  auto permute = Details::sortObjects(exec_space, weights);

  Kokkos::View<WeightedEdge *, MemorySpace> sorted_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::verification::sorted_edges"),
      num_edges);
  Kokkos::deep_copy(exec_space, sorted_edges, edges);
  Details::applyPermutation(exec_space, permute, sorted_edges);

  auto correct_parents = dendrogramUnionFind(exec_space, sorted_edges);
  {
    auto correct_parents_copy = KokkosExt::clone(exec_space, correct_parents);
    Kokkos::parallel_for(
        "ArborX::Dendrogram::verification::permute_results_back",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
        KOKKOS_LAMBDA(int i) {
          correct_parents(permute(i)) = correct_parents_copy(i);
        });
  }

  int num_different = 0;
  Kokkos::parallel_reduce(
      "ArborX::Dendrogram::verify",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int i, int &update) {
        if (parents(i) != correct_parents(i))
        {
          printf("[%d] %d vs %d\n", i, parents(i), correct_parents(i));
          ++update;
        }
      },
      num_different);

  Kokkos::Profiling::popRegion();

  return (num_different == 0);
}

} // namespace ArborX::Details

#endif
