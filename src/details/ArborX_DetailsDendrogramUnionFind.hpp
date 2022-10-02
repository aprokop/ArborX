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
                    Kokkos::View<WeightedEdge *, MemorySpace> sorted_mst_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");

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

} // namespace ArborX::Details

#endif
