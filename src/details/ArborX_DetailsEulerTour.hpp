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

#ifndef ARBORX_DETAILS_EULER_TOUR_HPP
#define ARBORX_DETAILS_EULER_TOUR_HPP

#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

namespace ArborX::Details
{

// Assumption: edges are bidirectional and unique and form a tree
template <typename ExecutionSpace, typename Edges>
Kokkos::View<int *, typename Edges::memory_space>
eulerTourList(ExecutionSpace const &exec_space, Edges const &edges)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::euler_tour_list");

  using MemorySpace = typename Edges::memory_space;

  int const num_edges = edges.size();
  int const num_vertices = num_edges + 1;

  ARBORX_ASSERT(num_edges > 0);

  Kokkos::View<int *, MemorySpace> offsets("ArborX::euler_tour_list::offsets",
                                           num_vertices + 1);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::compute_counts",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const edge_index) {
        auto const &edge = edges(edge_index);
        Kokkos::atomic_increment(&offsets(edge.source));
        Kokkos::atomic_increment(&offsets(edge.target));
      });
  exclusivePrefixSum(exec_space, offsets);

  ARBORX_ASSERT(KokkosExt::lastElement(exec_space, offsets) == 2 * num_edges);

  Kokkos::View<unsigned *, MemorySpace> permute(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::euler_tour_list::permute"),
      2 * num_edges);

  auto offsets_clone = KokkosExt::clone(exec_space, offsets);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::compute_intermediate_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const edge_index) {
        auto const &edge = edges(edge_index);
        permute(Kokkos::atomic_fetch_add(&offsets_clone(edge.source), 1)) =
            2 * edge_index + 0;
        permute(Kokkos::atomic_fetch_add(&offsets_clone(edge.target), 1)) =
            2 * edge_index + 1;
      });
  Kokkos::resize(offsets_clone, 0); // deallocate memory

  Kokkos::View<int *, MemorySpace> successors(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::euler_tour_list::successors"),
      2 * num_edges);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::build_successors",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_vertices),
      KOKKOS_LAMBDA(int const i) {
        int start = offsets(i);
        int len = offsets(i + 1) - start;

        auto value = [i, start, &edges, &permute](int k) {
          auto const &edge = edges(permute(start + k) / 2);
          return (i == edge.source ? edge.target : edge.source);
        };

        // FIXME This sorting is unnecessary. It will only affect the order of
        // subtrees to traverse, but will still produce a correct Euler Tour.
        // However, removing the sort will make the algorithm
        // non-deterministic, so the testing would need to change.
        //
        // Sort edges originating from each vertex and leading to a vertex with
        // larger index
        //
        // Using insertion sort, assuming that the degree of each node is small.
        // This works well for the trees coming from the Euclidean
        // minimum-spanning tree calculations. Probably does not work well for
        // power law kind of tree structures.
        for (int k = 1, j; k < len; ++k)
        {
          auto p = permute(start + k);
          auto t = value(k);

          for (j = k; j > 0 && value(j - 1) > t; --j)
            permute(start + j) = permute(start + j - 1);
          permute(start + j) = p;
        }

        // Build successors following the formula
        //      succ(twin(e)) = next(e)
        for (int k = 0; k < len; ++k)
        {
          int e = permute(start + k);
          int const twin = (e % 2 == 0 ? e + 1 : e - 1);
          int const next = start + (k + 1) % len;
          successors(twin) = permute(next);
        }
      });

  return successors;
}

// Implementation of list ranking using GPU-optimized Wei-Jaja algorithm
template <typename ExecutionSpace, typename List>
Kokkos::View<int *, typename List::memory_space>
rankList(ExecutionSpace const &exec_space, List &list, int head)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::list_ranking");

  using MemorySpace = typename List::memory_space;

  int const n = list.extent_int(0);

  ARBORX_ASSERT(n > 1);
  ARBORX_ASSERT(head < n);

  // FIXME: choosing appropriate number of splitters would require some
  // experimentation. Currently, the selection is taken from "Euler meets Cuda"
  // paper, which was optimized for GTX 980.
  int segment_size = sqrt(n) / 1.6;
  if (n < 100000)
    segment_size = 100;

  int num_splitters = n / segment_size;
  if (num_splitters == 0)
    num_splitters = 1;

  int num_sublists = num_splitters + 1;

  constexpr int end_of_list = -1;

  // Cut the list by resetting the link connecting to head
  Kokkos::parallel_for(
      "ArborX::list_ranking::cut_list",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        if (list(i) == head)
          list(i) = end_of_list;
      });

  // Initialize splitters
  Kokkos::View<int *, MemorySpace> sublists_heads(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::list_ranking::sublists_heads"),
      num_sublists);
  Kokkos::parallel_for(
      "ArborX::list_ranking::init_sublists",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_splitters),
      KOKKOS_LAMBDA(int const i) {
        using KokkosExt::min;

        int start = i * segment_size;
        int end = min(start + segment_size, n);

        // Lehmer (or Park-Miller) RNG
        unsigned int state = i + 1; // any positive number less than modulus
        auto lehmer = [&state]() {
          state = ((unsigned long long)state * 48271) % 0x7fffffff;
          return state;
        };

        // Choose the splitter randomly in the [start, end) interval as long as
        // it's not the last element in the list.
        int splitter;
        do
        {
          splitter = start + lehmer() % (end - start);
        } while (list(splitter) == end_of_list);

        sublists_heads(i + 1) = list(splitter);
        if (i == 0)
          sublists_heads(0) = head;

        list(splitter) = -i - 2; // -2 to avoid confusion with -1
      });

  Kokkos::View<int *, MemorySpace> ranks("ArborX::list_ranking::ranks", n);
  Kokkos::View<int *, MemorySpace> sublist_ids(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::list_ranking::sublists_ids"),
      n);
  Kokkos::View<int *, MemorySpace> sublists_total(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::list_ranking::sublists_total"),
      num_sublists);
  Kokkos::View<int *, MemorySpace> sublists_last(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::list_ranking::sublists_last"),
      num_sublists);
  Kokkos::parallel_for(
      "ArborX::list_ranking::scan_sublists",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_sublists),
      KOKKOS_LAMBDA(int const sublist) {
        int i = sublists_heads(sublist);
        int count = 0;
        do
        {
          ranks(i) = count++;
          sublist_ids(i) = sublist;

          int next = list(i);
          if (next < 0)
          {
            sublists_total(sublist) = count;
            sublists_last(sublist) = i;
          }

          i = next;
        } while (i >= 0);
      });

  Kokkos::parallel_for(
      "ArborX::list_ranking::scan_sublist_heads",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 1),
      KOKKOS_LAMBDA(int) {
        int total = 0;
        int current = head;
        int sublist = 0;
        for (int i = 0; i < num_sublists; ++i)
        {
          total += sublists_total(sublist);
          sublists_total(sublist) = total - sublists_total(sublist);

          current = sublists_last(sublist);
          sublist = -list(current) - 1;
        }
      });

  Kokkos::parallel_for(
      "ArborX::list_ranking::final_update",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) { ranks(i) += sublists_total(sublist_ids(i)); });

  return ranks;
}

// Assumption: edges are bidirectional and unique
template <typename ExecutionSpace, typename... EdgesProperties>
auto eulerTour(ExecutionSpace const &exec_space,
               Kokkos::View<WeightedEdge *, EdgesProperties...> const &edges,
               int head = 0)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::euler_tour");

  auto successors = eulerTourList(exec_space, edges);
  auto euler_tour = rankList(exec_space, successors, head);

  return euler_tour;
}

} // namespace ArborX::Details

#endif
