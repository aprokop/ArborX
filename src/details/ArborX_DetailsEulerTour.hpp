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
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Random.hpp>

namespace ArborX::Details
{

struct Pair
{
  int first;
  int second;

private:
  friend KOKKOS_FUNCTION bool operator<(Pair const &a, Pair const &b)
  {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
  }
};

// Assumption: edges are bidirectional and unique
template <typename ExecutionSpace, typename Edges>
auto eulerTourList(ExecutionSpace const &exec_space, Edges const &edges)
{
  Kokkos::Profiling::pushRegion("ArborX::euler_tour_list");

  using MemorySpace = typename Edges::memory_space;

  int const n = 2 * edges.size();

  ARBORX_ASSERT(n > 0);

  // Construct a list of directed edges (half-edges)
  Kokkos::View<Pair *, MemorySpace> directed_edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::euler_tour_list::directed_edges"),
      n);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::compute_directed_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
      KOKKOS_LAMBDA(int const i) {
        directed_edges(2 * i + 0) = Pair{edges(i).source, edges(i).target};
        directed_edges(2 * i + 1) = Pair{edges(i).target, edges(i).source};
      });

  auto directed_edges_clone = KokkosExt::clone(exec_space, directed_edges);

  auto permute = sortObjects(exec_space, directed_edges_clone);
  auto const &sorted_directed_edges = directed_edges_clone;

  auto rev_permute =
      KokkosExt::cloneWithoutInitializingNorCopying(exec_space, permute);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::compute_rev_permute",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) { rev_permute(permute(i)) = i; });

  Kokkos::View<int *, MemorySpace> successors(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::euler_tour_list::successors"),
      n);
  Kokkos::parallel_for(
      "ArborX::euler_tour_list::build_successors",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        int const twin = (i % 2 == 0 ? i + 1 : i - 1);
        int next;
        int j = rev_permute(twin);
        if (j < n - 1 && (sorted_directed_edges(j + 1).first ==
                          sorted_directed_edges(j).first))
        {
          next = j + 1;
        }
        else
        {
          while (j > 0 && (sorted_directed_edges(j - 1).first ==
                           sorted_directed_edges(j).first))
            --j;
          next = j;
        }
        successors(i) = permute(next);
      });

  Kokkos::Profiling::popRegion();

  return successors;
}

// Implementation of list ranking using GPU-optimized Wei-Jaja algorithm
template <typename ExecutionSpace, typename List>
Kokkos::View<int *, typename List::memory_space>
rankList(ExecutionSpace const &exec_space, List &list, int head)
{
  Kokkos::Profiling::pushRegion("ArborX::list_ranking");

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
  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(1984);
  Kokkos::View<int *, MemorySpace> sublists_heads(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::list_ranking::sublists_heads"),
      num_sublists);
  Kokkos::parallel_for(
      "ArborX::HDBSCAN::init_sublists",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_splitters),
      KOKKOS_LAMBDA(int const i) {
        using KokkosExt::min;

        int start = i * segment_size;
        int end = min(start + segment_size, n);

        // Choose the splitter randomly in the [start, end) interval as long as
        // it's not the last element in the list.
        auto rand_gen = rand_pool.get_state();
        int splitter;
        do
        {
          splitter = start + rand_gen.urand() % (end - start);
        } while (list(splitter) == end_of_list);
        rand_pool.free_state(rand_gen);

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

  Kokkos::Profiling::popRegion();

  return ranks;
}

// Assumption: edges are bidirectional and unique
template <typename ExecutionSpace, typename... EdgesProperties>
auto eulerTour(ExecutionSpace const &exec_space,
               Kokkos::View<WeightedEdge *, EdgesProperties...> const &edges)
{
  Kokkos::Profiling::pushRegion("ArborX::euler_tour");

  auto successors = eulerTourList(exec_space, edges);

  int const head = 0; // does not matter
  auto euler_tour = rankList(exec_space, successors, head);

  Kokkos::Profiling::popRegion();

  return euler_tour;
}

} // namespace ArborX::Details

#endif
