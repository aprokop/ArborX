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

#ifndef ARBORX_DBSCAN_HPP
#define ARBORX_DBSCAN_HPP

#include <ArborX_LinearBVH.hpp>

#include <cmath> // ceil, log
#include <random>

namespace ArborX
{

template <typename View>
struct PrimitivesWithRadius
{
  View _view;
  float _r;
};

template <typename Primitives>
auto buildPredicates(Primitives const &primitives, float r)
{
  return PrimitivesWithRadius<Primitives>{primitives, r};
}

template <typename Primitives>
struct AccessTraits<PrimitivesWithRadius<Primitives>, PredicatesTag>
{
  using memory_space = typename Primitives::memory_space;
  using Predicates = PrimitivesWithRadius<Primitives>;
  static size_t size(Predicates const &w) { return w._view.extent(0); }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(Sphere{w._view(i), w._r}), (int)i);
  }
};

template <typename ExecutionSpace, typename Primitives, typename Degrees>
Kokkos::View<decltype(attach(intersects(std::declval<ArborX::Sphere>()),
                             std::declval<Kokkos::Array<int, 3>>())) *,
             typename Primitives::memory_space>
buildSampledPredicates(ExecutionSpace const &exec_space,
                       Primitives const &primitives, float r,
                       Degrees const &degrees, int num_samples)
{
  using MemorySpace = typename Primitives::memory_space;

  auto prefix_sum = clone(degrees);
  exclusivePrefixSum(exec_space, prefix_sum);

  int const num_vertices = prefix_sum.size() - 1;
  auto const num_edges = lastElement(prefix_sum);

  using QueryType = decltype(attach(intersects(std::declval<ArborX::Sphere>()),
                                    std::declval<Kokkos::Array<int, 3>>()));
  Kokkos::View<QueryType *, MemorySpace> queries(
      Kokkos::ViewAllocateWithoutInitializing("sampled_queries"), num_samples);

  Kokkos::View<int *, MemorySpace> samples(
      Kokkos::ViewAllocateWithoutInitializing("samples"), num_samples);
  auto samples_host = Kokkos::create_mirror_view(samples);

  // Knuth algorithm for unique samples
  auto const N = num_edges;
  auto const M = num_samples;
  for (int in = 0, im = 0; in < N && im < M; ++in)
  {
    int rn = N - in;
    int rm = M - im;
    if (rand() % rn < rm)
      samples_host(im++) = in;
  }
  Kokkos::deep_copy(samples, samples_host);

  Kokkos::parallel_for(
      "ArborX::npcf::build_random_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, num_samples),
      KOKKOS_LAMBDA(int const i) {
        int edge = samples(i);

        // upper_bound
        int first = 0;
        int last = num_vertices;
        while (first < last)
        {
          int split = first + ((last - first) >> 1);

          if (prefix_sum(split) <= edge)
            first = split + 1;
          else
            last = split;
        }

        int vertex = first - 1;
        int vertex_edge = edge - prefix_sum(vertex);

        Kokkos::Array<int, 3> data;
        data[0] = vertex;
        data[1] = vertex_edge;
        data[2] = i;
        queries(i) = attach(intersects(Sphere{primitives(vertex), r}), data);
      });

  return queries;
}

namespace NPCF
{
template <typename MemorySpace>
struct NumNeighExitCallback
{
  Kokkos::View<int *, MemorySpace> _degrees;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    auto i = getData(query);
    if (i < j)
      Kokkos::atomic_fetch_add(&_degrees(i), 1);
  }
};

template <typename MemorySpace>
struct ExactCountCallback
{
  Kokkos::View<ArborX::Point *, MemorySpace> _points;
  Kokkos::View<int, MemorySpace> _count;
  float a;
  float b;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    auto i = getData(query);
    if (i < j)
    {
      auto dist = Details::distance(_points(i), _points(j));
      if (dist >= a && dist <= b)
        Kokkos::atomic_fetch_add(&_count(), 1);
    }
  }
};

template <typename MemorySpace>
struct RandomizedCountCallback
{
  Kokkos::View<ArborX::Point *, MemorySpace> _points;
  Kokkos::View<int *, MemorySpace> _edge_counts;
  Kokkos::View<int, MemorySpace> _count;
  float a;
  float b;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int j) const
  {
    auto const &data = getData(query);
    int i = data[0];
    int edge = data[1];
    int sample_id = data[2];

    if (i < j &&
        (Kokkos::atomic_fetch_add(&_edge_counts(sample_id), 1) == edge))
    {
      auto dist = Details::distance(_points(i), _points(j));
      if (dist >= a && dist <= b)
        Kokkos::atomic_fetch_add(&_count(), 1);

      return ArborX::CallbackTreeTraversalControl::early_exit;
    }
    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

} // namespace NPCF

template <typename ExecutionSpace, typename Points>
int countExact(ExecutionSpace const &exec_space, Points const &points, float a,
               float b)
{
  using MemorySpace = typename Points::memory_space;

  ArborX::BVH<MemorySpace> bvh(exec_space, points);

  auto const predicates = buildPredicates(points, b);

  Kokkos::View<int, MemorySpace> count("ArborX::npcf::count");
  bvh.query(exec_space, predicates,
            NPCF::ExactCountCallback<MemorySpace>{points, count, a, b});

  auto count_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);

  return count_host();
}

template <typename ExecutionSpace, typename Points>
int countRandomized(ExecutionSpace const &exec_space, Points const &points,
                    float a, float b, float eps, float delta)
{
  using MemorySpace = typename Points::memory_space;

  ArborX::BVH<MemorySpace> bvh(exec_space, points);

  auto const predicates = buildPredicates(points, b);

  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;
  auto timer_start = [&exec_space](Kokkos::Timer &timer) {
    exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space](Kokkos::Timer const &timer) {
    exec_space.fence();
    return timer.seconds();
  };

  // Step 0: determine vertex degrees
  timer_start(timer);
  auto const num_points = points.size();
  // +1 for easier prefix sum
  Kokkos::View<int *, MemorySpace> degrees("ArborX::npcf::vertex_degrees",
                                           num_points + 1);
  bvh.query(exec_space, predicates,
            NPCF::NumNeighExitCallback<MemorySpace>{degrees});
  elapsed["degree"] = timer_seconds(timer);

  // NOTE: each edge is counted twice
  auto num_total_edges = accumulate(exec_space, degrees, 0);
  std::cout << "num_total_edges = " << num_total_edges << std::endl;

  // Step 1: randomly sample edges
  timer_start(timer);
  int const num_samples = std::ceil(0.5 / (eps * eps) * std::log(2 / delta));
  std::cout << "num_samples = " << num_samples << std::endl;
  auto sampled_predicates =
      buildSampledPredicates(exec_space, points, b, degrees, num_samples);
  elapsed["samples"] = timer_seconds(timer);

  timer_start(timer);
  Kokkos::View<int, MemorySpace> count("ArborX::npcf::count");
  Kokkos::View<int *, MemorySpace> edge_counts("ArborX::npcf::edge_counts",
                                               num_samples);
  bvh.query(exec_space, sampled_predicates,
            NPCF::RandomizedCountCallback<MemorySpace>{points, edge_counts,
                                                       count, a, b});
  elapsed["query"] = timer_seconds(timer);
  auto count_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);

  printf("--degree    : %10.3f\n", elapsed["degree"]);
  printf("--samples   : %10.3f\n", elapsed["samples"]);
  printf("--query     : %10.3f\n", elapsed["query"]);

  auto fraction = (float)count_host() / num_samples;
  printf("positive matches: %.2f%%\n", fraction * 100);
  return fraction * num_total_edges;
}

} // namespace ArborX

#endif
