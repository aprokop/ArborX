/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborXBenchmark_PointClouds.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>

#include <benchmark/benchmark.h>

using Point = ArborX::Point<3>;

template <typename MemorySpace, typename ExecutionSpace>
auto constructPoints(ExecutionSpace const &exec, int n,
                     ArborXBenchmark::PointCloudType point_cloud_type)
{
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Benchmark::random_points"),
      n);
  auto const a = std::cbrt(n);
  ArborXBenchmark::generatePointCloud(exec, point_cloud_type, a, random_points);

  return random_points;
}

struct ExtractCallback
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &pair,
                                  Output const &out) const
  {
    out(pair.index);
  }
};

void BM_benchmark_attach(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto point_cloud_type = ArborXBenchmark::PointCloudType::filled_box;
  auto const points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);

  auto query_points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);
  // Number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors
  constexpr int num_neighbors = 10;
  float const r = std::cbrt(static_cast<double>(num_neighbors) * 6.f /
                            Kokkos::numbers::pi_v<float>);
  auto queries = ArborX::Experimental::make_intersects(query_points, r);

  auto index = ArborX::BoundingVolumeHierarchy(
      exec_space, ArborX::Experimental::attach_indices(points));

  exec_space.fence();
  for (auto _ : state)
  {
    Kokkos::View<int *, MemorySpace> offset("offset", 0);
    Kokkos::View<int *, MemorySpace> indices("indices", 0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(exec_space, queries, ExtractCallback{}, indices, offset);

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] =
      benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename Counts>
struct Callback
{
  Counts _counts;

  template <typename Query, typename Value>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &) const
  {
    auto index = ArborX::getData(query);
    ++_counts(index);
  }
};

void BM_benchmark_attach_callback(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto point_cloud_type = ArborXBenchmark::PointCloudType::filled_box;
  auto const points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);

  auto query_points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);
  auto num_queries = query_points.size();
  // Number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors
  constexpr int num_neighbors = 10;
  float const r = std::cbrt(static_cast<double>(num_neighbors) * 6.f /
                            Kokkos::numbers::pi_v<float>);
  auto queries = ArborX::Experimental::attach_indices(
      ArborX::Experimental::make_intersects(query_points, r));

  auto index = ArborX::BoundingVolumeHierarchy(
      exec_space, ArborX::Experimental::attach_indices(points));

  Kokkos::View<int *, MemorySpace> counts("counts", num_queries);

  exec_space.fence();
  for (auto _ : state)
  {
    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(exec_space, queries, Callback<decltype(counts)>{counts});

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] =
      benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename MemorySpace>
struct Iota
{
  using memory_space = MemorySpace;

  int _n;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Iota<MemorySpace>>
{
  using Self = Iota<MemorySpace>;

  using memory_space = typename Self::memory_space;
  static KOKKOS_FUNCTION size_t size(Self const &self) { return self._n; }
  static KOKKOS_FUNCTION int get(Self const &, size_t i) { return (int)i; }
};

void BM_benchmark_iota(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto point_cloud_type = ArborXBenchmark::PointCloudType::filled_box;
  auto const points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);

  auto query_points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);
  // Number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors
  constexpr int num_neighbors = 10;
  float const r = std::cbrt(static_cast<double>(num_neighbors) * 6.f /
                            Kokkos::numbers::pi_v<float>);
  auto queries = ArborX::Experimental::make_intersects(query_points, r);

  auto index = ArborX::BoundingVolumeHierarchy(
      exec_space, Iota<MemorySpace>{(int)n}, points);

  exec_space.fence();
  for (auto _ : state)
  {
    Kokkos::View<int *, MemorySpace> offset("offset", 0);
    Kokkos::View<int *, MemorySpace> indices("indices", 0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(exec_space, queries, indices, offset);

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] =
      benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate);
}

void BM_benchmark_iota_callback(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto point_cloud_type = ArborXBenchmark::PointCloudType::filled_box;
  auto const points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);

  auto query_points =
      constructPoints<MemorySpace>(exec_space, n, point_cloud_type);
  auto num_queries = query_points.size();
  // Number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors
  constexpr int num_neighbors = 10;
  float const r = std::cbrt(static_cast<double>(num_neighbors) * 6.f /
                            Kokkos::numbers::pi_v<float>);
  auto queries = ArborX::Experimental::attach_indices(
      ArborX::Experimental::make_intersects(query_points, r));

  auto index = ArborX::BoundingVolumeHierarchy(
      exec_space, Iota<MemorySpace>{(int)n}, points);

  Kokkos::View<int *, MemorySpace> counts("counts", num_queries);

  exec_space.fence();
  for (auto _ : state)
  {
    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(exec_space, queries, Callback<decltype(counts)>{counts});

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] =
      benchmark::Counter(n, benchmark::Counter::kIsIterationInvariantRate);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  constexpr int min_size = 100;
  constexpr int max_size = 10000;
  constexpr int multiplier = 100;

  BENCHMARK(BM_benchmark_attach)
      ->RangeMultiplier(multiplier)
      ->Range(min_size, max_size)
      ->Unit(benchmark::kMillisecond)
      ->UseManualTime();

  BENCHMARK(BM_benchmark_attach_callback)
      ->RangeMultiplier(multiplier)
      ->Range(min_size, max_size)
      ->Unit(benchmark::kMillisecond)
      ->UseManualTime();

  BENCHMARK(BM_benchmark_iota)
      ->RangeMultiplier(multiplier)
      ->Range(min_size, max_size)
      ->Unit(benchmark::kMillisecond)
      ->UseManualTime();

  BENCHMARK(BM_benchmark_iota_callback)
      ->RangeMultiplier(multiplier)
      ->Range(min_size, max_size)
      ->Unit(benchmark::kMillisecond)
      ->UseManualTime();

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
