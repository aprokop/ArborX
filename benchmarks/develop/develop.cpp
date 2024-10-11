/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
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

template <typename MemorySpace, typename ExecutionSpace>
auto constructPoints(ExecutionSpace const &exec, int n,
                     ArborXBenchmark::PointCloudType point_cloud_type)
{
  using Point = ArborX::Point<3>;
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Benchmark::random_points"),
      n);
  auto const a = std::cbrt(n);
  ArborXBenchmark::generatePointCloud(exec, point_cloud_type, a, random_points);

  return random_points;
}

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
    ArborX::query(index, exec_space, queries, indices, offset);

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
struct ArborX::AccessTraits<Iota<MemorySpace>, ArborX::PrimitivesTag>
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
    ArborX::query(index, exec_space, queries, indices, offset);

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

  BENCHMARK(BM_benchmark_attach)
      ->RangeMultiplier(10)
      ->Range(100, 10000)
      ->UseManualTime();

  BENCHMARK(BM_benchmark_iota)
      ->RangeMultiplier(10)
      ->Range(100, 10000)
      ->UseManualTime();

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
