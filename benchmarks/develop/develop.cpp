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
#include <ArborX_TypeErasedGeometry.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>

#include <benchmark/benchmark.h>

struct CustomIndexableGetter
{
  KOKKOS_DEFAULTED_FUNCTION
  CustomIndexableGetter() = default;

  template <typename Geometry>
  KOKKOS_FUNCTION auto const &operator()(Geometry const &geometry) const
  {
    return geometry;
  }

  template <typename Geometry>
  KOKKOS_FUNCTION auto operator()(Geometry &&geometry) const
  {
    return geometry;
  }
};

template <typename DeviceType>
Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<3> *, DeviceType>
constructPoints(int n_values, ArborXBenchmark::PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::ExperimentalHyperGeometry::Point<3> *, DeviceType>
      random_points(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "Benchmark::random_points"),
                    n_values);
  // Generate random points uniformly distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  ArborXBenchmark::generatePointCloud(point_cloud_type, a, random_points);

  return random_points;
}

void BM_construction_points(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  exec_space.fence();
  for (auto _ : state)
  {
    ArborX::BVH<MemorySpace, decltype(points)::value_type> bvh(exec_space,
                                                               points);
    exec_space.fence();
  }
}

void BM_search_knn_points(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  using Points = decltype(points);

  ArborX::BVH<MemorySpace, typename Points::value_type> bvh(exec_space, points);
  auto query_points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<typename Points::value_type *, DeviceType> values("values", 0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    bvh.query(exec_space, ArborX::Experimental::make_nearest(query_points, 1),
              values, offset);

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <int Capacity>
void BM_construction_point_geometries(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  using Geometry = ArborX::Experimental::Geometry<Capacity>;

  Kokkos::View<Geometry *, ExecutionSpace> geometries(
      Kokkos::view_alloc("geometries", Kokkos::WithoutInitializing, exec_space),
      n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) { ::new (&geometries(i)) Geometry(points(i)); });

  exec_space.fence();
  for (auto _ : state)
  {
    ArborX::BVH<MemorySpace, Geometry, CustomIndexableGetter> bvh(exec_space,
                                                                  geometries);

    exec_space.fence();
  }
}

template <int Capacity>
void BM_search_knn_point_geometries(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  auto points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  using Geometry = ArborX::Experimental::Geometry<Capacity>;

  Kokkos::View<Geometry *, ExecutionSpace> geometries(
      Kokkos::view_alloc("geometries", Kokkos::WithoutInitializing, exec_space),
      n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) { ::new (&geometries(i)) Geometry(points(i)); });

  ArborX::BVH<MemorySpace, Geometry, CustomIndexableGetter> bvh(exec_space,
                                                                geometries);

  auto query_points = constructPoints<DeviceType>(
      n, ArborXBenchmark::PointCloudType::filled_box);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<Geometry *, DeviceType> values(
        Kokkos::view_alloc("values", Kokkos::WithoutInitializing, exec_space),
        0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    bvh.query(exec_space, ArborX::Experimental::make_nearest(query_points, 1),
              values, offset);

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    state.SetIterationTime(elapsed_seconds.count());
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  BENCHMARK(BM_construction_points)
      ->RangeMultiplier(10)
      ->Range(100, 10000000)
      ->Unit(benchmark::kMicrosecond);
  BENCHMARK(BM_construction_point_geometries<24>)
      ->RangeMultiplier(10)
      ->Range(100, 10000000)
      ->Unit(benchmark::kMicrosecond);
  // BENCHMARK(BM_construction_point_geometries<32>)
      // ->RangeMultiplier(10)
      // ->Range(100, 10000000)
      // ->Unit(benchmark::kMicrosecond);
  // BENCHMARK(BM_construction_point_geometries<64>)
      // ->RangeMultiplier(10)
      // ->Range(100, 10000000)
      // ->Unit(benchmark::kMicrosecond);

  BENCHMARK(BM_search_knn_points)
      ->RangeMultiplier(10)
      ->Range(100, 10000000)
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  BENCHMARK(BM_search_knn_point_geometries<24>)
      ->RangeMultiplier(10)
      ->Range(100, 10000000)
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  // BENCHMARK(BM_search_knn_point_geometries<32>)
      // ->RangeMultiplier(10)
      // ->Range(100, 10000000)
      // ->UseManualTime()
      // ->Unit(benchmark::kMicrosecond);
  // BENCHMARK(BM_search_knn_point_geometries<64>)
      // ->RangeMultiplier(10)
      // ->Range(100, 10000000)
      // ->UseManualTime()
      // ->Unit(benchmark::kMicrosecond);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
