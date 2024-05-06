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

#include <benchmark/benchmark.h>

template <typename Geometries, int Capacity>
struct TypeErasureWrapper
{
  Geometries _geometries;
};

template <typename Geometries, int Capacity>
struct ArborX::AccessTraits<TypeErasureWrapper<Geometries, Capacity>,
                            ArborX::PrimitivesTag>
{
  using Self = TypeErasureWrapper<Geometries, Capacity>;
  using Geometry = ArborX::Experimental::Geometry<Capacity>;

  using memory_space = typename Geometries::memory_space;

  static KOKKOS_FUNCTION auto size(Self const &x)
  {
    return x._geometries.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, int i)
  {
    return Geometry(x._geometries(i));
  }
};

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

  exec_space.fence();
  for (auto _ : state)
  {
    ArborX::BVH<MemorySpace, ArborX::Experimental::Geometry<Capacity>,
                CustomIndexableGetter>
        bvh(exec_space, TypeErasureWrapper<decltype(points), Capacity>{points});

    exec_space.fence();
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  BENCHMARK(BM_construction_points)->RangeMultiplier(10)->Range(100, 10000);
  BENCHMARK(BM_construction_point_geometries<32>)
      ->RangeMultiplier(10)
      ->Range(100, 10000);
  BENCHMARK(BM_construction_point_geometries<64>)
      ->RangeMultiplier(10)
      ->Range(100, 10000);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
