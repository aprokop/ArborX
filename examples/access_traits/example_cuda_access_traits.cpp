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

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <numeric>

struct PointCloud
{
  float *d_x;
  float *d_y;
  float *d_z;
  int N;
};

struct Spheres
{
  float *d_x;
  float *d_y;
  float *d_z;
  float *d_r;
  int N;
};

template <>
struct ArborX::AccessTraits<PointCloud>
{
  static KOKKOS_FUNCTION std::size_t size(PointCloud const &cloud)
  {
    return cloud.N;
  }
  static KOKKOS_FUNCTION auto get(PointCloud const &cloud, std::size_t i)
  {
    return ArborX::Point{cloud.d_x[i], cloud.d_y[i], cloud.d_z[i]};
  }
  using memory_space = Kokkos::CudaSpace;
};

template <>
struct ArborX::AccessTraits<Spheres>
{
  static KOKKOS_FUNCTION std::size_t size(Spheres const &d) { return d.N; }
  static KOKKOS_FUNCTION auto get(Spheres const &d, std::size_t i)
  {
    return ArborX::intersects(
        ArborX::Sphere{ArborX::Point{d.d_x[i], d.d_y[i], d.d_z[i]}, d.d_r[i]});
  }
  using memory_space = Kokkos::CudaSpace;
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr std::size_t N = 10;
  std::array<float, N> a;

  float *d_a;
  cudaMalloc(&d_a, sizeof(a));

  std::iota(std::begin(a), std::end(a), 1.0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  Kokkos::push_finalize_hook([stream]() { cudaStreamDestroy(stream); });

  cudaMemcpyAsync(d_a, a.data(), sizeof(a), cudaMemcpyHostToDevice, stream);

  Kokkos::Cuda cuda{stream};
  ArborX::BoundingVolumeHierarchy bvh{cuda, PointCloud{d_a, d_a, d_a, N}};

  Kokkos::View<ArborX::Point<3> *, Kokkos::CudaSpace> points("Example::points",
                                                             0);
  Kokkos::View<int *, Kokkos::CudaSpace> offset("Example::offset", 0);
  ArborX::query(bvh, cuda, Spheres{d_a, d_a, d_a, d_a, N}, points, offset);

  Kokkos::parallel_for(
      "Example::print_points", Kokkos::RangePolicy(cuda, 0, N),
      KOKKOS_LAMBDA(int i) {
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          printf("%i: (%.1f, %.1f, %.1f)\n", i, points(j)[0], points(j)[1],
                 points(j)[2]);
        }
      });

  return 0;
}
