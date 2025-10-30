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

#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

#include <benchmark/benchmark.h>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  using MemorySpace = Kokkos::HostSpace;
  ExecutionSpace space{};

  // using Scalar = float;
  using Scalar = double;

  using Point = ArborX::Point<2, Scalar>;

  auto f = KOKKOS_LAMBDA(Point p)
  {
    return Kokkos::sin(4 * p[0]) + Kokkos::sin(2 * p[1]);
  };

  constexpr int num_targets = 4;
  Kokkos::View<Point *, MemorySpace> target_coords("target_coords",
                                                   num_targets);
  Kokkos::View<Scalar *, MemorySpace> target_values("target_values",
                                                    num_targets);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, 1), KOKKOS_LAMBDA(int) {
        target_coords(0) = {.788675, .788675};
        target_coords(1) = {.211325, .788675};
        target_coords(2) = {.788675, .211325};
        target_coords(3) = {.211325, .211325};
        for (int i = 0; i < num_targets; ++i)
          target_values(i) = f(target_coords(i));
      });

  constexpr int num_refinements = 20;
  printf("   |    h     |  error\n");
  printf("-- | -------- | --------\n");
  for (int i = 1; i < num_refinements; ++i)
  {
    int n = (1 << i) + 1;
    Scalar h = 2. / (n - 1);

    // We construct 25 points around each target point to avoid constructing a
    // full mesh grid. 25 points is sufficient to find the 6 nearest neighbors
    // needed for the 2nd-order polynomial basis.
    int const n_points_per_target_1d = 5;
    int const n_points_per_target =
        n_points_per_target_1d * n_points_per_target_1d;
    Kokkos::View<Point *, MemorySpace> source_coords(
        "source_coords", n_points_per_target * num_targets);
    Kokkos::View<Scalar *, MemorySpace> source_values(
        "source_values", n_points_per_target * num_targets);
    Kokkos::parallel_for(
        Kokkos::RangePolicy(space, 0, num_targets),
        KOKKOS_LAMBDA(int target_index) {
          int const start_x = target_coords(target_index)[0] / h;
          int const start_y = target_coords(target_index)[1] / h;
          for (int i = -2; i <= 2; ++i)
            for (int j = -2; j <= 2; ++j)
            {
              int const linear_index = target_index * n_points_per_target +
                                       (i + 2) * n_points_per_target_1d +
                                       (j + 2);
              source_coords(linear_index) = {(start_x + i) * h,
                                             (start_y + j) * h};
              source_values(linear_index) = f(source_coords(linear_index));
            }
        });

    ArborX::Interpolation::MovingLeastSquares<MemorySpace, Scalar> mls(
        space, source_coords, target_coords);

    Kokkos::View<Scalar *, MemorySpace> interpolated_values(
        "interpolated_values", num_targets);
    mls.interpolate(space, source_values, interpolated_values);

    Scalar max_error;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy(space, 0, num_targets),
        KOKKOS_LAMBDA(int const i, Scalar &tmp_error) {
          auto error = Kokkos::abs(target_values(i) - interpolated_values(i));
          tmp_error = Kokkos::max(tmp_error, error);
        },
        Kokkos::Max<Scalar, Kokkos::HostSpace>{max_error});
    printf("%2d | %7.2e | %7.2e\n", i, h, max_error);
    fflush(stdout);
  }
}
