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

#include <Kokkos_Core.hpp>

#include <benchmark/benchmark.h>

void BM_kokkos(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  Kokkos::View<size_t *> view(Kokkos::view_alloc(exec_space, "Benchmark::view"),
                              n);

  exec_space.fence();
  for (auto _ : state)
  {
    Kokkos::parallel_scan(
        "Benchmark::scan",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(const size_t i, size_t &update, bool const is_final) {
          update += i;
          if (is_final)
            view(i) = update;
        });
    exec_space.fence();
  }
}
void BM_thrust(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  thrust::device_vector<size_t> out(n);

  exec_space.fence();
  for (auto _ : state)
  {
    thrust::inclusive_scan(thrust::counting_iterator<size_t>(0),
                           thrust::counting_iterator<size_t>(n), view.begin());
    exec_space.fence();
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  BENCHMARK(BM_thrust)->RangeMultiplier(10)->Range(100, 10000000);
  BENCHMARK(BM_kokkos)->RangeMultiplier(10)->Range(100, 10000000);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
