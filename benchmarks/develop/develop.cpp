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

#include <random>

#include <benchmark/benchmark.h>

void BM_kokkos_random(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const n = state.range(0);
  auto const batch_size = state.range(1);

  using Coordinate = float;

  Kokkos::View<Coordinate *> data(
      Kokkos::view_alloc(exec_space, "Benchmark::data",
                         Kokkos::WithoutInitializing),
      n);

  using GeneratorPool = Kokkos::Random_XorShift1024_Pool<ExecutionSpace>;
  using GeneratorType = typename GeneratorPool::generator_type;

  GeneratorPool random_pool(0);

  exec_space.fence();
  for (auto _ : state)
  {
    Kokkos::parallel_for(
        "ArborXBenchmark::filledBoxCloud::generate",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n / batch_size),
        KOKKOS_LAMBDA(int i) {
          auto generator = random_pool.get_state();
          auto random = [&generator]() {
            return Kokkos::rand<GeneratorType, Coordinate>::draw(generator, -1,
                                                                 1);
          };
          auto begin = i * batch_size;
          auto end = Kokkos::min((i + 1) * batch_size, n);
          for (unsigned int k = begin; k < end; ++k)
            data(k) = random();
          random_pool.free_state(generator);
        });
    exec_space.fence();
  }
}

void BM_std_random(benchmark::State &state)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  ExecutionSpace exec_space;

  auto const n = state.range(0);

  using Coordinate = float;

  Kokkos::View<Coordinate *> data(
      Kokkos::view_alloc(exec_space, "Benchmark::data",
                         Kokkos::WithoutInitializing),
      n);

  std::uniform_real_distribution<Coordinate> distribution(-1, 1);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  auto data_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, data);

  exec_space.fence();
  for (auto _ : state)
  {
    for (int i = 0; i < n; ++i)
      data_host(i) = random();
    Kokkos::deep_copy(exec_space, data, data_host);
    exec_space.fence();
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);

  BENCHMARK(BM_std_random)
      ->ArgsProduct({benchmark::CreateRange(100, 100000000, /*multi=*/100)})
      ->Unit(benchmark::kMicrosecond);

  BENCHMARK(BM_kokkos_random)
      ->ArgsProduct({benchmark::CreateRange(100, 100000000, /*multi=*/100),
                     benchmark::CreateRange(1, 4096, /*step=*/4)})
      ->Unit(benchmark::kMicrosecond);

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
