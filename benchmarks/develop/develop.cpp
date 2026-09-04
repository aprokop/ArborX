#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  using value_type = float;

  value_type min_value = -1;
  value_type max_value = 1;

  int const n = 100'000;
  int const num_vectors = 10;

  Kokkos::View<value_type **, MemorySpace> x("x", n, num_vectors);

  constexpr int seed = 1337;
  Kokkos::Random_SFC64_Pool<ExecutionSpace> rand_pool(seed);

  Kokkos::parallel_for(
      "fill_random", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        auto generator = rand_pool.get_state(i);
#else
        auto generator = rand_pool.get_state();
#endif
        for (int j = 0; j < num_vectors; ++j)
          x(i, j) = Kokkos::rand<decltype(generator), value_type>::draw(
              generator, min_value, max_value);
        rand_pool.free_state(generator);
      });

  return EXIT_SUCCESS;
}
