/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>

#include <boost/program_options.hpp>

struct UnweightedEdge
{
  unsigned int source;
  unsigned int target;
};

struct WangUnionFind
{
  std::vector<int> _parents;

  // initialize n elements all as roots
  WangUnionFind(int n) { _parents.resize(n, -1); }

  int representative(int i)
  {
    if (is_root(i))
      return i;
    int p = _parents[i];
    if (is_root(p))
      return p;

    // find root, shortcutting along the way
    do
    {
      int gp = _parents[p];
      _parents[i] = gp;
      i = p;
      p = gp;
    } while (!is_root(p));
    return p;
  }

  bool is_root(int u) const { return _parents[u] == -1; }

  // TODO: This is different from the original Wang's code. There, the code
  // simply did
  //    _parents[u] = v;
  // I don't understand the preconditions for this to work. Lets say we have 3
  // points, 0, 1, 2, and two edges, [0, 1] and [0, 2]. If we use the original
  // function, both 1 and 2 would be roots, and 0 would point to 2. So,
  // representative(1) != representative(2).
  //
  // They talk a bit about cycles or always linking from larger vertex id to a
  // smaller one. Don't understand what that means.
  // void merge(int u, int v) { _parents[representative(u)] = v; }
  void merge(int u, int v)
  {
    if (u > v)
      _parents[representative(u)] = v;
    else
      _parents[representative(v)] = u;
  }
};

template <typename ExecutionSpace, typename Edges, typename UnionFind>
double kernel(ExecutionSpace const &exec_space, Edges edges_in,
              UnionFind union_find)
{
  auto edges = Kokkos::create_mirror_view_and_copy(
      typename ExecutionSpace::memory_space{}, edges_in);
  Kokkos::fence();
  Kokkos::Timer timer;
  if constexpr (!std::is_same_v<ExecutionSpace, Kokkos::Serial>)
  {
    Kokkos::parallel_for(
        "ArborX::Bechmark::union-find",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
        KOKKOS_LAMBDA(int e) {
          int i = edges(e).source;
          int j = edges(e).target;

          // printf("(%d, %d)\n", i, j);

          union_find.merge(i, j);
        });
  }
  else
  {
    int num_edges = edges.size();
    for (int e = 0; e < num_edges; ++e)
    {
      int i = edges(e).source;
      int j = edges(e).target;

      union_find.merge(i, j);
    }
  }
  Kokkos::fence();
  return timer.seconds();
}

template <typename ExecutionSpace>
auto buildUnionFind(ExecutionSpace const &exec_space, int n)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::labels"),
      n);
  ArborX::iota(exec_space, labels);
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Serial>)
    return ArborX::Details::UnionFind<MemorySpace, true>(labels);
  else
#endif
    return ArborX::Details::UnionFind<MemorySpace, false>(labels);
}

enum LoopStatus
{
  ALLOW,
  DISALLOW
};

template <typename ExecutionSpace>
Kokkos::View<UnweightedEdge *, typename ExecutionSpace::memory_space>
buildEdges(ExecutionSpace const &exec_space, int num_edges,
           LoopStatus loop_status = LoopStatus::ALLOW)
{
  using MemorySpace = typename ExecutionSpace::memory_space;
  Kokkos::View<UnweightedEdge *, MemorySpace> edges(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Benchmark::edges"),
      num_edges);

  Kokkos::Random_XorShift1024_Pool<MemorySpace> rand_pool(1984);
  if (loop_status == LoopStatus::ALLOW)
  {
    Kokkos::parallel_for(
        "ArborX::Bechmark::init",
        Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, num_edges),
        KOKKOS_LAMBDA(unsigned i) {
          auto rand_gen = rand_pool.get_state();
          do
          {
            edges(i) = {rand_gen.urand() % num_edges,
                        rand_gen.urand() % num_edges};
          } while (edges(i).source == edges(i).target); // no self loops
          rand_pool.free_state(rand_gen);
        });
  }
  else
  {
    // construct random permutation
    Kokkos::View<int *, MemorySpace> random_values(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Benchmark::random_values"),
        num_edges);
    Kokkos::parallel_for(
        "ArborX::Bechmark::init_random_values",
        Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, num_edges),
        KOKKOS_LAMBDA(int i) {
          auto rand_gen = rand_pool.get_state();
          random_values(i) = rand_gen.rand();
          rand_pool.free_state(rand_gen);
        });
    auto permute = ArborX::Details::sortObjects(exec_space, random_values);
    // init edges in random order
    Kokkos::parallel_for(
        "ArborX::Bechmark::init",
        Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, num_edges),
        KOKKOS_LAMBDA(unsigned i) {
          auto rand_gen = rand_pool.get_state();
          // edges(permute(i)) = {rand_gen.urand() % (i + 1), i + 1};
          edges(permute(i)) = {i + 1, rand_gen.urand() % (i + 1)};
          rand_pool.free_state(rand_gen);
        });
  }
  return edges;
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  namespace bpo = boost::program_options;

  int n;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "n", bpo::value<int>(&n)->default_value(50000000), "size" )
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }
  printf("n                 : %d\n", n);
  auto const num_edges = n - 1;

  ExecutionSpace exec_space;

  auto edges_with_loops = buildEdges(exec_space, num_edges, LoopStatus::ALLOW);
  auto edges_wo_loops = buildEdges(exec_space, num_edges, LoopStatus::DISALLOW);

#ifdef KOKKOS_ENABLE_SERIAL
  {
    Kokkos::Serial space;
    printf("Serial (loops)          : %.4lf\n",
           kernel(space, edges_with_loops, buildUnionFind(space, n)));
    printf("Serial (no loops)       : %.4lf\n",
           kernel(space, edges_wo_loops, buildUnionFind(space, n)));
    printf("Serial (no loops, Wang) : %.4lf\n",
           kernel(space, edges_wo_loops, WangUnionFind(n)));
  }
#endif

#ifdef KOKKOS_ENABLE_CUDA
  {
    Kokkos::Cuda space;
    printf("CUDA   (loops)          : %.4lf\n",
           kernel(space, edges_with_loops, buildUnionFind(space, n)));
    printf("CUDA   (no loops)       : %.4lf\n",
           kernel(space, edges_wo_loops, buildUnionFind(space, n)));
  }
#endif

  return EXIT_SUCCESS;
}
