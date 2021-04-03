/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_NPCF.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    int max_num_points = -1)
{
  std::cout << "Reading in \"" << filename << "\"...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

  if (max_num_points > 0 && max_num_points < num_points)
    num_points = max_num_points;

  std::vector<ArborX::Point> v(num_points);
  input.read(reinterpret_cast<char *>(v.data()),
             num_points * sizeof(ArborX::Point));

  input.close();
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  std::string filename;
  int max_num_points;
  float a;
  float b;
  float delta;
  float eps;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "a", bpo::value<float>(&a)->default_value(0.01f), "a")
      ( "b", bpo::value<float>(&b)->default_value(0.02f), "b")
      ( "delta", bpo::value<float>(&delta)->default_value(1e-2f), "delta")
      ( "eps", bpo::value<float>(&eps)->default_value(1e-2f), "eps")
      ( "filename", bpo::value<std::string>(&filename), "filename containing data" )
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in")
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

  // Print out the runtime parameters
  printf("filename          : %s [max_pts = %d]\n", filename.c_str(),
         max_num_points);
  printf("matcher           : [%.3f, %.3f]\n", a, b);
  printf("delta             : %.3f\n", delta);
  printf("eps               : %.1e\n", eps);

  // read in data
  std::vector<ArborX::Point> data = loadData(filename, max_num_points);
  auto const points = vec2view<MemorySpace>(data, "points");

  ExecutionSpace exec_space;

  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;
  auto timer_start = [&exec_space](Kokkos::Timer &timer) {
    exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space](Kokkos::Timer const &timer) {
    exec_space.fence();
    return timer.seconds();
  };

  timer_start(timer);
  int num_exact = countExact(exec_space, points, a, b);
  elapsed["exact"] = timer_seconds(timer);

  timer_start(timer);
  int num_randomized = countRandomized(exec_space, points, a, b, eps, delta);
  elapsed["randomized"] = timer_seconds(timer);

  printf("#exact      : %d\n", num_exact);
  printf("#randomized : %d\n", num_randomized);
  printf("error       : %.5f\n",
         (float)std::abs(num_randomized - num_exact) / num_exact);

  printf("\nTimers:\n");
  printf("exact      : %10.3f\n", elapsed["exact"]);
  printf("randomized : %10.3f\n", elapsed["randomized"]);

  return EXIT_SUCCESS;
}
