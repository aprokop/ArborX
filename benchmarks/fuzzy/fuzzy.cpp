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

#include <ArborX_DBSCAN.hpp>
#include <ArborX_DBSCANVerification.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

struct Parameters
{
  bool binary;
  int core_min_size;
  float eps;
  std::string filename;
  std::string implementation;
  int n;
  bool verify;
};

using ArborX::ExperimentalHyperGeometry::Point;

template <int DIM>
std::vector<Point<DIM>> loadData(std::string const &filename,
                                 bool binary = true)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  std::vector<Point<DIM>> v;

  int num_points = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }

  ARBORX_ASSERT(dim == DIM);

  v.resize(num_points);
  if (!binary)
  {
    auto it = std::istream_iterator<float>(input);
    for (int i = 0; i < num_points; ++i)
      for (int d = 0; d < DIM; ++d)
        v[i][d] = *it++;
  }
  else
  {
    // Directly read into a point
    input.read(reinterpret_cast<char *>(v.data()),
               num_points * sizeof(Point<DIM>));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  return v;
}

// FIXME: ideally, this function would be next to `loadData` in
// dbscan_timpl.hpp. However, that file is used for explicit instantiation,
// which would result in multiple duplicate symbols. So it is kept here.
int getDataDimension(std::string const &filename, bool binary)
{
  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Error reading file \"" + filename + "\"");

  int num_points;
  int dim;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  input.close();

  return dim;
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

template <int DIM>
bool main_(Parameters const &params)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace exec_space;

  auto data = loadData<DIM>(params.filename, params.binary);

  auto const primitives = vec2view<MemorySpace>(data, "Benchmark::primitives");

  using Primitives = decltype(primitives);

  Kokkos::View<int *, MemorySpace> labels("Example::labels", 0);
  bool success = true;
  using ArborX::DBSCAN::Implementation;
  Implementation implementation = Implementation::FDBSCAN;
  if (params.implementation == "fdbscan-densebox")
    implementation = Implementation::FDBSCAN_DenseBox;

  ArborX::DBSCAN::Parameters dbscan_params;
  dbscan_params.setVerbosity(true).setImplementation(implementation);

  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::total");

  labels = ArborX::dbscan<ExecutionSpace, Primitives>(
      exec_space, primitives, params.eps, params.core_min_size, dbscan_params);

  success = ArborX::Details::verifyDBSCAN(exec_space, primitives, params.eps,
                                          params.core_min_size, labels);
  printf("Verification %s\n", (success ? "passed" : "failed"));

  return success;
}

template <typename T>
std::string vec2string(std::vector<T> const &s, std::string const &delim = ", ")
{
  assert(s.size() > 1);

  std::ostringstream ss;
  std::copy(s.begin(), s.end(),
            std::ostream_iterator<std::string>{ss, delim.c_str()});
  auto delimited_items = ss.str().erase(ss.str().length() - delim.size());
  return "(" + delimited_items + ")";
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
            << std::endl;

  namespace bpo = boost::program_options;

  Parameters params;

  std::vector<std::string> allowed_impls = {"fdbscan", "fdbscan-densebox"};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "binary", bpo::bool_switch(&params.binary), "binary file indicator")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "eps", bpo::value<float>(&params.eps), "DBSCAN eps" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "impl", bpo::value<std::string>(&params.implementation)->default_value("fdbscan"), ("implementation " + vec2string(allowed_impls, " | ")).c_str() )
      ( "verify", bpo::bool_switch(&params.verify), "verify connected components")
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

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };

  if (!found(allowed_impls, params.implementation))
  {
    std::cerr << "Implementation must be one of " << vec2string(allowed_impls)
              << "\n";
    return 2;
  }
  if (params.filename.empty())
  {
    std::cerr << "Must provide filename with data\n";
    return 3;
  }

  std::stringstream ss;
  ss << params.implementation;

  // Print out the runtime parameters
  printf("eps               : %f\n", params.eps);
  printf("implementation    : %s\n", ss.str().c_str());
  printf("verify            : %s\n", (params.verify ? "true" : "false"));
  printf("filename          : %s [%s]\n", params.filename.c_str(),
         (params.binary ? "binary" : "text"));

  auto dim = getDataDimension(params.filename, params.binary);

  bool success;
  switch (dim)
  {
  case 2:
    success = main_<2>(params);
    break;
  case 3:
    success = main_<3>(params);
    break;
  case 4:
    success = main_<4>(params);
    break;
  case 5:
    success = main_<5>(params);
    break;
  case 6:
    success = main_<6>(params);
    break;
  default:
    std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    success = false;
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
