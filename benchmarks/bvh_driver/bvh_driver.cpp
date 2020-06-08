/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_BoostRTreeHelpers.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_NanoflannAdapters.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

#ifdef ARBORX_PERFORMANCE_TESTING
#include <mpi.h>
#endif

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

// #define CALCULATE_NEIGHBOR_STATS

struct Spec
{
  std::string create_label_construction(std::string const &tree_name) const
  {
    std::string s = std::string("BM_construction<") + tree_name + ">";
    for (auto const &var :
         {n_values, static_cast<int>(source_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

  std::string create_label_knn_search(std::string const &tree_name) const
  {
    std::string s = std::string("BM_knn_search<") + tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

  std::string create_label_radius_search(std::string const &tree_name) const
  {
    std::string s = std::string("BM_radius_search<") + tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          buffer_size, static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  bool sort_predicates;
  int buffer_size;
  PointCloudType source_point_cloud_type;
  PointCloudType target_point_cloud_type;
};

Spec create_spec_from_string(std::string const &spec_string)
{
  std::istringstream ss(spec_string);
  std::string token;

  Spec spec;

  // clang-format off
    getline(ss, token, '/');  spec.backends = token;
    getline(ss, token, '/');  spec.n_values = std::stoi(token);
    getline(ss, token, '/');  spec.n_queries = std::stoi(token);
    getline(ss, token, '/');  spec.n_neighbors = std::stoi(token);
    getline(ss, token, '/');  spec.sort_predicates = static_cast<bool>(std::stoi(token));
    getline(ss, token, '/');  spec.buffer_size = std::stoi(token);
    getline(ss, token, '/');  spec.source_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    getline(ss, token, '/');  spec.target_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
  // clang-format on

  if (!(spec.backends == "all" || spec.backends == "serial" ||
        spec.backends == "openmp" || spec.backends == "threads" ||
        spec.backends == "cuda" || spec.backends == "rtree" ||
        spec.backends == "nanoflann"))
    throw std::runtime_error("Backend " + spec.backends + " invalid!");

  return spec;
}

#ifdef KOKKOS_ENABLE_SERIAL
class NanoflannKDTree
{
public:
  using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
  using ExecutionSpace = Kokkos::Serial;
  using device_type = DeviceType;

  using DatasetAdapter = ArborX::NanoflannPointCloudAdapter<Kokkos::HostSpace>;
  using CoordinateType = DatasetAdapter::CoordinateType;
  using SizeType = DatasetAdapter::SizeType;

  NanoflannKDTree(Kokkos::View<ArborX::Point *, DeviceType> points)
      : _dataset_adapter(points)
      , _tree(3, _dataset_adapter)
  {
    _tree.buildIndex();
  }

  template <typename Query>
  void query(Kokkos::View<Query *, DeviceType> queries,
             Kokkos::View<int *, DeviceType> &indices,
             Kokkos::View<int *, DeviceType> &offset,
             ArborX::Experimental::TraversalPolicy const &)
  {
    using Predicates = Kokkos::View<Query *, DeviceType>;
    using Access =
        ArborX::Traits::Access<Predicates, ArborX::Traits::PredicatesTag>;
    using Tag = typename ArborX::Traits::Helper<Access>::tag;

    int const n_queries = queries.extent_int(0);
    Kokkos::realloc(offset, n_queries + 1);

    std::vector<std::pair<SizeType, CoordinateType>> returned_indices_distances;

    queryDispatch(Tag{}, queries, returned_indices_distances, offset);

    ArborX::exclusivePrefixSum(ExecutionSpace{}, offset);
    int const n_results = ArborX::lastElement(offset);

    Kokkos::realloc(indices, n_results);
    for (int i = 0; i < n_queries; ++i)
      for (int j = offset(i); j < offset(i + 1); ++j)
        indices(j) = returned_indices_distances[j].first;
  }

private:
  using DistanceType =
      nanoflann::L2_Simple_Adaptor<CoordinateType, DatasetAdapter>;
  using KDTree =
      nanoflann::KDTreeSingleIndexAdaptor<DistanceType, DatasetAdapter, 3,
                                          SizeType>;

  template <typename Query>
  void queryDispatch(ArborX::Details::SpatialPredicateTag,
                     Kokkos::View<Query *, DeviceType> queries,
                     std::vector<std::pair<SizeType, CoordinateType>>
                         &returned_indices_distances,
                     Kokkos::View<int *, DeviceType> &offset)
  {
    int const n_queries = queries.extent_int(0);

    std::vector<std::pair<SizeType, CoordinateType>> ret_matches;
    for (int i = 0; i < n_queries; ++i)
    {
      auto const sphere = ArborX::getGeometry(queries(i));
      auto const radius = sphere.radius();
      auto const *const centroid = &sphere.centroid()[0];

      ret_matches.resize(0);

      nanoflann::SearchParams params;
      offset(i) =
          _tree.radiusSearch(centroid, radius * radius, ret_matches, params);

      returned_indices_distances.insert(returned_indices_distances.end(),
                                        ret_matches.begin(), ret_matches.end());
    }
  }

  template <typename Query>
  void queryDispatch(ArborX::Details::NearestPredicateTag,
                     Kokkos::View<Query *, DeviceType> queries,
                     std::vector<std::pair<SizeType, CoordinateType>>
                         &returned_indices_distances,
                     Kokkos::View<int *, DeviceType> &offset)
  {
    int const n_queries = queries.extent_int(0);

    std::vector<SizeType> query_indices;
    std::vector<CoordinateType> distances_sq;
    for (int i = 0; i < n_queries; ++i)
    {
      auto const k = queries(i)._k;
      auto const *const query_point = &ArborX::getGeometry(queries(i))[0];

      query_indices.resize(k);
      distances_sq.resize(k);
      memset(query_indices.data(), 0, k * sizeof(SizeType));
      memset(distances_sq.data(), 0, k * sizeof(CoordinateType));

      offset(i) = _tree.knnSearch(query_point, k, query_indices.data(),
                                  distances_sq.data());

      for (int j = 0; j < offset(j); ++j)
        returned_indices_distances.push_back(
            std::make_pair(query_indices[j], std::sqrt(distances_sq[j])));
    }
  }

  DatasetAdapter _dataset_adapter;
  KDTree _tree;
};
#endif

template <typename DeviceType>
Kokkos::View<ArborX::Point *, DeviceType>
constructPoints(int n_values, PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_values);
  // Generate random points uniformly distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  generatePointCloud(point_cloud_type, a, random_points);

  return random_points;
}

template <typename DeviceType>
Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
makeNearestQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_knn_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) =
            ArborX::nearest<ArborX::Point>(random_points(i), n_neighbors);
      });
  return queries;
}

template <typename DeviceType>
Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries(Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  // Radius is computed so that the number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors.
  // Calculation: n_values*(4/3*M_PI*r^3)/(2a)^3 = n_neighbors
  double const r = std::cbrt(static_cast<double>(n_neighbors) * 6. / M_PI);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_radius_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::intersects(ArborX::Sphere{random_points(i), r});
      });
  return queries;
}

template <class TreeType>
void BM_construction(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;
  auto const points =
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type);

  for (auto _ : state)
  {
    auto const start = std::chrono::high_resolution_clock::now();
    TreeType index(points);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_knn_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_radius_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

#ifdef CALCULATE_NEIGHBOR_STATS
  bool first_pass = true;
#endif
  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(spec.sort_predicates)
                    .setBufferSize(spec.buffer_size));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());

#ifdef CALCULATE_NEIGHBOR_STATS
    if (first_pass)
    {
      auto offset_clone = ArborX::clone(offset);
      ArborX::adjacentDifference(offset, offset_clone);
      int const max = ArborX::max(offset_clone);
      int const min = ArborX::min(Kokkos::subview(
          offset_clone, std::make_pair(1, offset_clone.extent_int(0))));
      double const avg = ((double)ArborX::lastElement(offset)) / n_queries;

      printf("#values: %10d  #queries: %10d  neighbors: min=%3d, max=%3d, "
             "avg=%3.2f\n",
             n_values, n_queries, min, max, avg);

      first_pass = false;
    }
#endif
  }
}

class KokkosScopeGuard
{
public:
  KokkosScopeGuard(int &argc, char *argv[]) { Kokkos::initialize(argc, argv); }
  ~KokkosScopeGuard() { Kokkos::finalize(); }
};

template <typename TreeType>
void register_benchmark(std::string const &description, Spec const &spec)
{
  benchmark::RegisterBenchmark(
      spec.create_label_construction(description).c_str(),
      [=](benchmark::State &state) { BM_construction<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark(
      spec.create_label_knn_search(description).c_str(),
      [=](benchmark::State &state) { BM_knn_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark(
      spec.create_label_radius_search(description).c_str(),
      [=](benchmark::State &state) { BM_radius_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

// NOTE Motivation for this class that stores the argument count and values is
// I could not figure out how to make the parser consume arguments with
// Boost.Program_options
// Benchmark removes its own arguments from the command line arguments. This
// means, that by virtue of returning references to internal data members in
// argc() and argv() function, it will necessarily modify the members. It will
// decrease _argc, and "reduce" _argv data. Hence, we must keep a copy of _argv
// that is not modified from the outside to release memory in the destructor
// correctly.
class CmdLineArgs
{
private:
  int _argc;
  std::vector<char *> _argv;
  std::vector<char *> _owner_ptrs;

public:
  CmdLineArgs(std::vector<std::string> const &args, char const *exe)
      : _argc(args.size() + 1)
      , _owner_ptrs{new char[std::strlen(exe) + 1]}
  {
    std::strcpy(_owner_ptrs[0], exe);
    _owner_ptrs.reserve(_argc);
    for (auto const &s : args)
    {
      _owner_ptrs.push_back(new char[s.size() + 1]);
      std::strcpy(_owner_ptrs.back(), s.c_str());
    }
    _argv = _owner_ptrs;
  }

  ~CmdLineArgs()
  {
    for (auto *p : _owner_ptrs)
    {
      delete[] p;
    }
  }

  int &argc() { return _argc; }

  char **argv() { return _argv.data(); }
};

int main(int argc, char *argv[])
{
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  Spec single_spec;
  std::string source_pt_cloud;
  std::string target_pt_cloud;
  std::vector<std::string> exact_specs;
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&single_spec.n_values)->default_value(50000), "number of indexable values (source)" )
        ( "queries", bpo::value<int>(&single_spec.n_queries)->default_value(20000), "number of queries (target)" )
        ( "predicate-sort", bpo::value<bool>(&single_spec.sort_predicates)->default_value(true), "sort predicates" )
        ( "neighbors", bpo::value<int>(&single_spec.n_neighbors)->default_value(10), "desired number of results per query" )
        ( "buffer", bpo::value<int>(&single_spec.buffer_size)->default_value(0), "size for buffer optimization in radius search" )
        ( "source-point-cloud-type", bpo::value<std::string>(&source_pt_cloud)->default_value("filled_box"), "shape of the source point cloud"  )
        ( "target-point-cloud-type", bpo::value<std::string>(&target_pt_cloud)->default_value("filled_box"), "shape of the target point cloud"  )
        ( "no-header", bpo::bool_switch(), "do not print version and hash" )
        ( "exact-spec", bpo::value<std::vector<std::string>>(&exact_specs)->multitoken(), "exact specification (can be specified multiple times for batch)" )
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                   .options(desc)
                                   .allow_unregistered()
                                   .run();
  bpo::store(parsed, vm);
  CmdLineArgs pass_further{
      bpo::collect_unrecognized(parsed.options, bpo::include_positional),
      argv[0]};
  bpo::notify(vm);

  if (!vm["no-header"].as<bool>())
  {
    std::cout << "ArborX version: " << ArborX::version() << std::endl;
    std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
  }

  if (vm.count("help") > 0)
  {
    // Full list of options consists of Kokkos + Boost.Program_options +
    // Google Benchmark and we still need to call benchmark::Initialize() to
    // get those printed to the standard output.
    std::cout << desc << "\n";
    int ac = 2;
    char *av[] = {(char *)"ignored", (char *)"--help"};
    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit(Kokkos::finalize);
    benchmark::Initialize(&ac, av);
    return 1;
  }

  if (vm.count("exact-spec") > 0)
  {
    for (std::string option :
         {"values", "queries", "predicate-sort", "neighbors", "buffer",
          "source-point-cloud-type", "target-point-cloud-type"})
    {
      if (!vm[option].defaulted())
      {
        std::cout << "Conflicting options: 'exact-spec' and '" << option
                  << "', exiting..." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  benchmark::Initialize(&pass_further.argc(), pass_further.argv());
  // Throw if some of the arguments have not been recognized.
  std::ignore =
      bpo::command_line_parser(pass_further.argc(), pass_further.argv())
          .options(bpo::options_description(""))
          .run();

  std::vector<Spec> specs;
  specs.reserve(exact_specs.size());
  for (auto const &spec_string : exact_specs)
    specs.push_back(create_spec_from_string(spec_string));

  if (vm.count("exact-spec") == 0)
  {
    single_spec.backends = "all";
    single_spec.source_point_cloud_type = to_point_cloud_enum(source_pt_cloud);
    single_spec.target_point_cloud_type = to_point_cloud_enum(target_pt_cloud);
    specs.push_back(single_spec);
  }

  for (auto const &spec : specs)
  {
#ifdef KOKKOS_ENABLE_SERIAL
    if (spec.backends == "all" || spec.backends == "serial")
      register_benchmark<ArborX::BVH<Kokkos::Serial::device_type>>(
          "ArborX::BVH<Serial>", spec);
#else
    if (spec.backends == "serial")
      throw std::runtime_error("Serial backend not available!");
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    if (spec.backends == "all" || spec.backends == "openmp")
      register_benchmark<ArborX::BVH<Kokkos::OpenMP::device_type>>(
          "ArborX::BVH<OpenMP>", spec);
#else
    if (spec.backends == "openmp")
      throw std::runtime_error("OpenMP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_THREADS
    if (spec.backends == "all" || spec.backends == "threads")
      register_benchmark<ArborX::BVH<Kokkos::Threads::device_type>>(
          "ArborX::BVH<Threads>", spec);
#else
    if (spec.backends == "threads")
      throw std::runtime_error("Threads backend not available!");
#endif

#ifdef KOKKOS_ENABLE_CUDA
    if (spec.backends == "all" || spec.backends == "cuda")
      register_benchmark<ArborX::BVH<Kokkos::Cuda::device_type>>(
          "ArborX::BVH<Cuda>", spec);
#else
    if (spec.backends == "cuda")
      throw std::runtime_error("CUDA backend not available!");
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    if (spec.backends == "all" || spec.backends == "rtree")
    {
      using BoostRTree = BoostExt::RTree<ArborX::Point>;
      register_benchmark<BoostRTree>("BoostRTree", spec);
    }
#endif

#ifdef KOKKOS_ENABLE_SERIAL
    if (spec.backends == "all" || spec.backends = "nanoflann")
    {
      register_benchmark<NanoflannKDTree>("NanoflannKDTree", spec);
    }
#endif
  }

  benchmark::RunSpecifiedBenchmarks();

  Kokkos::finalize();
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
