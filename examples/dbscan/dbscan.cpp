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

#include <ArborX_DBSCAN.hpp>
#include <ArborX_DetailsDBSCANVerification.hpp>
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_DetailsOperatorFunctionObjects.hpp> // Less
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

std::vector<ArborX::Point> parsePoints(std::string const &filename,
                                       bool binary = false,
                                       int max_num_points = -1)
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

  int num_points = 0;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  if (!binary)
  {
    input >> num_points;

    x.reserve(num_points);
    y.reserve(num_points);
    z.reserve(num_points);

    auto read_float = [&input]() {
      return *(std::istream_iterator<float>(input));
    };
    std::generate_n(std::back_inserter(x), num_points, read_float);
    std::generate_n(std::back_inserter(y), num_points, read_float);
    std::generate_n(std::back_inserter(z), num_points, read_float);
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);
    input.read(reinterpret_cast<char *>(x.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(y.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(z.data()), num_points * sizeof(float));
  }
  input.close();
  if (max_num_points != -1)
  {
    num_points = std::min(num_points, max_num_points);
    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);
  }
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  std::vector<ArborX::Point> v(num_points);
  for (int i = 0; i < num_points; i++)
  {
    v[i] = {x[i], y[i], z[i]};
  }

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

template <typename ExecutionSpace, typename LabelsView,
          typename ClusterIndicesView, typename ClusterOffsetView>
void sortAndFilterClusters(ExecutionSpace const &exec_space,
                           LabelsView const &labels,
                           ClusterIndicesView &cluster_indices,
                           ClusterOffsetView &cluster_offset,
                           int cluster_min_size = 2)
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::sortAndFilterClusters");

  static_assert(Kokkos::is_view<LabelsView>{}, "");
  static_assert(Kokkos::is_view<ClusterIndicesView>{}, "");
  static_assert(Kokkos::is_view<ClusterOffsetView>{}, "");

  using MemorySpace = typename LabelsView::memory_space;

  static_assert(std::is_same<typename LabelsView::value_type, int>{}, "");
  static_assert(std::is_same<typename ClusterIndicesView::value_type, int>{},
                "");
  static_assert(std::is_same<typename ClusterOffsetView::value_type, int>{},
                "");

  static_assert(std::is_same<typename LabelsView::memory_space, MemorySpace>{},
                "");
  static_assert(
      std::is_same<typename ClusterIndicesView::memory_space, MemorySpace>{},
      "");
  static_assert(
      std::is_same<typename ClusterOffsetView::memory_space, MemorySpace>{},
      "");

  ARBORX_ASSERT(cluster_min_size >= 2);

  int const n = labels.extent_int(0);

  Kokkos::View<int *, MemorySpace> cluster_sizes(
      "ArborX::DBSCAN::cluster_sizes", n);
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_sizes",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // Ignore noise points
                         if (labels(i) < 0)
                           return;

                         Kokkos::atomic_fetch_add(&cluster_sizes(labels(i)), 1);
                       });

  // This kernel serves dual purpose:
  // - it constructs an offset array through exclusive prefix sum, with a
  //   caveat that small clusters (of size < cluster_min_size) are filtered out
  // - it creates a mapping from a cluster index into the cluster's position in
  //   the offset array
  // We reuse the cluster_sizes array for the second, creating a new alias for
  // it for clarity.
  auto &map_cluster_to_offset_position = cluster_sizes;
  int constexpr IGNORED_CLUSTER = -1;
  int num_clusters;
  ArborX::reallocWithoutInitializing(cluster_offset, n + 1);
  Kokkos::parallel_scan(
      "ArborX::DBSCAN::compute_cluster_offset_with_filter",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        bool is_cluster_too_small = (cluster_sizes(i) < cluster_min_size);
        if (!is_cluster_too_small)
        {
          if (final_pass)
          {
            cluster_offset(update) = cluster_sizes(i);
            map_cluster_to_offset_position(i) = update;
          }
          ++update;
        }
        else
        {
          if (final_pass)
            map_cluster_to_offset_position(i) = IGNORED_CLUSTER;
        }
      },
      num_clusters);
  Kokkos::resize(Kokkos::WithoutInitializing, cluster_offset, num_clusters + 1);
  ArborX::exclusivePrefixSum(exec_space, cluster_offset);

  auto cluster_starts = ArborX::clone(exec_space, cluster_offset);
  ArborX::reallocWithoutInitializing(cluster_indices,
                                     ArborX::lastElement(cluster_offset));
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_indices",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // Ignore noise points
                         if (labels(i) < 0)
                           return;

                         auto offset_pos =
                             map_cluster_to_offset_position(labels(i));
                         if (offset_pos != IGNORED_CLUSTER)
                         {
                           auto position = Kokkos::atomic_fetch_add(
                               &cluster_starts(offset_pos), 1);
                           cluster_indices(position) = i;
                         }
                       });

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename Primitives,
          typename ClusterIndicesView, typename ClusterOffsetView>
void printClusterSizesAndCenters(ExecutionSpace const &exec_space,
                                 Primitives const &primitives,
                                 ClusterIndicesView &cluster_indices,
                                 ClusterOffsetView &cluster_offset)
{
  auto const num_clusters = static_cast<int>(cluster_offset.size()) - 1;

  using MemorySpace = typename ClusterIndicesView::memory_space;

  Kokkos::View<ArborX::Point *, MemorySpace> cluster_centers(
      Kokkos::ViewAllocateWithoutInitializing("Testing::centers"),
      num_clusters);
  Kokkos::parallel_for(
      "Testing::compute_centers",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_clusters),
      KOKKOS_LAMBDA(int const i) {
        // The only reason we sort indices here is for reproducibility.
        // Current DBSCAN algorithm does not guarantee that the indices
        // corresponding to the same cluster are going to appear in the same
        // order from run to run. Using sorted indices, we explicitly
        // guarantee the same summation order when computing cluster centers.

        auto *cluster_start = cluster_indices.data() + cluster_offset(i);
        auto cluster_size = cluster_offset(i + 1) - cluster_offset(i);

        // Sort cluster indices in ascending order. This uses heap for
        // sorting, only because there is no other convenient utility that
        // could sort within a kernel.
        ArborX::Details::makeHeap(cluster_start, cluster_start + cluster_size,
                                  ArborX::Details::Less<int>());
        ArborX::Details::sortHeap(cluster_start, cluster_start + cluster_size,
                                  ArborX::Details::Less<int>());

        // Compute cluster centers
        ArborX::Point cluster_center{0.f, 0.f, 0.f};
        for (int j = cluster_offset(i); j < cluster_offset(i + 1); j++)
        {
          auto const &cluster_point = primitives(cluster_indices(j));
          // NOTE The explicit casts below are intended to silence warnings
          // about narrowing conversion from 'int' to 'float'. A potential
          // accuracy issue here is that 'float' can represent all integer
          // values in the range [-2^23, 2^23] but 'int' can actually represent
          // values in the range [-2^31, 2^31-1]. However, we ignore it for
          // now.
          cluster_center[0] +=
              cluster_point[0] / static_cast<float>(cluster_size);
          cluster_center[1] +=
              cluster_point[1] / static_cast<float>(cluster_size);
          cluster_center[2] +=
              cluster_point[2] / static_cast<float>(cluster_size);
        }
        cluster_centers(i) = cluster_center;
      });

  auto cluster_offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cluster_offset);
  auto cluster_centers_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cluster_centers);
  for (int i = 0; i < num_clusters; i++)
  {
    int cluster_size = cluster_offset_host(i + 1) - cluster_offset_host(i);

    // This is HACC specific filtering. It is only interested in the clusters
    // with centers in [0,64]^3 domain.
    auto const &cluster_center = cluster_centers_host(i);
    if (cluster_center[0] >= 0 && cluster_center[1] >= 0 &&
        cluster_center[2] >= 0 && cluster_center[0] < 64 &&
        cluster_center[1] < 64 && cluster_center[2] < 64)
    {
      printf("%d %e %e %e\n", cluster_size, cluster_center[0],
             cluster_center[1], cluster_center[2]);
    }
  }
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
  std::string algorithm_str;
  bool binary;
  bool verify;
  bool print_dbscan_timers;
  bool print_sizes_centers;
  float eps;
  int cluster_min_size;
  int core_min_size;
  int max_num_points;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "algorithm", bpo::value<std::string>(&algorithm_str)->default_value("dbscan"), "DBSCAN algorithm ('dbscan' or 'dbscan_star')")
      ( "filename", bpo::value<std::string>(&filename), "filename containing data" )
      ( "binary", bpo::bool_switch(&binary)->default_value(false), "binary file indicator")
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in")
      ( "eps", bpo::value<float>(&eps), "DBSCAN eps" )
      ( "cluster-min-size", bpo::value<int>(&cluster_min_size)->default_value(2), "minimum cluster size")
      ( "core-min-size", bpo::value<int>(&core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "verify", bpo::bool_switch(&verify)->default_value(false), "verify connected components")
      ( "print-dbscan-timers", bpo::bool_switch(&print_dbscan_timers)->default_value(false), "print dbscan timers")
      ( "output-sizes-and-centers", bpo::bool_switch(&print_sizes_centers)->default_value(false), "print cluster sizes and centers")
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

  ARBORX_ASSERT(algorithm_str == "dbscan" || algorithm_str == "dbscan_star");

  // Print out the runtime parameters
  printf("eps               : %f\n", eps);
  printf("minpts            : %d\n", core_min_size);
  printf("cluster min size  : %d\n", cluster_min_size);
  printf("algorithm         : %s\n", algorithm_str.c_cstr());
  printf("filename          : %s [%s, max_pts = %d]\n", filename.c_str(),
         (binary ? "binary" : "text"), max_num_points);
  printf("verify            : %s\n", (verify ? "true" : "false"));
  printf("print timers      : %s\n", (print_dbscan_timers ? "true" : "false"));
  printf("output centers    : %s\n", (print_sizes_centers ? "true" : "false"));

  ArborX::DBSCAN::Algorithm algorithm;
  if (algorithm_str == "dbscan")
    algorithm = ArborX::DBSCAN::Algorithm::DBSCAN;
  else
    algorithm = ArborX::DBSCAN::Algorithm::DBSCANStar;

  // read in data
  auto const primitives = vec2view<MemorySpace>(
      parsePoints(filename, binary, max_num_points), "primitives");

  ExecutionSpace exec_space;

  Kokkos::Timer timer_total;
  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = print_dbscan_timers;
  auto timer_start = [&exec_space, verbose](Kokkos::Timer &timer) {
    if (verbose)
      exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space, verbose](Kokkos::Timer const &timer) {
    if (verbose)
      exec_space.fence();
    return timer.seconds();
  };

  timer_start(timer_total);

  auto labels = ArborX::dbscan(exec_space, primitives, eps, core_min_size,
                               ArborX::DBSCAN::Parameters()
                                   .setPrintTimers(print_dbscan_timers)
                                   .setAlgorithm(algorithm));

  timer_start(timer);
  Kokkos::View<int *, MemorySpace> cluster_indices("Testing::cluster_indices",
                                                   0);
  Kokkos::View<int *, MemorySpace> cluster_offset("Testing::cluster_offset", 0);
  sortAndFilterClusters(exec_space, labels, cluster_indices, cluster_offset,
                        cluster_min_size);
  elapsed["cluster"] = timer_seconds(timer);
  elapsed["total"] = timer_seconds(timer_total);

  printf("-- postprocess      : %10.3f\n", elapsed["cluster"]);
  printf("total time          : %10.3f\n", elapsed["total"]);

  if (verify)
  {
    auto passed = ArborX::Details::verifyDBSCAN(
        exec_space, primitives, eps, core_min_size, algorithm, labels);
    printf("Verification %s\n", (passed ? "passed" : "failed"));
  }

  if (print_sizes_centers)
    printClusterSizesAndCenters(exec_space, primitives, cluster_indices,
                                cluster_offset);

  return EXIT_SUCCESS;
}
