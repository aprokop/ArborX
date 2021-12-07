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
#include <ArborX_DBSCANVerification.hpp>
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_DetailsOperatorFunctionObjects.hpp> // Less
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    bool binary = true, int max_num_points = -1)
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

  std::vector<ArborX::Point> v;

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

  // For now, only allow reading in 2D or 3D data. Will relax in the future.
  ARBORX_ASSERT(dim == 2 || dim == 3);

  if (max_num_points > 0 && max_num_points < num_points)
    num_points = max_num_points;

  if (!binary)
  {
    v.reserve(num_points);

    auto it = std::istream_iterator<float>(input);
    auto read_point = [&it, dim]() {
      float xyz[3] = {0.f, 0.f, 0.f};
      for (int i = 0; i < dim; ++i)
        xyz[i] = *it++;
      return ArborX::Point{xyz[0], xyz[1], xyz[2]};
    };
    std::generate_n(std::back_inserter(v), num_points, read_point);
  }
  else
  {
    v.resize(num_points);

    if (dim == 3)
    {
      // Can directly read into ArborX::Point
      input.read(reinterpret_cast<char *>(v.data()),
                 num_points * sizeof(ArborX::Point));
    }
    else
    {
      std::vector<float> aux(num_points * dim);
      input.read(reinterpret_cast<char *>(aux.data()),
                 aux.size() * sizeof(float));

      for (int i = 0; i < num_points; ++i)
      {
        ArborX::Point p{0.f, 0.f, 0.f};
        for (int d = 0; d < dim; ++d)
          p[d] = aux[i * dim + d];
        v[i] = p;
      }
    }
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

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

template <typename MemorySpace>
struct Count
{
  Kokkos::View<int *, MemorySpace> _counts;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int) const
  {
    auto i = getData(query);
    Kokkos::atomic_fetch_add(&_counts(i), 1);
  }
};


int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  std::string filename;
  float eps;
  int core_min_size;
  int max_num_points;
  bool binary;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "filename", bpo::value<std::string>(&filename), "filename containing data" )
      ( "binary", bpo::bool_switch(&binary)->default_value(false), "binary file indicator")
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in")
      ( "eps", bpo::value<float>(&eps), "DBSCAN eps" )
      ( "core-min-size", bpo::value<int>(&core_min_size)->default_value(2), "DBSCAN min_pts")
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
  printf("eps               : %f\n", eps);
  printf("minpts            : %d\n", core_min_size);
  printf("filename          : %s [%s, max_pts = %d]\n", filename.c_str(),
         (binary ? "binary" : "text"), max_num_points);

  // read in data
  std::vector<ArborX::Point> data = loadData(filename, binary, max_num_points);
  auto const primitives = vec2view<MemorySpace>(data, "primitives");

  ExecutionSpace exec_space;

  using Primitives = decltype(primitives);
  auto const predicates =
      ArborX::Details::PrimitivesWithRadius<Primitives>{primitives, eps};

  auto const n = data.size();
  Kokkos::View<int *, MemorySpace> num_neigh_0("num_neighbors_0", n);

  ArborX::BVH<MemorySpace> bvh_0(exec_space, primitives);
  bvh_0.query(exec_space, predicates,
            Count<MemorySpace>{num_neigh_0},
            ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
  for (int k = 0; k < 1000; ++k) {
      printf("Iteration: %d\n", k);

      ArborX::BVH<MemorySpace> bvh(exec_space, primitives);

      Kokkos::View<int *, MemorySpace> num_neigh("num_neighbors", n);
      bvh.query(exec_space, predicates,
                Count<MemorySpace>{num_neigh},
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
      Kokkos::fence();

      int num_diff = 0;
      Kokkos::parallel_reduce("compare_neighbors", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                           KOKKOS_LAMBDA(int i, int &update) {
          if (num_neigh(i) != num_neigh_0(i))
              ++update;
      }, num_diff);
      Kokkos::fence();


      if (num_diff > 0) {
          auto num_neigh_0_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_neigh_0);
          auto num_neigh_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_neigh);
          Kokkos::fence();

          for (int i = 0; i < (int)n; ++i)
              if (num_neigh_host(i) != num_neigh_0_host(i))
                  printf("[%d]: %d [%d]\n" , i, num_neigh_host(i), num_neigh_0_host(i));

          auto internal_and_leaf_nodes_0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bvh_0._internal_and_leaf_nodes);
          auto internal_and_leaf_nodes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bvh._internal_and_leaf_nodes);
          for (int i = 0; i < 2*(int)n-1; ++i) {
              auto &node0 = internal_and_leaf_nodes_0(i);
              auto &node = internal_and_leaf_nodes(i);

              if (node0.left_child != node.left_child || node0.rope != node.rope)
                  printf("[%d]: mismatch in structure: (%d, %d) vs (%d, %d)\n", i, node0.left_child, node0.rope, node.left_child, node.rope);
              else if (node0.bounding_volume != node.bounding_volume) {
                  auto box0 = node0.bounding_volume;
                  auto box = node.bounding_volume;
                  printf("[%d]: mismatch in box: [(%f, %f, %f), (%f, %f, %f)] vs [(%f, %f, %f), (%f, %f, %f)]\n",
                         i,
                         box0.minCorner()[0], box0.minCorner()[1], box0.minCorner()[2],
                         box0.maxCorner()[0], box0.maxCorner()[1], box0.maxCorner()[2],
                         box.minCorner()[0], box.minCorner()[1], box.minCorner()[2],
                         box.maxCorner()[0], box.maxCorner()[1], box.maxCorner()[2]
                         );
              }
          }


          break;
      }
  }

  return 0;
}
