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

#include <ArborX_Ray.hpp> // Vector
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

#include <SOD.hpp>

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 1>
applyPermutation(ExecutionSpace const &exec_space, Permute const &permute,
                 View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    view(i) = view_clone(permute(i));
}

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 2>
applyPermutation2(ExecutionSpace const &exec_space, Permute const &permute,
                  View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto const m = view.extent_int(1);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      view(i, j) = view_clone(permute(i), j);
}

void loadParticlesData(std::string const &filename, InputData &in,
                       int max_num_points = -1)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int N;
  input.read(reinterpret_cast<char *>(&N), sizeof(int));

  int n = N;
  if (max_num_points > 0 && max_num_points < N)
    n = max_num_points;

  Kokkos::resize(Kokkos::WithoutInitializing, in.particles, n);
  Kokkos::resize(Kokkos::WithoutInitializing, in.particle_masses, n);

  for (int d = 0; d < 3; ++d)
  {
    std::vector<float> tmp(n);
    input.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(float));
    input.ignore((N - n) * sizeof(float));

    for (int i = 0; i < n; ++i)
      in.particles(i)[d] = tmp[i];
  }
  input.read(reinterpret_cast<char *>(in.particle_masses.data()),
             n * sizeof(float));
  input.ignore((N - n) * sizeof(float));

  std::cout << "done\nRead in " << n << " particles" << std::endl;

  input.close();
}

void loadHalosData(std::string const &filename, InputData &in, OutputData &out)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_halos;
  input.read(reinterpret_cast<char *>(&num_halos), 4);

  auto read_vector = [&input](auto &v, int n) {
    v.resize(n);
    input.read(reinterpret_cast<char *>(v.data()), n * sizeof(v[0]));
  };
  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };

  read_view(in.fof_halo_tags, num_halos);
  read_view(in.fof_halo_sizes, num_halos);
  read_view(in.fof_halo_masses, num_halos);
  {
    std::vector<float> x, y, z;
    read_vector(x, num_halos);
    read_vector(y, num_halos);
    read_vector(z, num_halos);

    Kokkos::resize(Kokkos::WithoutInitializing, in.fof_halo_centers, num_halos);
    for (int i = 0; i < num_halos; i++)
      in.fof_halo_centers(i) = {x[i], y[i], z[i]};
  }
  read_view(out.sod_halo_masses, num_halos);
  read_view(out.sod_halo_sizes, num_halos);
  read_view(out.sod_halo_rdeltas, num_halos);

  // Filter out
  // - small halos (FOF halo size < 500)
  // - invalid halos (SOD count size = -101)
  auto swap = [](auto &view, int i, int j) { std::swap(view(i), view(j)); };
  int i = 0;
  int num_filtered = 0;
  do
  {
    if (in.fof_halo_sizes(i) < 500 || out.sod_halo_sizes(i) < 0)
    {
      // Instead of using erase(), swap with the last element
      ++num_filtered;

      int j = num_halos - num_filtered;
      if (i < j)
      {
        swap(in.fof_halo_tags, i, j);
        swap(in.fof_halo_sizes, i, j);
        swap(in.fof_halo_masses, i, j);
        swap(in.fof_halo_centers, i, j);
        swap(out.sod_halo_masses, i, j);
        swap(out.sod_halo_sizes, i, j);
        swap(out.sod_halo_rdeltas, i, j);
      }
    }
    else
    {
      ++i;
    }
  } while (i < num_halos - num_filtered);

  if (num_filtered > 0)
  {
    num_halos -= num_filtered;
    Kokkos::resize(in.fof_halo_tags, num_halos);
    Kokkos::resize(in.fof_halo_sizes, num_halos);
    Kokkos::resize(in.fof_halo_masses, num_halos);
    Kokkos::resize(in.fof_halo_centers, num_halos);
    Kokkos::resize(out.sod_halo_masses, num_halos);
    Kokkos::resize(out.sod_halo_sizes, num_halos);
    Kokkos::resize(out.sod_halo_rdeltas, num_halos);
  }

  printf("done\nRead in %d halos [%d total, %d filtered]\n", num_halos,
         num_halos + num_filtered, num_filtered);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, in.fof_halo_tags);
  applyPermutation(host_space, permute, in.fof_halo_sizes);
  applyPermutation(host_space, permute, in.fof_halo_sizes);
  applyPermutation(host_space, permute, in.fof_halo_masses);
  applyPermutation(host_space, permute, in.fof_halo_centers);
  applyPermutation(host_space, permute, out.sod_halo_masses);
  applyPermutation(host_space, permute, out.sod_halo_sizes);
  applyPermutation(host_space, permute, out.sod_halo_rdeltas);

  input.close();
}

void loadProfilesData(std::string const &filename, InputData const &in,
                      OutputData &out)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  // The profile file does not contain first bin with < R_min)
  ARBORX_ASSERT(NUM_BINS == 21);

  int num_records;
  input.read(reinterpret_cast<char *>(&num_records), 4);
  ARBORX_ASSERT(num_records % (NUM_BINS - 1) == 0);

  int num_halos = num_records / (NUM_BINS - 1);

  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };
  auto read_bin_view = [&input](auto &view, int n) {
    using view_type = std::decay_t<decltype(view)>;
    using value_type = typename view_type::value_type;

    Kokkos::View<value_type *, typename view_type::device_type> v(
        Kokkos::ViewAllocateWithoutInitializing("tmp"), n * (NUM_BINS - 1));
    input.read(reinterpret_cast<char *>(v.data()),
               v.extent(0) * sizeof(value_type));

    // First bin is unused, shift data
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    for (int i = 0; i < n; ++i)
    {
      view(i, 0) = -1; // just want a value that is clearly unidentifiable
      for (int j = 0; j < NUM_BINS - 1; ++j)
        view(i, j + 1) = v(i * (NUM_BINS - 1) + j);
    }
  };

  // FOF halo tags are repeated in groups of size NUM_BINS-1, make them unique
  read_view(out.fof_halo_tags, num_records);
  for (int i = 1; i < num_halos; ++i)
    out.fof_halo_tags(i) = out.fof_halo_tags(i * (NUM_BINS - 1));
  Kokkos::resize(out.fof_halo_tags, num_halos);

  read_bin_view(out.sod_halo_bin_ids, num_halos);
  read_bin_view(out.sod_halo_bin_counts, num_halos);
  read_bin_view(out.sod_halo_bin_masses, num_halos);
  read_bin_view(out.sod_halo_bin_outer_radii, num_halos);
  read_bin_view(out.sod_halo_bin_rhos, num_halos);
  read_bin_view(out.sod_halo_bin_rho_ratios, num_halos);
  read_bin_view(out.sod_halo_bin_radial_velocities, num_halos);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, out.fof_halo_tags);
  applyPermutation2(host_space, permute, out.sod_halo_bin_ids);
  applyPermutation2(host_space, permute, out.sod_halo_bin_counts);
  applyPermutation2(host_space, permute, out.sod_halo_bin_masses);
  applyPermutation2(host_space, permute, out.sod_halo_bin_outer_radii);
  applyPermutation2(host_space, permute, out.sod_halo_bin_rhos);
  applyPermutation2(host_space, permute, out.sod_halo_bin_rho_ratios);
  applyPermutation2(host_space, permute, out.sod_halo_bin_radial_velocities);

  printf("done\nRead in %d halos\n", num_halos);

  input.close();

  // Validate tags
  ARBORX_ASSERT(out.fof_halo_tags.extent_int(0) ==
                in.fof_halo_tags.extent_int(0));
  for (int i = 0; i < num_halos; ++i)
    ARBORX_ASSERT(in.fof_halo_tags(i) == out.fof_halo_tags(i));
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  int max_num_points;
  std::string filename_particles;
  std::string filename_halos;
  std::string filename_profiles;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "filename-particles", bpo::value<std::string>(&filename_particles), "filename containing particles data" )
      ( "filename-halos", bpo::value<std::string>(&filename_halos), "filename containing halos data" )
      ( "filename-profiles", bpo::value<std::string>(&filename_profiles), "filename containing profiles data" )
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

  // print out the runtime parameters
  printf("filename [particles] : %s [max_pts = %d]\n",
         filename_particles.c_str(), max_num_points);
  printf("filename [halos]     : %s\n", filename_halos.c_str());
  printf("filename [profiles]  : %s\n", filename_profiles.c_str());

  // read in data
  InputData input_data;
  OutputData validation_data;
  loadParticlesData(filename_particles, input_data, max_num_points);
  loadHalosData(filename_halos, input_data, validation_data);
  loadProfilesData(filename_profiles, input_data, validation_data);

  // run SOD
  OutputData output_data;
  sod(ExecutionSpace{}, input_data, output_data);

  // validate
  auto const num_halos = input_data.fof_halo_tags.extent_int(0);
#if 0
  // outer radii
  printf("validating radii\n");
  for (int i = 0; i < num_halos; ++i)
  {
    bool matched = true;
    for (int j = 1; j < NUM_BINS; ++j)
      matched &= (output_data.sod_halo_bin_outer_radii(i, j) ==
                  validation_data.sod_halo_bin_outer_radii(i, j));
    if (!matched)
    {
      printf("radii for halo tag %ld do not match: relative errors [",
             input_data.fof_halo_tags(i));
      for (int j = 1; j < NUM_BINS; ++j)
      {
        float a = output_data.sod_halo_bin_outer_radii(i, j);
        float b = validation_data.sod_halo_bin_outer_radii(i, j);
        printf(" %e", std::abs((a - b) / a));
      }
      printf(" ]\n");
    }
  }
#endif

  // bin counts
  printf("validating bin counts\n");
  for (int i = 0; i < num_halos; ++i)
  {
    bool matched = true;
    for (int j = 1; j < NUM_BINS; ++j)
      matched &= (output_data.sod_halo_bin_counts(i, j) ==
                  validation_data.sod_halo_bin_counts(i, j));
    if (!matched)
    {
      printf("counts for halo tag %ld do not match: [",
             input_data.fof_halo_tags(i));
      printf("%ld: result = [", input_data.fof_halo_tags(i));
      for (int j = 1; j < NUM_BINS; ++j)
        printf(" %d", output_data.sod_halo_bin_counts(i, j));
      printf(" ], validation = [");
      for (int j = 1; j < NUM_BINS; ++j)
        printf(" %d", validation_data.sod_halo_bin_counts(i, j));
      printf(" ]\n");
    }
  }

  // bin masses
  printf("validating bin masses\n");
  for (int i = 0; i < num_halos; ++i)
  {
    bool matched = true;
    for (int j = 1; j < NUM_BINS; ++j)
      matched &= (output_data.sod_halo_bin_masses(i, j) ==
                  validation_data.sod_halo_bin_masses(i, j));
    if (!matched)
    {
      printf("masses for halo tag %ld do not match: relative errors [",
             input_data.fof_halo_tags(i));
      for (int j = 1; j < NUM_BINS; ++j)
      {
        float a = output_data.sod_halo_bin_masses(i, j);
        float b = validation_data.sod_halo_bin_masses(i, j);
        printf(" %e", std::abs((a - b) / a));
      }
      printf(" ]\n");
    }
  }

  return EXIT_SUCCESS;
}
