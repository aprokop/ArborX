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

struct Data
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;

  View<ArborX::Point> particles;
  View<float> particle_masses;

  View<int64_t> fof_halo_tags;
  View<int> fof_halo_sizes;
  View<float> fof_halo_masses;
  View<ArborX::Point> fof_halo_centers;
  View<float> sod_halo_masses;
  View<int64_t> sod_halo_sizes;
  View<float> sod_halo_rdeltas;

  Data()
      : particles("particles", 0)
      , particle_masses("particle_masses", 0)
      , fof_halo_tags("fof_halo_tags", 0)
      , fof_halo_sizes("fof_halo_sizes", 0)
      , fof_halo_masses("fof_halo_masses", 0)
      , fof_halo_centers("fof_halo_centers", 0)
      , sod_halo_masses("sod_halo_masses", 0)
      , sod_halo_sizes("sod_halo_sizes", 0)
      , sod_halo_rdeltas("sod_halo_rdeltas", 0)
  {
  }
};

struct ProfilesData
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;
  template <typename T>
  using BinView = Kokkos::View<T * [NUM_BINS - 1], Kokkos::HostSpace>;

  View<int64_t> fof_halo_tags;
  BinView<int> sod_halo_bin_ids;
  BinView<int> sod_halo_bin_counts;
  BinView<float> sod_halo_bin_masses;
  BinView<float> sod_halo_bin_outer_radii;
  BinView<float> sod_halo_bin_rhos;
  BinView<float> sod_halo_bin_rho_ratios;
  BinView<float> sod_halo_bin_radial_velocities;

  ProfilesData()
      : fof_halo_tags("fof_halo_tags", 0)
      , sod_halo_bin_ids("sod_halo_bin_ids", 0)
      , sod_halo_bin_counts("sod_halo_bin_counts", 0)
      , sod_halo_bin_masses("sod_halo_bin_masses", 0)
      , sod_halo_bin_outer_radii("sod_halo_bin_outer_radii", 0)
      , sod_halo_bin_rhos("sod_halo_bin_rhos", 0)
      , sod_halo_bin_rho_ratios("sod_halo_bin_rho_ratios", 0)
      , sod_halo_bin_radial_velocities("sod_hlo_bin_radial_velocities", 0)
  {
  }
};

void loadParticlesData(std::string const &filename, Data &data,
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

  Kokkos::resize(Kokkos::WithoutInitializing, data.particles, n);
  Kokkos::resize(Kokkos::WithoutInitializing, data.particle_masses, n);

  for (int d = 0; d < 3; ++d)
  {
    std::vector<float> tmp(n);
    input.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(float));
    input.ignore((N - n) * sizeof(float));

    for (int i = 0; i < n; ++i)
      data.particles(i)[d] = tmp[i];
  }
  input.read(reinterpret_cast<char *>(data.particle_masses.data()),
             n * sizeof(float));
  input.ignore((N - n) * sizeof(float));

  std::cout << "done\nRead in " << n << " particles" << std::endl;

  input.close();
}

void loadHalosData(std::string const &filename, Data &data)
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

  read_view(data.fof_halo_tags, num_halos);
  read_view(data.fof_halo_sizes, num_halos);
  read_view(data.fof_halo_masses, num_halos);
  {
    std::vector<float> x, y, z;
    read_vector(x, num_halos);
    read_vector(y, num_halos);
    read_vector(z, num_halos);

    Kokkos::resize(Kokkos::WithoutInitializing, data.fof_halo_centers,
                   num_halos);
    for (int i = 0; i < num_halos; i++)
      data.fof_halo_centers(i) = {x[i], y[i], z[i]};
  }
  read_view(data.sod_halo_masses, num_halos);
  read_view(data.sod_halo_sizes, num_halos);
  read_view(data.sod_halo_rdeltas, num_halos);

  // Filter out
  // - small halos (FOF halo size < 500)
  // - invalid halos (SOD count size = -101)
  auto swap = [](auto &view, int i, int j) { std::swap(view(i), view(j)); };
  int i = 0;
  int num_filtered = 0;
  do
  {
    if (data.fof_halo_sizes(i) < 500 || data.sod_halo_sizes(i) < 0)
    {
      // printf("Filtering out halo tag %ld: fof size = %d, sod size = %ld\n",
      // data.fof_halo_tags[i], data.fof_halo_sizes[i],
      // data.sod_halo_sizes[i]);

      // Instead of using erase(), swap with the last element
      ++num_filtered;

      int j = num_halos - num_filtered;
      if (i < j)
      {
        swap(data.fof_halo_tags, i, j);
        swap(data.fof_halo_sizes, i, j);
        swap(data.fof_halo_masses, i, j);
        swap(data.fof_halo_centers, i, j);
        swap(data.sod_halo_masses, i, j);
        swap(data.sod_halo_sizes, i, j);
        swap(data.sod_halo_rdeltas, i, j);
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
    Kokkos::resize(data.fof_halo_tags, num_halos);
    Kokkos::resize(data.fof_halo_sizes, num_halos);
    Kokkos::resize(data.fof_halo_masses, num_halos);
    Kokkos::resize(data.fof_halo_centers, num_halos);
    Kokkos::resize(data.sod_halo_masses, num_halos);
    Kokkos::resize(data.sod_halo_sizes, num_halos);
    Kokkos::resize(data.sod_halo_rdeltas, num_halos);
  }

  printf("done\nRead in %d halos [%d total, %d filtered]\n", num_halos,
         num_halos + num_filtered, num_filtered);

  input.close();
}

void loadProfilesData(std::string const &filename, ProfilesData &data)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  // The profile file does not contain first bin with < R_min)
  int constexpr num_bins_in_input = 20;
  assert(num_bins_in_input == NUM_BINS - 1);

  int num_records;
  input.read(reinterpret_cast<char *>(&num_records), 4);
  int num_halos = num_records / num_bins_in_input;

  auto read_view = [&input](auto &view, int n, int s = 1) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n / s);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };

  read_view(data.fof_halo_tags, num_records);
  for (int i = 1; i < num_halos; ++i)
    data.fof_halo_tags(i) = data.fof_halo_tags(i * num_bins_in_input);
  Kokkos::resize(data.fof_halo_tags, num_halos);

  read_view(data.sod_halo_bin_ids, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_counts, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_masses, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_outer_radii, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_rhos, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_rho_ratios, num_records, num_bins_in_input);
  read_view(data.sod_halo_bin_radial_velocities, num_records,
            num_bins_in_input);

  printf("done\nRead in %d halos\n", num_halos);

  input.close();
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

  // Print out the runtime parameters
  printf("filename [particles] : %s [max_pts = %d]\n",
         filename_particles.c_str(), max_num_points);
  printf("filename [halos]     : %s\n", filename_halos.c_str());
  printf("filename [profiles]  : %s\n", filename_profiles.c_str());

  // read in data
  Data data;
  loadParticlesData(filename_particles, data, max_num_points);
  loadHalosData(filename_halos, data);

  ProfilesData profiles_data;
  loadProfilesData(filename_profiles, profiles_data);

  ExecutionSpace exec_space;

  auto const particles =
      Kokkos::create_mirror_view_and_copy(exec_space, data.particles);
  auto const particle_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, data.particle_masses);
  auto const fof_halo_centers =
      Kokkos::create_mirror_view_and_copy(exec_space, data.fof_halo_centers);
  auto const fof_halo_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, data.fof_halo_masses);

  auto const num_halos = fof_halo_centers.extent(0);

  // HACC constants
  float constexpr MIN_FACTOR = 0.05;
  float constexpr MAX_FACTOR = 2.0;
  float constexpr R_SMOOTH =
      250.f / 3072; // interparticle separation, rl/np, where rl is the boxsize
                    // of the simulation, and np is the number of particles
  float constexpr SOD_MASS = 1e14;
  float constexpr RHO_C = 2.77536627e11;

  // rho_c = RHO_C * Efact*Efact * a*a
  // At redshift = 0, the factors are trivial:
  //   a = 1, Efact = 1,
  // so rho_c = RHO_C.
  float rho_c = RHO_C;

  // Compute r_min and r_max
  float r_min = MIN_FACTOR * R_SMOOTH;
  Kokkos::View<float *, MemorySpace> r_max("r_max", fof_halo_centers.extent(0));
  Kokkos::parallel_for(
      "compute_r_max",
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        float R_init = std::cbrt(fof_halo_masses(i) / SOD_MASS);
        r_max(i) = MAX_FACTOR * R_init;
      });

  sod(ExecutionSpace{}, particles, particle_masses, fof_halo_centers, r_min,
      r_max, rho_c);

  return EXIT_SUCCESS;
}
