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
  std::vector<ArborX::Point> particles;
  std::vector<float> particle_masses;

  std::vector<int64_t> fof_halo_tags;
  std::vector<int> fof_halo_sizes;
  std::vector<float> fof_halo_masses;
  std::vector<ArborX::Point> fof_halo_centers;
  std::vector<float> sod_halo_masses;
  std::vector<int64_t> sod_halo_sizes;
  std::vector<float> sod_halo_rdeltas;
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

  data.particles.resize(n);
  data.particle_masses.resize(n);

  for (int d = 0; d < 3; ++d)
  {
    std::vector<float> tmp(n);
    input.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(float));
    input.ignore((N - n) * sizeof(float));

    for (int i = 0; i < n; ++i)
      data.particles[i][d] = tmp[i];
  }
  input.read(reinterpret_cast<char *>(data.particle_masses.data()),
             n * sizeof(float));
  input.ignore((N - n) * sizeof(float));

#if 0
  for (int i = 0; i < n; ++i)
    printf("%d: (%f, %f, %f), %f\n", i, data.particles[i][0],
           data.particles[i][1], data.particles[i][2], data.particle_masses[i]);
#endif

  std::cout << "done\nRead in " << n << " particles" << std::endl;

  input.close();
}

#define READ_ARRAY(array, n)                                                   \
  array.resize((n));                                                           \
  input.read(reinterpret_cast<char *>(array.data()), (n) * sizeof(array[0]))

void loadHalosData(std::string const &filename, Data &data)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_halos;
  input.read(reinterpret_cast<char *>(&num_halos), 4);

  READ_ARRAY(data.fof_halo_tags, num_halos);
  READ_ARRAY(data.fof_halo_sizes, num_halos);
  READ_ARRAY(data.fof_halo_masses, num_halos);
  {
    std::vector<float> x, y, z;
    READ_ARRAY(x, num_halos);
    READ_ARRAY(y, num_halos);
    READ_ARRAY(z, num_halos);

    data.fof_halo_centers.resize(num_halos);
    for (int i = 0; i < num_halos; i++)
      data.fof_halo_centers[i] = {x[i], y[i], z[i]};
  }
  READ_ARRAY(data.sod_halo_masses, num_halos);
  READ_ARRAY(data.sod_halo_sizes, num_halos);
  READ_ARRAY(data.sod_halo_rdeltas, num_halos);

  // Filter out
  // - small halos (FOF halo size < 500)
  // - invalid halos (SOD count size = -101)
  auto SWAP = [](auto &v, int i, int j) { std::swap(v[i], v[j]); };
  int i = 0;
  int num_filtered = 0;
  do
  {
    if (data.fof_halo_sizes[i] < 500 || data.sod_halo_sizes[i] < 0)
    {
      // printf("Filtering out halo tag %ld: fof size = %d, sod size = %ld\n",
      // data.fof_halo_tags[i], data.fof_halo_sizes[i],
      // data.sod_halo_sizes[i]);

      // Instead of using erase(), swap with the last element
      ++num_filtered;

      int j = num_halos - num_filtered;
      if (i < j)
      {
        SWAP(data.fof_halo_tags, i, j);
        SWAP(data.fof_halo_sizes, i, j);
        SWAP(data.fof_halo_masses, i, j);
        SWAP(data.fof_halo_centers, i, j);
        SWAP(data.sod_halo_masses, i, j);
        SWAP(data.sod_halo_sizes, i, j);
        SWAP(data.sod_halo_rdeltas, i, j);
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
    data.fof_halo_tags.resize(num_halos);
    data.fof_halo_sizes.resize(num_halos);
    data.fof_halo_masses.resize(num_halos);
    data.fof_halo_centers.resize(num_halos);
    data.sod_halo_masses.resize(num_halos);
    data.sod_halo_sizes.resize(num_halos);
    data.sod_halo_rdeltas.resize(num_halos);
  }

  printf("done\nRead in %d halos [%d total, %d filtered]\n", num_halos,
         num_halos + num_filtered, num_filtered);

  input.close();
}
#undef READ_FIELD

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

  auto const particles = vec2view<MemorySpace>(data.particles, "particles");
  auto const particle_masses =
      vec2view<MemorySpace>(data.particle_masses, "particle_masses");
  auto const fof_halo_centers =
      vec2view<MemorySpace>(data.fof_halo_centers, "fof_halo_centers");
  auto const fof_halo_masses =
      vec2view<MemorySpace>(data.fof_halo_masses, "fof_halo_masses");

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
        // printf("r_max(%d) = %f\n", i, r_max(i));
      });

  sod(ExecutionSpace{}, particles, particle_masses, fof_halo_centers, r_min,
      r_max, rho_c);

  return EXIT_SUCCESS;
}
