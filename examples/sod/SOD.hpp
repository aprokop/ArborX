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

#ifndef SOD_HPP
#define SOD_HPP

#include <ArborX.hpp>

int constexpr NUM_BINS = 20 + 1;

struct InputData
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;

  View<ArborX::Point> particles;
  View<float> particle_masses;

  View<int64_t> fof_halo_tags;
  View<int> fof_halo_sizes;
  View<float> fof_halo_masses;
  View<ArborX::Point> fof_halo_centers;

  InputData()
      : particles("particles", 0)
      , particle_masses("particle_masses", 0)
      , fof_halo_tags("fof_halo_tags", 0)
      , fof_halo_sizes("fof_halo_sizes", 0)
      , fof_halo_masses("fof_halo_masses", 0)
      , fof_halo_centers("fof_halo_centers", 0)
  {
  }
};

struct OutputData
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;
  template <typename T>
  using BinView = Kokkos::View<T * [NUM_BINS], Kokkos::HostSpace>;

  View<int64_t> fof_halo_tags;

  View<float> sod_halo_masses;
  View<int64_t> sod_halo_sizes;
  View<float> sod_halo_rdeltas;

  BinView<int> sod_halo_bin_ids;
  BinView<int> sod_halo_bin_counts;
  BinView<float> sod_halo_bin_masses;
  BinView<float> sod_halo_bin_outer_radii;
  BinView<float> sod_halo_bin_rhos;
  BinView<float> sod_halo_bin_rho_ratios;
  BinView<float> sod_halo_bin_radial_velocities;

  OutputData()
      : fof_halo_tags("fof_halo_tags", 0)
      , sod_halo_masses("sod_halo_masses", 0)
      , sod_halo_sizes("sod_halo_sizes", 0)
      , sod_halo_rdeltas("sod_halo_rdeltas", 0)
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

template <typename MemorySpace>
struct Spheres
{
  Kokkos::View<ArborX::Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Spheres<MemorySpace>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t size(const Spheres<MemorySpace> &spheres)
  {
    return spheres._centers.extent(0);
  }
  KOKKOS_FUNCTION static ArborX::Box get(Spheres<MemorySpace> const &spheres,
                                         std::size_t const i)
  {
    auto const &c = spheres._centers(i);
    auto const r = spheres._radii(i);
    return {{c[0] - r, c[1] - r, c[2] - r}, {c[0] + r, c[1] + r, c[2] + r}};
  }
};

template <typename Particles>
struct ParticlesWrapper
{
  Particles _particles;
};

template <typename Particles>
struct ArborX::AccessTraits<ParticlesWrapper<Particles>, ArborX::PredicatesTag>
{
  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  using memory_space = typename ParticlesAccess::memory_space;
  using Predicates = ParticlesWrapper<Particles>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return ParticlesAccess::size(w._particles);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(ParticlesAccess::get(w._particles, i)), (int)i);
  }
};

KOKKOS_FUNCTION
inline float rDelta(float r_min, float r_max)
{
  return log10(r_max / r_min) / (NUM_BINS - 1);
}

KOKKOS_FUNCTION
inline int binID(float r_min, float r_max, float r)
{

  float r_delta = rDelta(r_min, r_max);

  int bin_id = 0;
  if (r > r_min)
    bin_id = (int)floor(log10(r / r_min) / r_delta) + 1;
  if (bin_id >= NUM_BINS)
    bin_id = NUM_BINS - 1;

  return bin_id;
}

template <typename MemorySpace, typename Particles>
struct BinAccumulator
{
  Particles _particles;
  Kokkos::View<float *, MemorySpace> _masses;
  Kokkos::View<float * [NUM_BINS], MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<int * [NUM_BINS], MemorySpace> _sod_halo_bin_counts;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  using ParticlesAccess =
      ArborX::AccessTraits<Particles, ArborX::PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);
    ArborX::Point const &point =
        ParticlesAccess::get(_particles, particle_index);

    float dist =
        ArborX::Details::distance(point, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    int bin_id = binID(_r_min, _r_max(halo_index), dist);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_counts(halo_index, bin_id), 1);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_masses(halo_index, bin_id),
                             _masses(particle_index));
  }
};

template <typename MemorySpace, typename Particles>
struct OverlapCount
{
  Particles _particles;
  Kokkos::View<int *, MemorySpace> _counts;
  Kokkos::View<ArborX::Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;

  using ParticlesAccess =
      ArborX::AccessTraits<Particles, ArborX::PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int j) const
  {
    auto i = getData(query);

    ArborX::Point const &point = ParticlesAccess::get(_particles, i);
    if (ArborX::Details::distance(point, _centers(j)) <= _radii(j))
      ++_counts(i);
  }
};

template <typename ExecutionSpace>
void sod(ExecutionSpace const &exec_space, InputData const &in, OutputData &out)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::HostSpace host_space;

  Kokkos::Timer timer_total;
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

  auto const num_halos = in.fof_halo_tags.extent_int(0);

  auto const particles =
      Kokkos::create_mirror_view_and_copy(exec_space, in.particles);
  auto const particle_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, in.particle_masses);
  auto const fof_halo_centers =
      Kokkos::create_mirror_view_and_copy(exec_space, in.fof_halo_centers);
  auto const fof_halo_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, in.fof_halo_masses);

  using Particles = decltype(particles);

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
  // float rho_c = RHO_C;

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

  // Do not sort for now, so as to not allocate additional memory, which would
  // take 8*n bytes (4 for Morton index, 4 for permutation index)
  bool const sort_predicates = false;

  // Step 1: construct the search index based on spheres (FOF centers with
  // R_max)
  timer_start(timer);
  ArborX::BVH<MemorySpace> bvh(exec_space,
                               Spheres<MemorySpace>{fof_halo_centers, r_max});
  elapsed["construction"] = timer_seconds(timer);

  // Store outer radii
  Kokkos::View<float * [NUM_BINS], MemorySpace> sod_halo_bin_outer_radii(
      "sod_halo_bin_outer_radii", num_halos);
  Kokkos::parallel_for(
      "recompute_outer_radii",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        float r_delta = rDelta(r_min, r_max(halo_index));
        sod_halo_bin_outer_radii(halo_index, 0) = r_min;
        for (int bin_id = 1; bin_id < NUM_BINS; ++bin_id)
          sod_halo_bin_outer_radii(halo_index, bin_id) =
              pow(10.0, (bin_id * r_delta)) * r_min;
      });
  auto sod_halo_bin_outer_radii_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_outer_radii);
  Kokkos::resize(out.sod_halo_bin_outer_radii, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_outer_radii,
                    sod_halo_bin_outer_radii_host);

  // Step 2: compute mass and count profiles
  timer_start(timer);
  Kokkos::View<float * [NUM_BINS], MemorySpace> sod_halo_bin_masses(
      "sod_halo_bin_masses", num_halos);
  Kokkos::View<int * [NUM_BINS], MemorySpace> sod_halo_bin_counts(
      "sod_halo_bin_counts", num_halos);
  bvh.query(exec_space, ParticlesWrapper<Particles>{particles},
            BinAccumulator<MemorySpace, Particles>{
                particles, particle_masses, sod_halo_bin_masses,
                sod_halo_bin_counts, fof_halo_centers, r_min, r_max},
            ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                sort_predicates));
  auto sod_halo_bin_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_masses);
  auto sod_halo_bin_counts_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_counts);
  Kokkos::resize(out.sod_halo_bin_masses, num_halos);
  Kokkos::resize(out.sod_halo_bin_counts, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_masses, sod_halo_bin_masses_host);
  Kokkos::deep_copy(out.sod_halo_bin_counts, sod_halo_bin_counts_host);
  elapsed["binning"] = timer_seconds(timer);

#if 0
  // Step 3: recompute R_max based on sod_halo_bin_masses
  float const DELTA = 200;
  Kokkos::parallel_for(
      "recompute_R_max",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        float r_delta = rDelta(r_min, r_max(halo_index));

        float accumulated_mass = sod_halo_bin_masses(halo_index, 0);
        for (int bin_id = 1; bin_id < NUM_BINS; ++bin_id)
        {
          accumulated_mass += sod_halo_bin_masses(halo_index, bin_id);

          float bin_inner_radius = pow(10.0, ((bin_id - 1) * r_delta)) * r_min;
          float volume_inner = 4.f / 3 * M_PI * pow(bin_inner_radius, 3);

          float density_upper_bound = accumulated_mass / rho_c / volume_inner;
          if (density_upper_bound < DELTA)
          {
#if 0
            float bin_outer_radius = pow(10.0, (bin_id * r_delta)) * r_min;
            float volume_outer = 4.f / 3 * M_PI * pow(bin_outer_radius, 3);
            float density_lower_bound = accumulated_mass / rho_c / volume_outer;
            int critical_bin_id = bin_id - 1;
            printf(
                "[%d]: critical bin %d (next bin's density is in [%f, %f])\n",
                halo_index, critical_bin_id, density_lower_bound,
                density_upper_bound);
#endif
            r_max(halo_index) = bin_inner_radius;
            break;
          }
        }
      });

  // Step 4: compute overlap counts
  timer_start(timer);
  auto const n = in.particles.extent_int(0);
  Kokkos::View<int *, MemorySpace> counts("counts", n);
  bvh.query(exec_space, ParticlesWrapper<Particles>{particles},
            OverlapCount<MemorySpace, Particles>{particles, counts, fof_halo_centers,
                                              r_max},
            ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                sort_predicates));
  elapsed["query"] = timer_seconds(timer);

  printf("-- construction     : %10.3f\n", elapsed["construction"]);
  printf("-- binning          : %10.3f\n", elapsed["binning"]);
  printf("-- query            : %10.3f\n", elapsed["query"]);

  // Compute some statistics
  printf("Stats:\n");

  int num_inside = 0;
  Kokkos::parallel_reduce("a",
                          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                          KOKKOS_LAMBDA(int i, int &update) {
                            if (counts(i) > 0)
                              ++update;
                          },
                          num_inside);
  printf("  #particles inside spheres: %d/%d [%.2f]\n", num_inside, n,
         (100.f * num_inside) / n);

  int num_inside_multiple = 0;
  Kokkos::parallel_reduce("b",
                          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                          KOKKOS_LAMBDA(int i, int &update) {
                            if (counts(i) > 1)
                              ++update;
                          },
                          num_inside_multiple);
  printf("  #particles in multiple: %d\n", num_inside_multiple);

  if (num_inside_multiple > 0)
  {
    int max_multiple = 0;
    Kokkos::parallel_reduce(
        "c", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          if (counts(i) > update)
            update = counts(i);
        },
        Kokkos::Max<int>(max_multiple));
    if (num_inside_multiple > 0)
      printf("  max number of owners: %d\n", max_multiple);
  }
#endif
}

#endif
