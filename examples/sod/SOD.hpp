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

template <typename Points>
struct PointsWrapper
{
  Points _points;
};

template <typename Points>
struct ArborX::AccessTraits<PointsWrapper<Points>, ArborX::PredicatesTag>
{
  using PointsAccess = AccessTraits<Points, PrimitivesTag>;

  using memory_space = typename PointsAccess::memory_space;
  using Predicates = PointsWrapper<Points>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return PointsAccess::size(w._points);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(PointsAccess::get(w._points, i)), (int)i);
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

template <typename MemorySpace, typename Points>
struct BinAccumulator
{
  Points _points;
  Kokkos::View<float *, MemorySpace> _masses;
  Kokkos::View<float * [NUM_BINS], MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<float * [NUM_BINS], MemorySpace> _sod_halo_bin_counts;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  using PointsAccess = ArborX::AccessTraits<Points, ArborX::PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);
    ArborX::Point const &point = PointsAccess::get(_points, particle_index);

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

template <typename MemorySpace, typename Points>
struct OverlapCount
{
  Points _points;
  Kokkos::View<int *, MemorySpace> _counts;
  Kokkos::View<ArborX::Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;

  using PointsAccess = ArborX::AccessTraits<Points, ArborX::PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int j) const
  {
    auto i = getData(query);

    ArborX::Point const &point = PointsAccess::get(_points, i);
    if (ArborX::Details::distance(point, _centers(j)) <= _radii(j))
      ++_counts(i);
  }
};

template <typename ExecutionSpace, typename Points, typename MemorySpace>
void sod(ExecutionSpace const &exec_space, Points points,
         Kokkos::View<float *, MemorySpace> masses,
         Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers,
         float r_min, Kokkos::View<float *, MemorySpace> r_max, float rho_c)
{
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

  int const n = points.extent(0);
  int const num_halos = fof_halo_centers.extent(0);

  // Do not sort for now, so as to not allocate additional memory, which would
  // take 8*n bytes (4 for Morton index, 4 for permutation index)
  bool const sort_predicates = false;

  // Step 1: construct the search index based on spheres (FOF centers with
  // R_max)
  timer_start(timer);
  ArborX::BVH<MemorySpace> bvh(exec_space,
                               Spheres<MemorySpace>{fof_halo_centers, r_max});
  elapsed["construction"] = timer_seconds(timer);

  // Step 2: compute mass profiles
  timer_start(timer);
  Kokkos::View<float * [NUM_BINS], MemorySpace> sod_halo_bin_masses(
      "sod_halo_bin_masses", num_halos);
  Kokkos::View<float * [NUM_BINS], MemorySpace> sod_halo_bin_counts(
      "sod_halo_bin_counts", num_halos);
  bvh.query(exec_space, PointsWrapper<Points>{points},
            BinAccumulator<MemorySpace, Points>{
                points, masses, sod_halo_bin_masses, sod_halo_bin_counts,
                fof_halo_centers, r_min, r_max},
            ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                sort_predicates));
  elapsed["binning"] = timer_seconds(timer);

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
  Kokkos::View<int *, MemorySpace> counts("counts", n);
  bvh.query(exec_space, PointsWrapper<Points>{points},
            OverlapCount<MemorySpace, Points>{points, counts, fof_halo_centers,
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
  printf("  #points inside spheres: %d/%d [%.2f]\n", num_inside, n,
         (100.f * num_inside) / n);

  int num_inside_multiple = 0;
  Kokkos::parallel_reduce("b",
                          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                          KOKKOS_LAMBDA(int i, int &update) {
                            if (counts(i) > 1)
                              ++update;
                          },
                          num_inside_multiple);
  printf("  #points in multiple: %d\n", num_inside_multiple);

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
}

#endif
