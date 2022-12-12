/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*
 * This example demonstrates how to use ArborX for a raytracing example where
 * rays are used to model thermal radiation in a participating medium
 * (e.g. a flame). Similar to example_raytracing.cpp, rays carry energy
 * and are traced through boxes. But unlike in example_raytracing.cpp,
 * the rays here are traced according to the Backwards Monte-Carlo approach,
 * where the rays accumulate intensity as they travel which they then deposit
 * into their originating box. This is done using a distributed approach,
 * where accumulated ray intensities are gathered for each MPI rank that ray
 * intersects.
 */

#include <ArborX.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <numeric>

#include <mpi.h>

// The total energy that is distributed across all rays.
constexpr float temp = 2000.f;   // medium temperature [Kelvin]
constexpr float sigma = 5.67e-8; // stefan-boltzmann constant [W/m^2K]
constexpr float pi = Kokkos::Experimental::pi_v<float>;
constexpr float kappa = 10.f; // radiative absorption coefficient [1/m]

KOKKOS_INLINE_FUNCTION float sigmaT4overPi()
{
#if KOKKOS_VERSION >= 30700
  using Kokkos::pow;
#else
  using Kokkos::Experimental::pow;
#endif
  return sigma * pow(temp, 4.f) / pi;
}

namespace MPIbased
{

template <typename MemorySpace>
struct Rays
{
  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> _rays;
};

/*
 * IntersectedRank applies for all
 * intersections between rays and overall MPI ranks that are detected when
 * calling AccumulateRayRankIntersectionData struct. The member variables that
 * are relevant for sorting the intersection according to rank and ray are
 * contained in the base class IntersectedRankForSorting as performance
 * improvement.
 */
struct IntersectedRankForSorting
{
  float entrylength;
  int ray_id;
  friend KOKKOS_FUNCTION bool operator<(IntersectedRankForSorting const &l,
                                        IntersectedRankForSorting const &r)
  {
    if (l.ray_id == r.ray_id)
      return l.entrylength < r.entrylength;
    return l.ray_id < r.ray_id;
  }
};

struct IntersectedRank : public IntersectedRankForSorting
{
  float optical_path_length;    // optical distance through rank
  float intensity_contribution; // contribution of rank to ray intensity
  KOKKOS_FUNCTION IntersectedRank() = default;
  KOKKOS_FUNCTION IntersectedRank(float entry_length, float path_length,
                                  float rank_intensity_contribution,
                                  int predicate_index)
      : IntersectedRankForSorting{entry_length, predicate_index}
      , optical_path_length(path_length)
      , intensity_contribution(rank_intensity_contribution)
  {}
};

struct RayDataAccumulator
{
  float optical_path_length;
  float intensity_contribution;
  float rank_entry_length;
};

/*
 *  Callback for storing two accumated values for each ray/rank
 *  intersection:
 *    1. Accumulated optical distance (length * kappa)
 *    2. Intensity contribution of the rank to the ray
 *
 */
template <typename MemorySpace>
struct AccumulateRayRankIntersectionData
{
  using tag = ArborX::Details::PostCallbackTag;
  Kokkos::View<ArborX::Box *, MemorySpace> _boxes;
  int rank;

  /*
   * Callback to accumulate optical distance (kappa*length) and intensity
   * contribution from boxes.
   */
  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate &predicate,
                                  int const primitive_index) const
  {

#if KOKKOS_VERSION >= 30700
    using Kokkos::exp;
#else
    using Kokkos::Experimental::exp;
#endif

    float length;
    float entrylength;
    auto const &ray = ArborX::getGeometry(predicate);
    auto &accumulated_data = ArborX::getData(predicate);
    auto const &box = _boxes(primitive_index);

    overlapDistance(ray, box, length, entrylength);
    float const optical_path_length = kappa * length;
    float const optical_path_length_in = accumulated_data.optical_path_length;

    accumulated_data.optical_path_length += optical_path_length;
    accumulated_data.intensity_contribution +=
        sigmaT4overPi() * (exp(optical_path_length_in) -
                         exp(accumulated_data.optical_path_length));
    accumulated_data.rank_entry_length =
        accumulated_data.rank_entry_length > entrylength
            ? entrylength : accumulated_data.rank_entry_length;
  }

  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView &in,
                  OutView &out) const
  {
    auto const n = offset.extent(0) - 1;
    auto const num_rays = queries.extent(0);
    auto const num_intersections = in.extent(0);
    Kokkos::realloc(out, n); // one for each ray
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;

    // Accumulating two ouputted values for this rank
    Kokkos::parallel_for(
        "Evaluating ray-box interaction", num_rays, KOKKOS_LAMBDA(int i) {

          auto const &ray = ArborX::getGeometry(queries(i));
          auto const &accumulated_data = ArborX::getData(queries(i));

          // Rank output data structure
          out(i) = IntersectedRank{
              /*entrylength*/ accumulated_data.rank_entry_length,
              /*optical_path_length*/ accumulated_data.optical_path_length,
              /*intensity contr*/ accumulated_data.intensity_contribution,
              /*ray_id*/ 0}; // ray ID on originating rank is tracked by
                             // distributedTree and will be applied later for
                             // sorting

          // offset values reset to reflect only one output per
          // ray/rank intersection
          // required for communicateResultsBack to work
          offset(i) = i;
          if (i == n - 1)
            offset(n) = n; // set last value as well
        });
  }
};

} // namespace MPIbased

template <typename MemorySpace>
struct ArborX::AccessTraits<MPIbased::Rays<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = std::size_t;

  KOKKOS_FUNCTION
  static size_type size(MPIbased::Rays<MemorySpace> const &rays)
  {
    return rays._rays.extent(0);
  }
  KOKKOS_FUNCTION
  static auto get(MPIbased::Rays<MemorySpace> const &rays, size_type i)
  {
    //return attach(ordered_intersects(rays._rays(i)), (int)i);
    return attach(ordered_intersects(rays._rays(i)), MPIbased::RayDataAccumulator{0.f,0.f,0.f});
  }
};

template <typename View>
void printoutput(View &energies, float const dx, float const dy, float const dz)
{
  for (int k = 0; k < energies.extent(2); ++k)
  {
    for (int j = 0; j < energies.extent(1); ++j)
    {
      for (int i = 0; i < energies.extent(0); ++i)
      {
        int bid = i + j * energies.extent(0) +
                  k * energies.extent(0) * energies.extent(1);
        printf("%10d %20.5f %20.5f %20.5f %20.5f\n", bid,
               (dx * (float)i + dx * ((float)i + 1.f)) / 2.f,
               (dy * (float)j + dy * ((float)j + 1.f)) / 2.f,
               (dz * (float)k + dz * ((float)k + 1.f)) / 2.f,
               energies(i, j, k));
      }
    }
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank, num_ranks;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &num_ranks);
  if (comm_rank == 0)
  {
    std::cout << "ArborX version: " << ArborX::version() << std::endl;
    std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version: " << KokkosExt::version() << std::endl;
  }

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiply.
  if (comm_rank != 0)
  {
    auto *help_it = std::find_if(argv, argv + argc, [](std::string const &x) {
      return x == "--help" || x == "--kokkos-help";
    });
    if (help_it != argv + argc)
    {
      std::swap(*help_it, *(argv + argc - 1));
      --argc;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostSpace = HostExecutionSpace::memory_space;

    namespace bpo = boost::program_options;

    int nx;
    int ny;
    int nz;
    int nx_mpi;
    int ny_mpi;
    int nz_mpi;
    int rays_per_box;
    float lx;
    float ly;
    float lz;
    bool print;

    bpo::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
      ("help", "help message" )
      ("rays", bpo::value<int>(&rays_per_box)->default_value(10), 
       "number of rays") 
      ("lx", bpo::value<float>(&lx)->default_value(100.0), "Length of X side")
      ("ly", bpo::value<float>(&ly)->default_value(100.0), "Length of Y side")
      ("lz", bpo::value<float>(&lz)->default_value(100.0), "Length of Z side")
      ("nx", bpo::value<int>(&nx)->default_value(10), "number of X boxes")
      ("ny", bpo::value<int>(&ny)->default_value(10), "number of Y boxes")
      ("nz", bpo::value<int>(&nz)->default_value(10), "number of Z boxes")
      ("nx_mpi", bpo::value<int>(&nx_mpi)->default_value(1), "number of X ranks")
      ("ny_mpi", bpo::value<int>(&ny_mpi)->default_value(1), "number of Y ranks")
      ("nz_mpi", bpo::value<int>(&nz_mpi)->default_value(1), "number of Z ranks")
      ("print", bpo::value<bool>(&print)->default_value(false), "Print output")
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
    
    if (nx % nx_mpi + ny % ny_mpi + nz % nz_mpi != 0)
    {
      std::cerr << "ERROR: Please make the number of boxes in each direction"
                   " divisible by the number of MPI ranks in that direction."
                << std::endl;
      return EXIT_FAILURE;
    }

    int num_ranks_requested = nx_mpi * ny_mpi * nz_mpi;

    if (comm_rank == 0)
    {
      std::cout << "Running with " << num_ranks << " MPI ranks" << std::endl;
      if (num_ranks != num_ranks_requested)
      {
        std::cerr << "ERROR: Number of requested ranks (" << num_ranks_requested
                  << ") not equal number of existing ranks (" << num_ranks
                  << ")" << std::endl;
        return EXIT_FAILURE;
      }
    }
    
    int num_boxes = nx * ny * nz;
    float dx = lx / (float)nx;
    float dy = ly / (float)ny;
    float dz = lz / (float)nz;

    // Gathering global MPI coordinates
    int ix_mpi = comm_rank % nx_mpi;
    int iy_mpi = ((comm_rank - ix_mpi) / nx_mpi) % ny_mpi;
    int iz_mpi = (comm_rank - (ix_mpi + iy_mpi * nx_mpi)) / (nx_mpi * ny_mpi);

    ExecutionSpace exec_space{};

    Kokkos::Profiling::pushRegion("Example::problem_setup");
    Kokkos::Profiling::pushRegion("Example::make_grid");
    Kokkos::View<ArborX::Box *, MemorySpace> boxes(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "Example::boxes"),
        num_boxes);
    
    // Only construct boxes for this rank
    Kokkos::parallel_for(
        "Example::initialize_boxes",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecutionSpace>(
            exec_space, {0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          int const local_box_id = i + nx * j + nx * ny * k;
          int const i_global = ix_mpi * nx + i;
          int const j_global = iy_mpi * ny + j;
          int const k_global = iz_mpi * nz + k;
          boxes(local_box_id) = {
              {i_global * dx, j_global * dy, k_global * dz},
              {(i_global + 1) * dx, (j_global + 1) * dy, (k_global + 1) * dz}};
        });
    Kokkos::Profiling::popRegion();
    
    // For every box shoot rays from random (uniformly distributed) points
    // inside the box in random (uniformly distributed) directions.
    Kokkos::Profiling::pushRegion("Example::make_rays");
    Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> rays(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::rays"),
        rays_per_box * num_boxes);
    {
      using RandPoolType = Kokkos::Random_XorShift64_Pool<>;
      RandPoolType rand_pool(5374857);
      using GeneratorType = RandPoolType::generator_type;

      Kokkos::parallel_for(
          "Example::initialize_rays",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>(
              exec_space, {0, 0}, {num_boxes, rays_per_box}),
          KOKKOS_LAMBDA(const size_t i, const size_t j) {
            // The origins of rays are uniformly distributed in the boxes. The
            // direction vectors are uniformly sampling of a full sphere.
            GeneratorType g = rand_pool.get_state();
#if KOKKOS_VERSION >= 30700
            using Kokkos::cos;
            using Kokkos::sin;
            using Kokkos::acos;
#else
            using Kokkos::Experimental::cos;
            using Kokkos::Experimental::sin;
            using Kokkos::Experimental::acos;
#endif

            ArborX::Box const &b = boxes(i);
            ArborX::Point origin{
                b.minCorner()[0] +
                    Kokkos::rand<GeneratorType, float>::draw(g, dx),
                b.minCorner()[1] +
                    Kokkos::rand<GeneratorType, float>::draw(g, dy),
                b.minCorner()[2] +
                    Kokkos::rand<GeneratorType, float>::draw(g, dz)};

            float upsilon =
                Kokkos::rand<GeneratorType, float>::draw(g, 2.f * M_PI);
            float theta =
                acos(1 - 2 * Kokkos::rand<GeneratorType, float>::draw(g));
            ArborX::Experimental::Vector direction{cos(upsilon) * sin(theta),
                                                   sin(upsilon) * sin(theta),
                                                   cos(theta)};

            rays(j + i * rays_per_box) =
                ArborX::Experimental::Ray{origin, direction};

            rand_pool.free_state(g);
          });
    }
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::popRegion();
    
    ArborX::DistributedTree<MemorySpace> distributed_bvh{MPI_COMM_WORLD,
                                                         exec_space, boxes};

        Kokkos::View<MPIbased::IntersectedRank *, MemorySpace> values(
        "Example::values", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
    Kokkos::View<float *, MemorySpace> optical_path_lengths(
        "Example::optical_path_lengths", 0);
    Kokkos::View<float *, MemorySpace> intensity_contributions(
        "Example::intensity_contributions", 0);
    Kokkos::View<float *, MemorySpace> rank_entry_lengths(
        "Example::rank_entry_lengths", 0);
    distributed_bvh.query(
        exec_space, MPIbased::Rays<MemorySpace>{rays},
        MPIbased::AccumulateRayRankIntersectionData<MemorySpace>{boxes, comm_rank},
        values, offsets);

    // Ray IDs from originating rank need to be applied for sorting
    Kokkos::parallel_for(
        "Example::applying ray IDs", rays.extent(0),
        KOKKOS_LAMBDA(int const i) {
          for (int j = offsets(i); j < offsets(i + 1); j++)
          {
            values(j).ray_id = i;
          }
        });

    // Sorting ranks by intersection length
    Kokkos::View<MPIbased::IntersectedRankForSorting *, MemorySpace> sort_array(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "Example::sort_array"),
        values.size());
    Kokkos::parallel_for(
        "Example::copy sort_array",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, values.size()),
        KOKKOS_LAMBDA(int i) { sort_array(i) = values(i); });
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
    defined(KOKKOS_ENABLE_SYCL)
    auto permutation = ArborX::Details::sortObjects(exec_space, sort_array);
#else
    auto sort_array_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sort_array);
    Kokkos::View<int *, Kokkos::HostSpace> permutation_host(
        Kokkos::view_alloc("Example::permutation", Kokkos::WithoutInitializing),
        sort_array_host.size());
    std::iota(permutation_host.data(),
              permutation_host.data() + sort_array_host.size(), 0);
    std::sort(permutation_host.data(),
              permutation_host.data() + sort_array_host.size(),
              [&](int const &a, int const &b) {
                return (sort_array_host(a) < sort_array_host(b));
              });
    auto permutation =
        Kokkos::create_mirror_view_and_copy(MemorySpace{}, permutation_host);
#endif

    Kokkos::View<float *, MemorySpace> local_energies("Example::local_energies",
                                                      num_boxes);

    Kokkos::parallel_for(
        "Example::evaluating ray intensities", rays.extent(0),
        KOKKOS_LAMBDA(int const i) {
#if KOKKOS_VERSION >= 30700
          using Kokkos::exp;
#else
          using Kokkos::Experimental::exp;
#endif
          float accum_opt_dist = 0, ray_intensity = 0;
          for (int j = offsets(i); j < offsets(i + 1); ++j)
          {
            const auto &v = values(permutation(j));
            ray_intensity += exp(-accum_opt_dist) * v.intensity_contribution;
            accum_opt_dist += v.optical_path_length;
          }
          int box_id = (int)i / rays_per_box;
          Kokkos::atomic_add(&local_energies(box_id),
                             ray_intensity * 4 * pi * kappa / rays_per_box);
        });

    Kokkos::Profiling::pushRegion("Example::printing_output");
    if (print)
    {
      int num_boxes_global = num_boxes * num_ranks_requested;
      auto local_energies_h =
          Kokkos::create_mirror_view_and_copy(HostSpace{}, local_energies);

      if (comm_rank != 0)
      {
        MPI_Send(local_energies_h.data(), int(local_energies_h.size()),
                 MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      }
      else
      {
        Kokkos::View<float ***, HostSpace> global_energies_h(
            "all boxes", nx_mpi * nx, ny_mpi * ny, nz_mpi * nz);
        Kokkos::View<float *, HostSpace> mpi_buffer_h("mpi_buffer", num_boxes);
        MPI_Status status;

        for (int source_rank = 0; source_rank < num_ranks; ++source_rank)
        {
          if (source_rank > 0)
            MPI_Recv(mpi_buffer_h.data(), int(mpi_buffer_h.size()), MPI_FLOAT,
                     source_rank, 0, MPI_COMM_WORLD, &status);
          else
            mpi_buffer_h = local_energies;

          // Gather source rank's MPI coordinates
          int ix_mpi_source = source_rank % nx_mpi;
          int iy_mpi_source = ((source_rank - ix_mpi_source) / nx_mpi) % ny_mpi;
          int iz_mpi_source =
              (source_rank - (ix_mpi_source + iy_mpi_source * nx_mpi)) /
              (nx_mpi * ny_mpi);

          Kokkos::parallel_for(
              "Energy Copying",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>, HostExecutionSpace>(
                  exec_space, {0, 0, 0}, {nx, ny, nz}),
              KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
                int bid_local = i + j * nx + k * nx * ny;
                int i_global = i + ix_mpi_source * nx;
                int j_global = j + iy_mpi_source * ny;
                int k_global = k + iz_mpi_source * nz;
                global_energies_h(i_global, j_global, k_global) =
                    mpi_buffer_h(bid_local);
              });
        }

        std::cout << "Net radiative absorptions:" << std::endl;
        printoutput(global_energies_h, dx, dy, dz);
      }
    }
    Kokkos::Profiling::popRegion();
  }
  Kokkos::finalize();
  MPI_Finalize();

  return EXIT_SUCCESS;
}
