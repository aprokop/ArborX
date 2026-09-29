/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_LinearBVH.hpp>
#include <misc/ArborX_IndexType.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace
{

// Points 0, 1, 2, ..., n - 1 on a line, generated on the fly so that the
// input does not consume any memory
template <typename MemorySpace>
struct Line
{
  std::int64_t _n;
};

} // namespace

template <typename MemorySpace>
struct ArborX::AccessTraits<Line<MemorySpace>>
{
  using memory_space = MemorySpace;
  static KOKKOS_FUNCTION std::int64_t size(Line<MemorySpace> const &line)
  {
    return line._n;
  }
  static KOKKOS_FUNCTION auto get(Line<MemorySpace> const &,
                                  ArborX::Details::index_type i)
  {
    return ArborX::Point<1, double>{{static_cast<double>(i)}};
  }
};

namespace
{

// Returns the available memory in bytes (0 if unknown)
std::uint64_t physicalMemory()
{
#if defined(__linux__) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
  auto const pages = sysconf(_SC_PHYS_PAGES);
  auto const page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0)
    return static_cast<std::uint64_t>(pages) *
           static_cast<std::uint64_t>(page_size);
#endif
  return 0;
}

} // namespace

BOOST_AUTO_TEST_SUITE(LargeIndex)

BOOST_AUTO_TEST_CASE(index_type_matches_configuration)
{
#ifdef ARBORX_ENABLE_LARGE_INDEX
  static_assert(sizeof(ArborX::Details::index_type) == 8);
#else
  static_assert(sizeof(ArborX::Details::index_type) == 4);
#endif
  static_assert(std::is_signed_v<ArborX::Details::index_type>);
}

#ifdef ARBORX_ENABLE_LARGE_INDEX
BOOST_AUTO_TEST_CASE_TEMPLATE(more_than_max_int_values, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  std::int64_t const n =
      static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1000;

  // The tree, the sorting and the construction take roughly 100 bytes per
  // value; skip if there is clearly not enough memory
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>)
  {
    auto const memory = physicalMemory();
    if (memory != 0 && memory < static_cast<std::uint64_t>(n) * 128)
    {
      BOOST_TEST_MESSAGE("Skipping: not enough memory");
      return;
    }
  }

  ExecutionSpace space;
  Line<MemorySpace> const points{n};
  ArborX::BoundingVolumeHierarchy<MemorySpace, ArborX::Point<1, double>> bvh(
      space, points);
  BOOST_TEST(static_cast<std::int64_t>(bvh.size()) == n);

  // Spatial query for the last few points (indices beyond MAX_INT)
  using Box = ArborX::Box<1, double>;
  using Predicate = decltype(ArborX::intersects(Box{}));
  Kokkos::View<Predicate *, MemorySpace> predicates("Test::predicates", 1);
  Kokkos::parallel_for(
      "Test::fill_predicates", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int) {
        predicates(0) = ArborX::intersects(
            Box{{static_cast<double>(n - 4)}, {static_cast<double>(n - 1)}});
      });

  Kokkos::View<int, MemorySpace> count("Test::count");
  bvh.query(
      space, predicates, KOKKOS_LAMBDA(auto const &, auto const &) {
        Kokkos::atomic_inc(&count());
      });
  auto count_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);
  BOOST_TEST(count_host() == 4);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
