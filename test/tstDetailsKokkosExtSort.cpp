/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExtSort.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp> // iota
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE KokkosExtSort

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_by_key, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  for (auto const &values : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                             std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                             std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                             std::vector<int>{15, 5, 11, 3, 4, 8}})
  {
    auto const n = values.size();

    Kokkos::View<int *, Kokkos::HostSpace> host_view("data", n);
    std::copy(values.begin(), values.end(), host_view.data());
    auto device_view = Kokkos::create_mirror_view_and_copy(space, host_view);

    Kokkos::View<int *, typename DeviceType::memory_space> device_permutation(
        "permute", n);
    ArborX::Details::KokkosExt::iota(space, device_permutation);

    ArborX::Details::KokkosExt::sortByKey(space, device_view,
                                          device_permutation);

    Kokkos::deep_copy(space, host_view, device_view);

    // Check that values were sorted properly
    std::vector<int> values_copy = values;
    std::sort(values_copy.begin(), values_copy.end());
    auto host_permutation = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, device_permutation);
    BOOST_TEST(host_view == values_copy, tt::per_element());

    // Check correctness of the permutation
    for (unsigned int i = 0; i < values.size(); ++i)
      values_copy[i] = values[host_permutation(i)];
    BOOST_TEST(host_view == values_copy, tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_by_key_variadic, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  std::vector<int> ids_ = {4, 3, 2, 1, 4, 3, 2, 4, 3, 4};
  std::vector<int> sorted_ids = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  std::vector<int> offset = {0, 1, 3, 6, 10};
  int const n = 10;
  int const m = 4;
  BOOST_TEST(ids_.size() == n);
  BOOST_TEST(sorted_ids.size() == n);
  BOOST_TEST(offset.size() == m + 1);
  std::vector<int> results_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<std::set<int>> sorted_results = {
      {3},
      {6, 2},
      {8, 5, 1},
      {9, 7, 4, 0},
  };
  std::vector<int> ranks_ = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<std::set<int>> sorted_ranks = {
      {13},
      {16, 12},
      {18, 15, 11},
      {19, 17, 14, 10},
  };
  BOOST_TEST(results_.size() == n);
  BOOST_TEST(ranks_.size() == n);

  Kokkos::View<int *, DeviceType> ids("query_ids", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ids_host(ids_.data(), ids_.size());
  Kokkos::deep_copy(ids, ids_host);

  Kokkos::View<int *, DeviceType> results("results", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      results_host(results_.data(), results_.size());
  Kokkos::deep_copy(results, results_host);

  Kokkos::View<int *, DeviceType> ranks("ranks", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ranks_host(ranks_.data(), ranks_.size());
  Kokkos::deep_copy(ranks, ranks_host);

  using ExecutionSpace = typename DeviceType::execution_space;
  ArborX::KokkosExt::sortByKey(ExecutionSpace{}, ids, results, ranks);

  Kokkos::deep_copy(results_host, results);
  Kokkos::deep_copy(ranks_host, ranks);
  for (int q = 0; q < m; ++q)
    for (int i = offset[q]; i < offset[q + 1]; ++i)
    {
      BOOST_TEST(sorted_results[q].count(results_host[i]) == 1);
      BOOST_TEST(sorted_ranks[q].count(ranks_host[i]) == 1);
    }

  Kokkos::View<int *, DeviceType> not_sized_properly("", m);
  BOOST_CHECK_THROW(ArborX::KokkosExt::sortResultsByKey(ExecutionSpace{}, ids,
                                                        not_sized_properly),
                    ArborX::SearchException);
}
