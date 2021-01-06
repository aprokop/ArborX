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

#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_KDTree.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"

#define BOOST_TEST_MODULE KDTree

namespace tt = boost::test_tools;

template <class DeviceType>
struct CustomCallback
{
  Kokkos::View<int *, DeviceType> counts;
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int) const
  {
    int i = getData(predicate);
    ++counts(i);
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(find_yourself, DeviceType, ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::KDTree<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  int const n = 100;
  Kokkos::View<ArborX::Point *, ExecutionSpace> points("Testing::points", n);
  auto points_host = Kokkos::create_mirror_view(points);
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution_x(0.0, 10.0);
  std::uniform_real_distribution<float> distribution_y(0.0, 10.0);
  std::uniform_real_distribution<float> distribution_z(0.0, 10.0);
  for (int i = 0; i < n; ++i)
  {
    float x = distribution_x(generator);
    float y = distribution_y(generator);
    float z = distribution_z(generator);
    points_host(i) = {{x, y, z}};
  }
  Kokkos::deep_copy(points, points_host);

  Kokkos::View<decltype(
                   ArborX::attach(ArborX::intersects(ArborX::Sphere{}), 0)) *,
               DeviceType>
      queries("Testing::queries", n);
  Kokkos::parallel_for(
      "Testing::construct_queries", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::attach(
            ArborX::intersects(ArborX::Sphere{points(i), 1e-10}), i);
      });

  Tree tree(ExecutionSpace{}, points);

  Kokkos::View<int *, DeviceType> counts("counts", n);

  std::vector<int> counts_ref(n, 1);

  tree.query(ExecutionSpace{}, queries, CustomCallback<DeviceType>{counts});

  auto counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counts);

  BOOST_TEST(counts_host == counts_ref, tt::per_element());
}

#if 0
template <class DeviceType>
struct Experimental_CustomCallbackEarlyExit
{
  Kokkos::View<int *, DeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> counts;
  template <class Predicate>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate, int) const
  {
    int i = getData(predicate);

    if (counts(i)++ < i)
    {
      return ArborX::CallbackTreeTraversalControl::normal_continuation;
    }

    // return ArborX::CallbackTreeTraversalControl::early_exit;
    return ArborX::CallbackTreeTraversalControl::normal_continuation;
    ;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_early_exit, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::KDTree<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  int const n = 4;
  Kokkos::View<ArborX::Point *, typename Tree::memory_space> points(
      "Testing::points", n);
  auto points_host = Kokkos::create_mirror_view(points);
  for (int i = 0; i < n; ++i)
    points_host(i) = {{(float)i, (float)i, (float)i}};
  Kokkos::deep_copy(points, points_host);

  Tree tree(ExecutionSpace{}, points);

  Kokkos::View<int *, DeviceType> counts("counts", 4);

  std::vector<int> counts_ref(4);
  std::iota(counts_ref.begin(), counts_ref.end(), 1);

  auto b = tree.bounds();
  auto predicates = makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
      {b, b, b, b}, {0, 1, 2, 3});

  tree.query(ExecutionSpace{}, predicates,
             Experimental_CustomCallbackEarlyExit<DeviceType>{counts});

  auto counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counts);

  BOOST_TEST(counts_host == counts_ref, tt::per_element());
}
#endif
