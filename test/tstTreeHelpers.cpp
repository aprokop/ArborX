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
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_ExperimentalTreeHelpers.hpp>
#include <ArborX_LinearBVH.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <random>

template <typename DeviceType, typename T>
auto toView(std::vector<T> const &v, std::string const &lbl = "")
{
  Kokkos::View<T *, DeviceType> view(lbl, v.size());
  Kokkos::deep_copy(view, Kokkos::View<T const *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              v.data(), v.size()));
  return view;
}

namespace Test
{
template <class ExecutionSpace>
auto reduceLabels(ExecutionSpace const &exec_space,
                  std::vector<int> const &parents_host,
                  std::vector<int> const &labels_host)
{
  auto labels = toView<ExecutionSpace>(labels_host, "Test::labels");
  auto parents = toView<ExecutionSpace>(parents_host, "Test::parents");

  using ArborX::Experimental::reduceBVHLabels;
  reduceBVHLabels(exec_space, parents, labels);

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);
}

template <typename ExecutionSpace>
auto buildRandomBVH(ExecutionSpace const &exec_space, int n)
{
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::View<ArborX::Point *, MemorySpace> points("Test::points", n);

  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  auto points_host = Kokkos::create_mirror_view(points);
  for (int i = 0; i < n; ++i)
    points_host(i) = ArborX::Point{dist(gen), dist(gen), dist(gen)};
  Kokkos::deep_copy(points, points_host);

  return ArborX::BVH<MemorySpace>(exec_space, points);
}

} // namespace Test

#define ARBORX_TEST_REDUCE_LABELS(exec_space, parents, labels, ref)            \
  BOOST_TEST(Test::reduceLabels(exec_space, parents, labels) == ref,           \
             boost::test_tools::per_element())

BOOST_AUTO_TEST_SUITE(TreeHelpers)

BOOST_AUTO_TEST_CASE_TEMPLATE(reduce_labels, DeviceType, ARBORX_DEVICE_TYPES)
{
  /*
      [0]------------*--------------
                    / \
       ------*----[3] [4]*----------
            / \         / \
       --*[1] [2]*--   |  [5]----*--
        / \     / \    |        / \
       |   |   |   |   |   --*[6]  |
       |   |   |   |   |    / \    |
       0   1   2   3   4   5   6   7
  */
  auto const parents =
      std::vector<int>{-1, 3, 3, 0, 0, 4, 5, 1, 1, 2, 2, 4, 6, 6, 5};
  //                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  //                   [0][1][2][3][4][5][6] 0  1  2  3  4  5  6  7

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  ARBORX_TEST_REDUCE_LABELS(
      exec_space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0}),
      (std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  ARBORX_TEST_REDUCE_LABELS(
      exec_space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7}),
      (std::vector<int>{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}));

  ARBORX_TEST_REDUCE_LABELS(
      exec_space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7}),
      (std::vector<int>{-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7}));

  ARBORX_TEST_REDUCE_LABELS(
      exec_space, parents,
      (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 3, 4, 4, 4, 7}),
      (std::vector<int>{-1, 0, -1, -1, -1, -1, 4, 0, 0, 0, 3, 4, 4, 4, 7}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(find_parents, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  ExecutionSpace exec_space;

  int const n = 103;

  auto bvh = Test::buildRandomBVH(exec_space, n);

  Kokkos::View<int *, MemorySpace> bvh_parents("Test::bvh_parents", 2 * n - 1);
  Kokkos::deep_copy(exec_space, bvh_parents, -1);

  ArborX::Experimental::findBVHParents(exec_space, bvh, bvh_parents);

  int num_failures = 0;
  Kokkos::parallel_reduce(
      "Test::check_bvh_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, 2 * n - 1),
      KOKKOS_LAMBDA(int i, int &update) {
        using ArborX::Details::HappyTreeFriends;

        int const parent = bvh_parents(i);

        if (i == 0) // root
        {
          if (parent != -1)
            ++update;
          return;
        }

        if (parent < 0 || parent >= n)
        {
          ++update;
          return;
        }

        if (HappyTreeFriends::getLeftChild(bvh, parent) != i &&
            HappyTreeFriends::getRightChild(bvh, parent) != i)
          ++update;
      },
      num_failures);
  BOOST_TEST(num_failures == 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(init_labels, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  ExecutionSpace exec_space;

  int const n = 103;

  auto bvh = Test::buildRandomBVH(exec_space, n);

  Kokkos::View<int *, MemorySpace> labels("Test::labels", n);

  std::default_random_engine gen;
  std::uniform_int_distribution<int> dist(0, INT_MAX);
  auto labels_host = Kokkos::create_mirror_view(labels);
  for (int i = 0; i < n; ++i)
    labels_host(i) = dist(gen);
  Kokkos::deep_copy(labels, labels_host);

  Kokkos::View<int *, MemorySpace> bvh_labels("Test::labels", 2 * n - 1);

  ArborX::Experimental::initBVHLabels(exec_space, bvh, labels, bvh_labels);

  int num_failures = 0;
  Kokkos::parallel_reduce(
      "Test::check_bvh_labels",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, n - 1, 2 * n - 1),
      KOKKOS_LAMBDA(int i, int &update) {
        using ArborX::Details::HappyTreeFriends;

        int permuted_index = HappyTreeFriends::getLeafPermutationIndex(bvh, i);
        if (bvh_labels(i) != labels(permuted_index))
          ++update;
      },
      num_failures);
  BOOST_TEST(num_failures == 0);
}

BOOST_AUTO_TEST_SUITE_END()
