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
#include <ArborX_DetailsTreeNodeLabeling.hpp>
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

template <class MemorySpace>
struct MockBVH
{
  Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> children_;
  KOKKOS_FUNCTION auto size() const { return children_.extent(0) + 1; }
};

#define HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(MEMORY_SPACE)           \
  template <>                                                                  \
  auto ArborX::Details::HappyTreeFriends::getLeftChild<MockBVH<MEMORY_SPACE>>( \
      MockBVH<MEMORY_SPACE> const &x, int i)                                   \
  {                                                                            \
    return x.children_(i).first;                                               \
  }                                                                            \
  template <>                                                                  \
  auto                                                                         \
  ArborX::Details::HappyTreeFriends::getRightChild<MockBVH<MEMORY_SPACE>>(     \
      MockBVH<MEMORY_SPACE> const &x, int i)                                   \
  {                                                                            \
    return x.children_(i).second;                                              \
  }

HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(Kokkos::HostSpace)
#ifdef KOKKOS_ENABLE_CUDA
HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(Kokkos::Cuda::memory_space)
#endif
#ifdef KOKKOS_ENABLE_HIP
HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(
    Kokkos::Experimental::HIP::memory_space)
#endif
#ifdef KOKKOS_ENABLE_SYCL
HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(
    Kokkos::Experimental::SYCL::memory_space)
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
HAPPY_TREE_FRIENDS_GET_CHILDREN_SPECIALIZATION(
    Kokkos::Experimental::OpenMPTarget::memory_space)
#endif

namespace Test
{
template <class ExecutionSpace>
auto findParents(ExecutionSpace const &exec_space,
                 std::vector<Kokkos::pair<int, int>> const &children_host)
{
  auto children = toView<ExecutionSpace>(children_host, "Test::children");

  auto const n = children.extent(0) + 1;
  Kokkos::View<int *, ExecutionSpace> parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::parents"),
      2 * n - 1);
  Kokkos::deep_copy(parents, -1);

  using ArborX::Experimental::findParents;
  findParents(exec_space, MockBVH<Kokkos::HostSpace>{children}, parents);
  return parents;
}

#define ARBORX_TEST_FIND_PARENTS(exec_space, children, ref)                    \
  BOOST_TEST(Test::findParents(exec_space, children) == ref,                   \
             boost::test_tools::per_element())

template <class ExecutionSpace>
auto reduceLabels(ExecutionSpace const &exec_space,
                  std::vector<int> const &parents_host,
                  std::vector<int> const &labels_host)
{
  auto labels = toView<ExecutionSpace>(labels_host, "Test::labels");
  auto parents = toView<ExecutionSpace>(parents_host, "Test::parents");

  using ArborX::Experimental::reduceLabels;
  reduceLabels(exec_space, parents, labels);

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);
}

#define ARBORX_TEST_REDUCE_LABELS(exec_space, parents, labels, ref)            \
  BOOST_TEST(Test::reduceLabels(exec_space, parents, labels) == ref,           \
             boost::test_tools::per_element())

} // namespace Test

BOOST_AUTO_TEST_SUITE(TreeHelpers)

BOOST_AUTO_TEST_CASE_TEMPLATE(find_parents, DeviceType, ARBORX_DEVICE_TYPES)
{

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  // [0]--*--
  //     / \
  //    |   |
  //    0   1
  ARBORX_TEST_FIND_PARENTS(exec_space,
                           (std::vector<Kokkos::pair<int, int>>{{1, 2}}),
                           (std::vector<int>{-1, 0, 0}));

  // [0]----*----
  //       / \
  //  --*[1]  |
  //   / \    |
  //  |   |   |
  //  |   |   |
  //  0   1   2
  ARBORX_TEST_FIND_PARENTS(
      exec_space, (std::vector<Kokkos::pair<int, int>>{{1, 4}, {2, 3}}),
      (std::vector<int>{-1, 0, 1, 1, 0}));

  // [0]------------*--------------
  //               / \
  //  ------*----[3] [4]*----------
  //       / \         / \
  //  --*[1] [2]*--   |  [5]----*--
  //   / \     / \    |        / \
  //  |   |   |   |   |   --*[6]  |
  //  |   |   |   |   |    / \    |
  //  0   1   2   3   4   5   6   7
  auto const parents =
      std::vector<int>{-1, 3, 3, 0, 0, 4, 5, 1, 1, 2, 2, 4, 6, 6, 5};
  //                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  //                   [0][1][2][3][4][5][6] 0  1  2  3  4  5  6  7

  auto const children = std::vector<Kokkos::pair<int, int>>{
      {3, 4}, {7, 8}, {9, 10}, {1, 2}, {11, 5}, {6, 14}, {12, 13}};
  //  [0]     [1]     [2]      [3]      [4]     [5]       [6]
  //  [3][4]   0  1    2   3   [1][2]    4 [5]  [6]  7     5   6
  ARBORX_TEST_FIND_PARENTS(exec_space, children, parents);
}

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

BOOST_AUTO_TEST_SUITE_END()
