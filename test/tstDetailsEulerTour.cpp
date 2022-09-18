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
#include <ArborX_DetailsEulerTour.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

template <typename DeviceType, typename T>
auto toView(std::vector<T> const &v, std::string const &lbl = "")
{
  Kokkos::View<T *, DeviceType> view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, lbl), v.size());
  Kokkos::deep_copy(view, Kokkos::View<T const *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              v.data(), v.size()));
  return view;
}

using ArborX::Details::WeightedEdge;

namespace Test
{
template <class ExecutionSpace>
auto compute_successors(ExecutionSpace const &exec_space,
                        std::vector<WeightedEdge> const &edges_host)
{
  auto edges = toView<ExecutionSpace>(edges_host, "Test::edges");
  auto successors = ArborX::Details::eulerTourList(exec_space, edges);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, successors);
}

template <class ExecutionSpace>
auto rank_list(ExecutionSpace const &exec_space,
               std::vector<int> const &list_host, int head)
{
  auto list = toView<ExecutionSpace>(list_host, "Test::list");
  auto ranks = ArborX::Details::rankList(exec_space, list, head);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ranks);
}
} // namespace Test

#define ARBORX_TEST_COMPUTE_SUCCESSORS(exec_space, edges, ref)                 \
  BOOST_TEST(Test::compute_successors(exec_space, edges) == ref,               \
             boost::test_tools::per_element())

#define ARBORX_TEST_RANK_LIST(exec_space, list, head, ref)                     \
  BOOST_TEST(Test::rank_list(exec_space, list, head) == ref,                   \
             boost::test_tools::per_element())

BOOST_AUTO_TEST_SUITE(EulerTour)

BOOST_AUTO_TEST_CASE_TEMPLATE(compute_successors, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  ARBORX_TEST_COMPUTE_SUCCESSORS(exec_space,
                                 (std::vector<WeightedEdge>{{0, 1, 0.f}}),
                                 (std::vector<int>{1, 0}));

  // Example from the Polak's "Euler meets GPU" paper
  ARBORX_TEST_COMPUTE_SUCCESSORS(
      exec_space,
      (std::vector<WeightedEdge>{
          {0, 2, 0.f}, {0, 3, 0.f}, {0, 4, 0.f}, {2, 1, 0.f}, {2, 5, 0.f}}),
      (std::vector<int>{6, 2, 3, 4, 5, 0, 7, 8, 9, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(rank_list, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  ARBORX_TEST_RANK_LIST(exec_space, (std::vector<int>{1, 0}), 0,
                        (std::vector<int>{0, 1}));
  ARBORX_TEST_RANK_LIST(exec_space, (std::vector<int>{1, 0}), 1,
                        (std::vector<int>{1, 0}));

  //                    0  1  2  3  4  5   6  7  8  9  10
  std::vector<int> list{3, 9, 4, 6, 7, 2, 10, 0, 5, 8, 1};
  ARBORX_TEST_RANK_LIST(exec_space, list, 1,
                        (std::vector<int>{7, 0, 4, 8, 5, 3, 9, 6, 2, 1, 10}));
  ARBORX_TEST_RANK_LIST(exec_space, list, 7,
                        (std::vector<int>{1, 5, 9, 2, 10, 8, 3, 0, 7, 6, 4}));

  int const n = 101483;
  int const step = 13; // step must be co-prime with n to avoid sub-loops
  int const large_head = 1337;
  std::vector<int> large_list(n);
  std::vector<int> large_ref(n);
  for (int count = 0; count < n; ++count)
  {
    int i = (large_head + step * count) % n;
    large_list[i] = (i + step) % n;
    large_ref[i] = count;
  }
  ARBORX_TEST_RANK_LIST(exec_space, large_list, large_head, large_ref);
}

BOOST_AUTO_TEST_SUITE_END()
