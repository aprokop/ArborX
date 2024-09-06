/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
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

#include <boost/test/unit_test.hpp>

#include <string>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(KokkosToolsExecutionSpaceInstances)

namespace tt = boost::test_tools;

namespace
{
// Lambdas can only be converted to function pointers if they do not capture.
// Using a global non-static variable in an unnamed namespace to "capture" the
// device id.
uint32_t arborx_test_device_id = -1;
uint32_t arborx_test_root_device_id = -1;

void arborx_test_parallel_x_callback(char const *label, uint32_t device_id,
                                     uint64_t * /*kernel_id*/)
{
  std::string label_str(label);

  for (std::string s : {"Kokkos::View::destruction []"})
    if (label_str.find(s) != std::string::npos)
      return;

  BOOST_TEST(device_id == arborx_test_device_id,
             "\"" << label
                  << "\" kernel not on the right execution space instance: "
                  << device_id << " != " << arborx_test_device_id);
}

void arborx_test_fence_callback(char const *label, uint32_t device_id,
                                uint64_t * /*fence_id*/)
{
  std::string label_str(label);
  // clang-format off
  for (std::string s : {
         "Kokkos::Tools",
         "HostSpace::impl_deallocate",
         "Kokkos::deep_copy: copy between contiguous views, pre view equality check",
         "Kokkos::deep_copy: copy between contiguous views, fence due to same spans",
         "Kokkos::deep_copy: copy between contiguous views, post deep copy fence",
         "fence after copying header from HostSpace",
         "Kokkos::create_mirror_view_and_copy: fence before returning src view",
         "Kokkos::Impl::ViewValueFunctor: View init/destroy fence"
      })
    if (label_str.find(s) != std::string::npos)
      return;
  // clang-format on

  if (device_id != arborx_test_root_device_id)
    BOOST_TEST(device_id == arborx_test_device_id,
               "\"" << label
                    << "\" fence not on the right execution space instance: "
                    << device_id << " != " << arborx_test_device_id);
}

template <class ExecutionSpace>
void arborx_test_set_tools_callbacks(ExecutionSpace exec)
{
  arborx_test_device_id = Kokkos::Tools::Experimental::device_id(exec);
  arborx_test_root_device_id =
      Kokkos::Tools::Experimental::device_id_root<ExecutionSpace>();

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      arborx_test_parallel_x_callback);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(
      arborx_test_parallel_x_callback);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(
      arborx_test_parallel_x_callback);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      arborx_test_fence_callback);
}

void arborx_test_unset_tools_callbacks()
{
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  arborx_test_device_id = -1;
  arborx_test_root_device_id = -1;
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_bvh_execution_space_instance, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  auto exec = Kokkos::Experimental::partition_space(ExecutionSpace{}, 1)[0];
  arborx_test_set_tools_callbacks(exec);

  { // default constructed
    Tree tree;
  }

  { // empty
    auto tree = make<Tree>(exec, {});
  }

  { // one leaf
    auto tree = make<Tree>(exec, {
                                     {{{0, 0, 0}}, {{1, 1, 1}}},
                                 });
  }

  { // two leaves
    auto tree = make<Tree>(exec, {
                                     {{{0, 0, 0}}, {{1, 1, 1}}},
                                     {{{0, 0, 0}}, {{1, 1, 1}}},
                                 });
  }

  arborx_test_unset_tools_callbacks();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_query_execution_space_instance, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto tree = make<ArborX::BVH<typename DeviceType::memory_space>>(
      ExecutionSpace{}, {
                            {{{0, 0, 0}}, {{1, 1, 1}}},
                            {{{0, 0, 0}}, {{1, 1, 1}}},
                        });

  auto exec = Kokkos::Experimental::partition_space(ExecutionSpace{}, 1)[0];
  arborx_test_set_tools_callbacks(exec);

  // spatial predicates
  query(exec, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {{{0, 0, 0}}, {{1, 1, 1}}},
            {{{0, 0, 0}}, {{1, 1, 1}}},
        }));

  // nearest predicates
  query(exec, tree,
        makeNearestQueries<DeviceType>({
            {{{0, 0, 0}}, 1},
            {{{0, 0, 0}}, 2},
        }));

  arborx_test_unset_tools_callbacks();
}

BOOST_AUTO_TEST_SUITE_END()
