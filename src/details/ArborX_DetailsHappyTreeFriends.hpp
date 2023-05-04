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

#ifndef ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP

#include <ArborX_DetailsNode.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility> // declval

namespace ArborX
{
namespace Details
{
struct HappyTreeFriends
{
  template <class BVH>
  static KOKKOS_FUNCTION int getRoot(BVH const &)
  {
    return 0;
  }

  template <class BVH>
  static KOKKOS_FUNCTION int leafIndex(BVH const &bvh, int i)
  {
    return i - ((int)bvh.size() - 1);
  }

  template <class BVH>
// FIXME_HIP See https://github.com/arborx/ArborX/issues/553
#ifdef __HIP_DEVICE_COMPILE__
  static KOKKOS_FUNCTION auto getBoundingVolume(BVH const &bvh, int i)
#else
  static KOKKOS_FUNCTION auto const &getBoundingVolume(BVH const &bvh, int i)
#endif
  {
    auto const leaf_index = leafIndex(bvh, i);
    return (leaf_index < 0 ? bvh._internal_nodes(i).bounding_volume
                           : bvh._leaf_nodes(leaf_index).bounding_volume);
  }

  template <class BVH>
  static KOKKOS_FUNCTION bool isLeaf(BVH const &bvh, int i)
  {
    return leafIndex(bvh, i) >= 0;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeafPermutationIndex(BVH const &bvh, int i)
  {
    auto const leaf_index = leafIndex(bvh, i);
    assert(leaf_index >= 0);
    return bvh._leaf_nodes(leaf_index).permutation_index;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeftChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    return bvh._internal_nodes(i).left_child;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRightChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    int const left_child = getLeftChild(bvh, i);
    int const leaf_left_child = leafIndex(bvh, left_child);
    return (leaf_left_child < 0 ? bvh._internal_nodes(left_child).rope
                                : bvh._leaf_nodes(leaf_left_child).rope);
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRope(BVH const &bvh, int i)
  {
    auto const leaf_index = leafIndex(bvh, i);
    return (leaf_index < 0 ? bvh._internal_nodes(i).rope
                           : bvh._leaf_nodes(leaf_index).rope);
  }
};
} // namespace Details
} // namespace ArborX

#endif
