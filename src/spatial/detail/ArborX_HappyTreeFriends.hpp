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

#ifndef ARBORX_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_HAPPY_TREE_FRIENDS_HPP

#include <misc/ArborX_IndexType.hpp>

#include <Kokkos_Assert.hpp> // KOKKOS_ASSERT
#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX::Details
{

struct HappyTreeFriends
{
  template <class BVH>
  static KOKKOS_FUNCTION index_type getRoot(BVH const &bvh)
  {
    KOKKOS_ASSERT(bvh.size() > 1);
    return bvh.size();
  }

  template <class BVH>
  static KOKKOS_FUNCTION bool isLeaf(BVH const &bvh, index_type i)
  {
    KOKKOS_ASSERT(bvh.size() > 1);
    KOKKOS_ASSERT(i >= 0 && i < 2 * (index_type)bvh.size() - 1);
    return i < (index_type)bvh.size();
  }

  template <class BVH>
  static KOKKOS_FUNCTION index_type internalIndex(BVH const &bvh, index_type i)
  {
    return i - (index_type)bvh.size();
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto const &getInternalBoundingVolume(BVH const &bvh,
                                                               index_type i)
  {
    return bvh._internal_nodes(internalIndex(bvh, i)).bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto) getIndexable(BVH const &bvh,
                                                     index_type i)
  {
    return bvh._indexable_getter(getValue(bvh, i));
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto const &getValue(BVH const &bvh, index_type i)
  {
    KOKKOS_ASSERT(i >= 0 && i < (index_type)bvh.size());
    return bvh._leaf_nodes(i).value;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeftChild(BVH const &bvh, index_type i)
  {
    KOKKOS_ASSERT(!isLeaf(bvh, i));
    return bvh._internal_nodes(internalIndex(bvh, i)).left_child;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRightChild(BVH const &bvh, index_type i)
  {
    KOKKOS_ASSERT(!isLeaf(bvh, i));
    return getRope(bvh, getLeftChild(bvh, i));
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRope(BVH const &bvh, index_type i)
  {
    return (isLeaf(bvh, i) ? bvh._leaf_nodes(i).rope
                           : bvh._internal_nodes(internalIndex(bvh, i)).rope);
  }
};
} // namespace ArborX::Details

#endif
