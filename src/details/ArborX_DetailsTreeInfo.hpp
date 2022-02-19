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
#ifndef ARBORX_DETAILS_TREE_INFO_HPP
#define ARBORX_DETAILS_TREE_INFO_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>

namespace ArborX
{
namespace Details
{

struct TreeInfo
{
  template <typename ExecutionSpace, typename BVH>
  static Kokkos::View<int *, typename BVH::memory_space>
  computeHeights(ExecutionSpace const &exec_space, BVH const &bvh)
  {
    using MemorySpace = typename BVH::memory_space;
    Kokkos::View<int *, MemorySpace> heights("ArborX::TreeInfo::heights",
                                             bvh.size());
    Kokkos::parallel_for(
        "ArborX::TreeInfo::compute_heights",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, bvh.size()),
        KOKKOS_LAMBDA(int i) {
          int const leaf_nodes_shift = bvh.size() - 1;

          int height = 0;

          int node;
          int next = HappyTreeFriends::getRoot(bvh);
          do
          {
            node = next;
            if (!HappyTreeFriends::isLeaf(bvh, node))
            {
              ++height;

              int const left_child = HappyTreeFriends::getLeftChild(bvh, node);
              if ((!HappyTreeFriends::isLeaf(bvh, left_child) &&
                   left_child >= i) ||
                  left_child - leaf_nodes_shift == i)
                next = left_child;
              else
                next = HappyTreeFriends::getRightChild(bvh, node);
            }
            else
            {
              if (node - leaf_nodes_shift == i)
                break;
            }
          } while (next != ROPE_SENTINEL);
          heights(i) = height;
        });
    return heights;
  }
};

} // namespace Details
} // namespace ArborX

#endif
