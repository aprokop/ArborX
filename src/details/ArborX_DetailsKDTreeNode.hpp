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

#ifndef ARBORX_KDTREENODE_HPP
#define ARBORX_KDTREENODE_HPP

#include <ArborX_DetailsAAPlane.hpp>
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_Point.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

struct KDTreeNode
{
};

struct KDTreeNodeLeaf : public KDTreeNode
{
  KOKKOS_DEFAULTED_FUNCTION constexpr KDTreeNodeLeaf() = default;
  KOKKOS_FUNCTION constexpr KDTreeNodeLeaf(int pindex, Point p)
      : permutation_index(pindex)
      , point(p)
  {
  }

  KOKKOS_FUNCTION constexpr std::size_t getLeafPermutationIndex() const noexcept
  {
    return permutation_index;
  }

  int permutation_index = -1;
  Point point = {KokkosExt::ArithmeticTraits::max<float>::value,
                 KokkosExt::ArithmeticTraits::max<float>::value,
                 KokkosExt::ArithmeticTraits::max<float>::value};
};

KOKKOS_FUNCTION constexpr KDTreeNodeLeaf
makeLeafNode(std::size_t permutation_index, Point point) noexcept
{
  return KDTreeNodeLeaf(static_cast<int>(permutation_index), std::move(point));
}

struct KDTreeNodeInternal : public KDTreeNode
{
  KOKKOS_DEFAULTED_FUNCTION constexpr KDTreeNodeInternal() = default;

  int left_child = -1;
  int right_child = -1;
  AAPlane plane;
};

} // namespace Details
} // namespace ArborX

#endif
