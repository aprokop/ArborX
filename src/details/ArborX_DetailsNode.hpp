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

#ifndef ARBORX_NODE_HPP
#define ARBORX_NODE_HPP

#include <ArborX_Box.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <climits> // INT_MIN
#include <utility> // std::move

namespace ArborX
{
namespace Details
{

constexpr int ROPE_SENTINEL = -1;

template <typename BoundingVolume>
struct LeafNodeWithLeftChildAndRope
{
  unsigned permutation_index = UINT_MAX;
  int rope = ROPE_SENTINEL;
  BoundingVolume bounding_volume;
  using bounding_volume_type = BoundingVolume;
};

template <typename BoundingVolume>
struct InternalNodeWithLeftChildAndRope
{
  // Right child is the rope of the left child
  int left_child = INT_MIN;
  int rope = ROPE_SENTINEL;
  BoundingVolume bounding_volume;
  using bounding_volume_type = BoundingVolume;
};

template <typename Value>
KOKKOS_INLINE_FUNCTION constexpr LeafNodeWithLeftChildAndRope<Value>
makeLeafNode(unsigned permutation_index, Value value) noexcept
{
  return {permutation_index, ROPE_SENTINEL, std::move(value)};
}
} // namespace Details
} // namespace ArborX

#endif
