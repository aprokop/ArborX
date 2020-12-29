/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
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

#include <Kokkos_Pair.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

// ------------------------------------
struct NodeWithTwoChildrenTag
{
};

struct NodeWithTwoChildrenLeaf
{
  using Tag = NodeWithTwoChildrenTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithTwoChildrenLeaf() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept { return true; }

  KOKKOS_INLINE_FUNCTION constexpr std::size_t getLeafPermutationIndex() const
      noexcept
  {
    return permutation_index;
  }

  int permutation_index = -1;
  int unused = -1;
  Box bounding_box;
};

KOKKOS_INLINE_FUNCTION constexpr NodeWithTwoChildrenLeaf
makeLeafNode(NodeWithTwoChildrenTag, std::size_t permutation_index,
             Box box) noexcept
{
  return {static_cast<int>(permutation_index), -1, std::move(box)};
}

struct NodeWithTwoChildrenInternal
{
  using Tag = NodeWithTwoChildrenTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithTwoChildrenInternal() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept
  {
    return false;
  }

  int left_child = -1;
  int right_child = -1;
  Box bounding_box;
};

// ------------------------------------

struct NodeWithLeftChildAndRopeTag
{
};

int constexpr ROPE_SENTINEL = -1;

struct NodeWithLeftChildAndRopeLeaf
{
  using Tag = NodeWithLeftChildAndRopeTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithLeftChildAndRopeLeaf() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept { return true; }

  KOKKOS_INLINE_FUNCTION constexpr std::size_t getLeafPermutationIndex() const
      noexcept
  {
    return permutation_index;
  }

  int permutation_index = -1;

  // An interesting property to remember: a right child is always the rope of
  // the left child.
  int rope = ROPE_SENTINEL;

  Point bounding_box;
};

KOKKOS_INLINE_FUNCTION constexpr NodeWithLeftChildAndRopeLeaf
makeLeafNode(NodeWithLeftChildAndRopeTag, std::size_t permutation_index,
             Point point) noexcept
{
  return {static_cast<int>(permutation_index), ROPE_SENTINEL, std::move(point)};
}

struct NodeWithLeftChildAndRopeInternal
{
  using Tag = NodeWithLeftChildAndRopeTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithLeftChildAndRopeInternal() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept
  {
    return false;
  }

  int left_child = -1;

  // An interesting property to remember: a right child is always the rope of
  // the left child.
  int rope = ROPE_SENTINEL;

  Box bounding_box;
};

} // namespace Details
} // namespace ArborX

#endif
