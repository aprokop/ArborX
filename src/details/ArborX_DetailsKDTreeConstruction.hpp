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

#ifndef ARBORX_DETAILS_KDTREE_CONSTRUCTION_HPP
#define ARBORX_DETAILS_KDTREE_CONSTRUCTION_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsKDTreeNode.hpp> // makeLeafNode
#include <ArborX_DetailsKokkosExt.hpp>  //clz
#include <ArborX_DetailsMortonCode.hpp> // compactBits
#include <ArborX_DetailsTags.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{
namespace KDTreeConstruction
{

KOKKOS_INLINE_FUNCTION void
computeReferencePlane(unsigned int index1, unsigned int index2, AAPlane &plane)
{
  int common_prefix_length = KokkosExt::clz(index1 ^ index2);

  if (common_prefix_length == 32)
  {
    // Invalidate a plane
    plane.axis() = -1;
    return;
  }

  plane.axis() = (common_prefix_length + 1) % 3;

  unsigned loc = index1 & ~(((1 << (32 - common_prefix_length)) - 1));
  loc |= (1 << (31 - common_prefix_length));
  loc = compactBits(loc >> (2 - plane.axis()));

  plane.location() = loc / 1024.;
}

namespace
{
// Ideally, this would be
//     static int constexpr UNTOUCHED_NODE = -1;
// inside the GenerateHierachyFunctor class. But prior to C++17, this would
// require to also have a definition outside of the class as it is odr-used.
// This is a workaround.
int constexpr UNTOUCHED_NODE = -1;
} // namespace

template <typename Primitives, typename MemorySpace>
class GenerateHierarchy
{
public:
  template <typename ExecutionSpace,
            typename... PermutationIndicesViewProperties,
            typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties>
  GenerateHierarchy(
      ExecutionSpace const &space, Primitives const &primitives,
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>
          permutation_indices,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<KDTreeNodeLeaf *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<KDTreeNodeInternal *, InternalNodesViewProperties...>
          internal_nodes,
      Box const &bounds)
      : _primitives(primitives)
      , _permutation_indices(permutation_indices)
      , _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _bounds(bounds)
      , _ranges(Kokkos::ViewAllocateWithoutInitializing(
                    "ArborX::KDTree::KDTree::ranges"),
                internal_nodes.extent(0))
      , _num_internal_nodes(_internal_nodes.extent_int(0))
  {
    Kokkos::deep_copy(space, _ranges, UNTOUCHED_NODE);

    Kokkos::parallel_for(
        "ArborX::KDTreeConstruction::generate_hierarchy",
        Kokkos::RangePolicy<ExecutionSpace>(space, _num_internal_nodes,
                                            2 * _num_internal_nodes + 1),
        *this);
  }

  KOKKOS_FUNCTION void computePlane(int i, int j, AAPlane &plane) const
  {
    computeReferencePlane(_sorted_morton_codes(i), _sorted_morton_codes(j),
                          plane);

    if (!isValid(plane))
      return;

    // translate back to absolute coordinates
    float a = _bounds.minCorner()[plane.axis()];
    float b = _bounds.maxCorner()[plane.axis()];
    plane.location() = a * (1. - plane.location()) + plane.location() * b;
  }

  KOKKOS_FUNCTION
  int delta(int const i) const
  {
    // Per Apetrei:
    //   Because we already know where the highest differing bit is for each
    //   internal node, the delta function basically represents a distance
    //   metric between two keys. Unlike the delta used by Karras, we are
    //   interested in the index of the highest differing bit and not the length
    //   of the common prefix. In practice, logical xor can be used instead of
    //   finding the index of the highest differing bit as we can compare the
    //   numbers. The higher the index of the differing bit, the larger the
    //   number.

    // This check is here simply to avoid code complications in the main
    // operator
    if (i < 0 || i >= _num_internal_nodes)
      return INT_MAX;

    // The Apetrei's paper does not mention dealing with duplicate indices. We
    // follow the original Karras idea in this situation:
    //   The case of duplicate Morton codes has to be handled explicitly, since
    //   our construction algorithm relies on the keys being unique. We
    //   accomplish this by augmenting each key with a bit representation of
    //   its index, i.e. k_i = k_i <+> i, where <+> indicates string
    //   concatenation.
    // In this case, if the Morton indices are the same, we want to compare is.
    // We also want the result in this situation to always be less than any
    // Morton comparison. Thus, we add INT_MIN to it.
    // We also avoid if/else statement by doing a "x + !x*<blah>" trick.
    auto x = _sorted_morton_codes(i) ^ _sorted_morton_codes(i + 1);
    return x + (!x) * (INT_MIN + (i ^ (i + 1)));
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const leaf_nodes_shift = _num_internal_nodes;

    // Index in the orginal order primitives were given in.
    auto const original_index = _permutation_indices(i - leaf_nodes_shift);

    // Initialize leaf node
    using Access = AccessTraits<Primitives, PrimitivesTag>;
    auto *leaf_node = &(_leaf_nodes(i - leaf_nodes_shift));
    *leaf_node =
        makeLeafNode(original_index, Access::get(_primitives, original_index));

    // For a leaf node, the range is just one index
    int range_left = i - leaf_nodes_shift;
    int range_right = range_left;

    int delta_left = delta(range_left - 1);
    int delta_right = delta(range_right);

    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    do
    {
      // Determine whether this node is left or right child of its parent
      bool const is_left_child = delta_right < delta_left;

      int left_child;
      int right_child;
      if (is_left_child)
      {
        // The main benefit of the Apetrei index (which is also called a split
        // in the Karras algorithm) is that each child can compute it based
        // just on the child's range. This is different from a Karras index,
        // where the index can only be computed based on the range of the
        // parent, and thus requires knowing the ranges of both children.
        int const apetrei_parent = range_right;

        // The range of the parent is the union of the ranges of children. Each
        // child updates one of these range values, the farthest from the
        // split. The first thread up stores the updated range value (which
        // also serves as a flag). The second thread up finishes constructing
        // the full parent range.
        range_right = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_left);

        // Use an atomic flag per internal node to terminate the first
        // thread that enters it, while letting the second one through.
        // This ensures that every node gets processed only once, and not
        // before both of its children are processed.
        if (range_right == UNTOUCHED_NODE)
          break;

        // This is slightly convoluted due to the fact that the indices of leaf
        // nodes have to be shifted. The determination whether the other child
        // is a leaf node depends on the position of the split (which is
        // apetrei index) to the range boundary.
        left_child = i;
        right_child = apetrei_parent + 1;
        if (right_child == range_right)
          right_child += leaf_nodes_shift;

        delta_right = delta(range_right);
      }
      else
      {
        // The comments for this clause are identical to the ones above (in the
        // if clause), and thus ommitted for brevity.

        int const apetrei_parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_right);
        if (range_left == UNTOUCHED_NODE)
          break;

        left_child = apetrei_parent;
        if (left_child == range_left)
          left_child += leaf_nodes_shift;
        right_child = i;

        delta_left = delta(range_left - 1);
      }

      // Having the full range for the parent, we can compute the Karras index.
      int const karras_parent =
          delta_right < delta_left ? range_right : range_left;

      auto *parent_node = &(_internal_nodes(karras_parent));
      parent_node->left_child = left_child;
      parent_node->right_child = right_child;

      computePlane(range_left, range_right, parent_node->plane);

      i = karras_parent;

    } while (i != 0);
  }

private:
  Primitives _primitives;
  Kokkos::View<unsigned int const *, MemorySpace> _permutation_indices;
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<KDTreeNodeLeaf *, MemorySpace> _leaf_nodes;
  Kokkos::View<KDTreeNodeInternal *, MemorySpace> _internal_nodes;
  Box _bounds;
  Kokkos::View<int *, MemorySpace> _ranges;
  int _num_internal_nodes;
};

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<KDTreeNodeLeaf *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<KDTreeNodeInternal *, InternalNodesViewProperties...>
        internal_nodes,
    Box const &bounds)
{
  using ConstPermutationIndices =
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>;
  using ConstMortonCodes =
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>;

  using MemorySpace = typename decltype(internal_nodes)::memory_space;

  GenerateHierarchy<Primitives, MemorySpace>(
      space, primitives, ConstPermutationIndices(permutation_indices),
      ConstMortonCodes(sorted_morton_codes), leaf_nodes, internal_nodes,
      bounds);
}

} // namespace KDTreeConstruction
} // namespace Details
} // namespace ArborX

#endif
