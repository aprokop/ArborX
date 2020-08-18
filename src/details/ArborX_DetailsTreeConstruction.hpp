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

#ifndef ARBORX_DETAILS_TREE_CONSTRUCTION_HPP
#define ARBORX_DETAILS_TREE_CONSTRUCTION_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsMortonCode.hpp> // morton3D
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{
namespace TreeConstruction
{

template <typename Primitives>
class CalculateBoundingBoxOfTheSceneFunctor
{
public:
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  CalculateBoundingBoxOfTheSceneFunctor(Primitives const &primitives)
      : _primitives(primitives)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void init(Box &box) const { box = Box(); }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i, Box &box) const
  {
    expand(box, Access::get(_primitives, i));
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile Box &dst, volatile Box const &src) const
  {
    expand(dst, src);
  }

private:
  Primitives _primitives;
};

template <typename ExecutionSpace, typename Primitives>
inline void calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                           Primitives const &primitives,
                                           Box &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_reduce(
      ARBORX_MARK_REGION("calculate_bounding_box_of_the_scene"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      CalculateBoundingBoxOfTheSceneFunctor<Primitives>(primitives),
      scene_bounding_box);
}

template <typename ExecutionSpace, typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(BoxTag, ExecutionSpace const &space,
                                      Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(ARBORX_MARK_REGION("assign_morton_codes"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         Point xyz;
                         centroid(Access::get(primitives, i), xyz);
                         translateAndScale(xyz, xyz, scene_bounding_box);
                         morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
                       });
}

template <typename ExecutionSpace, typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(PointTag, ExecutionSpace const &space,
                                      Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("assign_morton_codes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        Point xyz;
        translateAndScale(Access::get(primitives, i), xyz, scene_bounding_box);
        morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
      });
}

template <typename ExecutionSpace, typename Primitives,
          typename... MortonCodesViewProperties>
inline void assignMortonCodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
    Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(morton_codes.extent(0) == n);

  using Tag = typename AccessTraitsHelper<Access>::tag;
  assignMortonCodesDispatch(Tag{}, space, primitives, morton_codes,
                            scene_bounding_box);
}

template <typename ExecutionSpace, typename Primitives, typename Indices,
          typename Nodes>
inline void initializeLeafNodesDispatch(BoxTag, ExecutionSpace const &space,
                                        Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) =
            makeLeafNode(permutation_indices(i),
                         Access::get(primitives, permutation_indices(i)));
      });
}

template <typename ExecutionSpace, typename Primitives, typename Indices,
          typename Nodes>
inline void initializeLeafNodesDispatch(PointTag, ExecutionSpace const &space,
                                        Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) =
            makeLeafNode(permutation_indices(i),
                         {Access::get(primitives, permutation_indices(i)),
                          Access::get(primitives, permutation_indices(i))});
      });
}

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... LeafNodesViewProperties>
inline void initializeLeafNodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(permutation_indices.extent(0) == n);
  ARBORX_ASSERT(leaf_nodes.extent(0) == n);

  using Tag = typename AccessTraitsHelper<Access>::tag;
  initializeLeafNodesDispatch(Tag{}, space, primitives, permutation_indices,
                              leaf_nodes);
}

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... LeafNodesViewProperties>
inline void initializeLeafNodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
{
  initializeLeafNodes(
      space, primitives,
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>{
          permutation_indices},
      leaf_nodes);
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

struct LeftChildLevelsTag
{
};

template <typename MemorySpace>
class GenerateHierarchyFunctor
{
public:
  template <typename ExecutionSpace, typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties,
            typename... LeftChildLevelsViewProperties>
  GenerateHierarchyFunctor(
      ExecutionSpace const &space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int *, LeftChildLevelsViewProperties...> left_child_levels)
      : _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _left_child_levels(left_child_levels)
      , _ranges(Kokkos::ViewAllocateWithoutInitializing("ranges"),
                internal_nodes.extent(0))
      , _num_internal_nodes(_internal_nodes.extent_int(0))
  {
    Kokkos::deep_copy(space, _ranges, UNTOUCHED_NODE);
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

  KOKKOS_FUNCTION Node *getNodePtr(int i) const
  {
    int const n = _num_internal_nodes;
    return (i < n ? &(_internal_nodes(i))
                  : const_cast<Node *>(&(_leaf_nodes(i - n))));
  }

  KOKKOS_FUNCTION int calculateRope(int range_right, int delta_right) const
  {
    int rope;
    if (range_right != _num_internal_nodes)
    {
      // The way Karras indices constructed, the rope is going to be the right
      // child of the first internal that we are in the left subtree of. The
      // determination of whether that node is internal or leaf requires an
      // additional delta() evaluation.
      rope = range_right + 1;
      if (delta_right < delta(range_right + 1))
        rope += _num_internal_nodes;
    }
    else
    {
      // The node is on the right-most path in the tree. The only reason we
      // need to set it is because nodes may have been allocated without
      // initializing.
      rope = ROPE_SENTINEL;
    }
    return rope;
  }

  KOKKOS_FUNCTION void operator()(LeftChildLevelsTag, int i) const
  {
    // For thorough description, see the operator()

    auto const leaf_nodes_shift = _num_internal_nodes;

    // For a leaf node, the range is just one index
    int range_left = i - leaf_nodes_shift;
    int range_right = range_left;

    int delta_left = delta(range_left - 1);
    int delta_right = delta(range_right);

    // Walk toward the root and count
    _left_child_levels(i - leaf_nodes_shift) = 0;
    do
    {
      bool const is_left_child = delta_right < delta_left;

      int apetrei_parent;
      if (is_left_child)
      {
        apetrei_parent = range_right;

        range_right = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_left);

        if (range_right == UNTOUCHED_NODE)
          break;

        delta_right = delta(range_right);
      }
      else
      {
        apetrei_parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_right);

        if (range_left == UNTOUCHED_NODE)
          break;

        delta_left = delta(range_left - 1);
      }

      _left_child_levels(range_left)++;

      i = apetrei_parent;

    } while (range_right - range_left < _num_internal_nodes);
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const leaf_nodes_shift = _num_internal_nodes;

    // For a leaf node, the range is just one index
    int range_left = i - leaf_nodes_shift;
    int range_right = range_left;

    int delta_left = delta(range_left - 1);
    int delta_right = delta(range_right);

    Node *leaf_node = getNodePtr(i);
    Box bbox = leaf_node->bounding_box;
    leaf_node->rope = calculateRope(range_right, delta_right);

    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    do
    {
      // Determine whether this node is left or right child of its parent
      bool const is_left_child = delta_right < delta_left;

      int apetrei_parent;
      if (is_left_child)
      {
        // The main benefit of the Apetrei index (which is also called a split
        // in the Karras algorithm) is that each child can compute it based
        // just on the child's range. This is different from a Karras index,
        // where the index can only be computed based on the range of the
        // parent, and thus requires knowing the ranges of both children.
        apetrei_parent = range_right;

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

        delta_right = delta(range_right);
      }
      else
      {
        // The comments for this clause are identical to the ones above (in the
        // if clause), and thus ommitted for brevity.
        apetrei_parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_right);

        if (range_left == UNTOUCHED_NODE)
          break;

        delta_left = delta(range_left - 1);
      }

      // Reordering is based on the left range and the precomputed left child
      // levels inclusive sum. Each time we move up a level, left_child_levels
      // for the left is updated.
      int const reordered_parent = --_left_child_levels(range_left);

      // This is slightly convoluted due to the fact that the indices of leaf
      // nodes have to be shifted. The determination whether the other child
      // is a leaf node depends on the position of the split (which is
      // apetrei index) to the range boundary.
      int left_child =
          (apetrei_parent == range_left ? apetrei_parent + leaf_nodes_shift
                                        : reordered_parent + 1);
      int right_child = (apetrei_parent + 1 == range_right
                             ? apetrei_parent + 1 + leaf_nodes_shift
                             : _left_child_levels(apetrei_parent + 1));

      if (is_left_child)
        expand(bbox, getNodePtr(right_child)->bounding_box);
      else
        expand(bbox, getNodePtr(left_child)->bounding_box);

      auto *parent_node = getNodePtr(reordered_parent);
      parent_node->left_child = left_child;
      parent_node->bounding_box = bbox;

      // Internal nodes are not initialized, so we need to do that here. If
      // range_right == n, this means the node is on the right-most path from
      // the route, and all ropes lead nowhere. For all other internal nodes,
      // temporarily store right child index to shorcut later rope setting
      // traversal.
      parent_node->rope =
          (range_right != _num_internal_nodes ? right_child : -1);

      // Update ropes of all right-most children in the left subtree
      int next_right_child = parent_node->left_child;
      Node *child;
      do
      {
        child = getNodePtr(next_right_child);
        next_right_child = child->rope;
        child->rope = right_child;
      } while (!child->isLeaf());

      i = apetrei_parent;

    } while (range_right - range_left < _num_internal_nodes);
  }

private:
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<Node *, MemorySpace> _leaf_nodes;
  Kokkos::View<Node *, MemorySpace> _internal_nodes;
  Kokkos::View<int *, MemorySpace> _left_child_levels;
  Kokkos::View<int *, MemorySpace> _ranges;
  int _num_internal_nodes;
};

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  using MemorySpace = typename decltype(internal_nodes)::memory_space;
  auto const n_internal_nodes = internal_nodes.extent(0);

  Kokkos::View<int *, MemorySpace> left_child_levels(
      Kokkos::ViewAllocateWithoutInitializing("left_child_levels"),
      n_internal_nodes + 2);

  Kokkos::parallel_for(ARBORX_MARK_REGION("compute_left_child_levels"),
                       Kokkos::RangePolicy<ExecutionSpace, LeftChildLevelsTag>(
                           space, n_internal_nodes, 2 * n_internal_nodes + 1),
                       GenerateHierarchyFunctor<MemorySpace>(
                           space, sorted_morton_codes, leaf_nodes,
                           internal_nodes, left_child_levels));

  inclusivePrefixSum(space, left_child_levels);

  Kokkos::parallel_for(ARBORX_MARK_REGION("generate_hierarchy"),
                       Kokkos::RangePolicy<ExecutionSpace>(
                           space, n_internal_nodes, 2 * n_internal_nodes + 1),
                       GenerateHierarchyFunctor<MemorySpace>(
                           space, sorted_morton_codes, leaf_nodes,
                           internal_nodes, left_child_levels));
}

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  generateHierarchy(
      space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      leaf_nodes, internal_nodes);
}

} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
