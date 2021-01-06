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
#ifndef ARBORX_DETAILS_KDTREE_TRAVERSAL_HPP
#define ARBORX_DETAILS_KDTREE_TRAVERSAL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKDTreeNode.hpp>
#include <ArborX_DetailsNode.hpp> // ROPE_SENTINEL
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Predicates.hpp>

namespace ArborX
{
namespace Details
{

template <typename Tree, typename Predicates, typename Callback, typename Tag>
struct KDTreeTraversal
{
};

template <typename Tree, typename Predicates, typename Callback>
struct KDTreeTraversal<Tree, Predicates, Callback, SpatialPredicateTag>
{
  Tree _tree;
  Predicates _predicates;
  Callback _callback;

  using Access = AccessTraits<Predicates, PredicatesTag>;

  template <typename ExecutionSpace>
  KDTreeTraversal(ExecutionSpace const &space, Tree const &tree,
                  Predicates const &predicates, Callback const &callback)
      : _tree{tree}
      , _predicates{predicates}
      , _callback{callback}
  {
    if (_tree.empty())
    {
      // do nothing
    }
    else if (_tree.size() == 1)
    {
#if 0
      Kokkos::parallel_for(
      "ArborX::KDTreeTraversal::spatial::degenerated_one_leaf_tree",
      Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
      space, 0, Access::size(predicates)),
      *this);
#endif
    }
    else
    {
      Kokkos::parallel_for("ArborX::KDTreeTraversal::spatial",
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

#if 0
  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(_predicates, queryIndex);

    if (predicate(_tree.getBoundingVolume(_tree.getRoot())))
    {
      _callback(predicate, 0);
    }
  }
#endif

  // Stack-based traversal
  KOKKOS_FUNCTION void operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(_predicates, queryIndex);

    // This is optional optimization. It also imposes a requirement to implement
    // intersection with a Box.
    if (!predicate(_tree.bounds()))
      return;

    using Node = KDTreeNodeInternal;

    Node const *stack[64];
    Node const **stack_ptr = stack;
    *stack_ptr++ = nullptr;
    // Node const *node = _tree.getRoot();
    Node const *node =
        static_cast<KDTreeNodeInternal const *>(_tree.getNodePtr(0));
    do
    {
      KDTreeNode const *child_left = _tree.getNodePtr(node->left_child);
      KDTreeNode const *child_right = _tree.getNodePtr(node->right_child);

      auto r = (isValid(node->plane) ? predicate(node->plane)
                                     : HalfSpaceIntersection::BOTH);

      bool overlap_left = (r == HalfSpaceIntersection::LEFT ||
                           r == HalfSpaceIntersection::BOTH);
      bool overlap_right = (r == HalfSpaceIntersection::RIGHT ||
                            r == HalfSpaceIntersection::BOTH);

      if (overlap_left && _tree.isLeaf(node->left_child))
      {
        auto const *leaf_node = static_cast<KDTreeNodeLeaf const *>(child_left);
        if (predicate(leaf_node->point))
        {
          if (invoke_callback_and_check_early_exit(
                  _callback, predicate, leaf_node->getLeafPermutationIndex()))
            return;
        }
      }
      if (overlap_right && _tree.isLeaf(node->right_child))
      {
        auto const *leaf_node =
            static_cast<KDTreeNodeLeaf const *>(child_right);
        if (predicate(leaf_node->point))
        {
          if (invoke_callback_and_check_early_exit(
                  _callback, predicate, leaf_node->getLeafPermutationIndex()))
            return;
        }
      }

      bool traverse_left = (overlap_left && !_tree.isLeaf(node->left_child));
      bool traverse_right = (overlap_right && !_tree.isLeaf(node->right_child));

      if (!traverse_left && !traverse_right)
      {
        node = *--stack_ptr;
      }
      else
      {
        node = static_cast<KDTreeNodeInternal const *>(
            traverse_left ? child_left : child_right);
        if (traverse_left && traverse_right)
          *stack_ptr++ = static_cast<KDTreeNodeInternal const *>(child_right);
      }
    } while (node != nullptr);
  }
};

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Callback>
void KDtraverse(ExecutionSpace const &space, Tree const &tree,
                Predicates const &predicates, Callback const &callback)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;
  static_assert(std::is_same<Tag, SpatialPredicateTag>{}, "");
  KDTreeTraversal<Tree, Predicates, Callback, Tag>(space, tree, predicates,
                                                   callback);
}

} // namespace Details
} // namespace ArborX

#endif
