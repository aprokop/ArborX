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
#ifndef ARBORX_DETAILS_TREE_TRAVERSAL_HPP
#define ARBORX_DETAILS_TREE_TRAVERSAL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>
#include <ArborX_Predicates.hpp>

namespace ArborX
{
namespace Details
{

template <typename BVH, typename Predicates, typename Callback, typename Tag>
struct TreeTraversal
{
};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, SpatialPredicateTag>
{
  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = AccessTraits<Predicates, PredicatesTag>;

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:spatial_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      Kokkos::parallel_for(ARBORX_MARK_REGION("BVH:spatial_queries"),
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    if (predicate(bvh_.getBoundingVolume(bvh_.getRootIndex())))
    {
      callback_(predicate, 0);
    }
  }

  KOKKOS_FUNCTION void operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    constexpr int INVALID_INDEX = -1;

    int stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = INVALID_INDEX;

    int node_index = bvh_.getRootIndex();
    do
    {
      auto const *node = bvh_.getNodePtr(node_index);

      int child_left_index = node->children.first;
      int child_right_index = node->children.second;

      bool overlap_left = predicate(bvh_.getBoundingVolume(child_left_index));
      bool overlap_right = predicate(bvh_.getBoundingVolume(child_right_index));

      bool traverse_left = false;
      if (overlap_left)
      {
        auto const *child_left = bvh_.getNodePtr(child_left_index);
        if (child_left->isLeaf())
          callback_(predicate, child_left->getLeafPermutationIndex());
        else
          traverse_left = true;
      }

      bool traverse_right = false;
      if (overlap_right)
      {
        auto const *child_right = bvh_.getNodePtr(child_right_index);
        if (child_right->isLeaf())
          callback_(predicate, child_right->getLeafPermutationIndex());
        else
          traverse_right = true;
      }

      if (!traverse_left && !traverse_right)
      {
        node_index = *--stack_ptr;
      }
      else
      {
        node_index = traverse_left ? child_left_index : child_right_index;
        if (traverse_left && traverse_right)
          *stack_ptr++ = child_right_index;
      }
    } while (node_index != INVALID_INDEX);
  }
};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, NearestPredicateTag>
{
  using MemorySpace = typename BVH::memory_space;

  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = AccessTraits<Predicates, PredicatesTag>;

  using Buffer = Kokkos::View<Kokkos::pair<int, float> *, MemorySpace>;
  using Offset = Kokkos::View<int *, MemorySpace>;
  struct BufferProvider
  {
    Buffer buffer_;
    Offset offset_;

    KOKKOS_FUNCTION auto operator()(int i) const
    {
      auto const *offset_ptr = &offset_(i);
      return Kokkos::subview(buffer_,
                             Kokkos::make_pair(*offset_ptr, *(offset_ptr + 1)));
    }
  };

  BufferProvider buffer_;

  template <typename ExecutionSpace>
  void allocateBuffer(ExecutionSpace const &space)
  {
    auto const n_queries = Access::size(predicates_);

    Offset offset(Kokkos::ViewAllocateWithoutInitializing("offset"),
                  n_queries + 1);
    // NOTE workaround to avoid implicit capture of *this
    auto const &predicates = predicates_;
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
    exclusivePrefixSum(space, offset);
    int const buffer_size = lastElement(offset);
    // Allocate buffer over which to perform heap operations in
    // TreeTraversal::nearestQuery() to store nearest leaf nodes found so far.
    // It is not possible to anticipate how much memory to allocate since the
    // number of nearest neighbors k is only known at runtime.

    Buffer buffer(Kokkos::ViewAllocateWithoutInitializing("buffer"),
                  buffer_size);
    buffer_ = BufferProvider{buffer, offset};
  }

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:nearest_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      allocateBuffer(space);

      Kokkos::parallel_for(ARBORX_MARK_REGION("BVH:nearest_queries"),
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION int operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](int node_index) {
      return Details::distance(geometry, bvh.getBoundingVolume(node_index));
    };

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    callback_(predicate, 0, distance(bvh_.getRootIndex()));
    return 1;
  }

  KOKKOS_FUNCTION int operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](int node_index) {
      return Details::distance(geometry, bvh.getBoundingVolume(node_index));
    };
    auto const buffer = buffer_(queryIndex);

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

    using PairIndexDistance = Kokkos::pair<int, float>;
    static_assert(
        std::is_same<typename decltype(buffer)::value_type,
                     PairIndexDistance>::value,
        "Type of the elements stored in the buffer passed as argument to "
        "TreeTraversal::nearestQuery is not right");
    struct CompareDistance
    {
      KOKKOS_INLINE_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                             PairIndexDistance const &rhs) const
      {
        return lhs.second < rhs.second;
      }
    };
    // Use a priority queue for convenience to store the results and
    // preserve the heap structure internally at all time.  There is no
    // memory allocation, elements are stored in the buffer passed as an
    // argument. The farthest leaf node is on top.
    assert(k == (int)buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    constexpr int INVALID_INDEX = -1;

    int stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = INVALID_INDEX;
#if !defined(__CUDA_ARCH__)
    float stack_distance[64];
    auto *stack_distance_ptr = stack_distance;
    *stack_distance_ptr++ = 0.f;
#endif

    int node_index = bvh_.getRootIndex();
    int child_left_index = INVALID_INDEX;
    int child_right_index = INVALID_INDEX;

    float distance_left = 0.f;
    float distance_right = 0.f;
    float distance_node = 0.f;

    do
    {
      bool traverse_left = false;
      bool traverse_right = false;

      auto const *node = bvh_.getNodePtr(node_index);

      if (distance_node < radius)
      {
        // Insert children into the stack and make sure that the
        // closest one ends on top.
        child_left_index = node->children.first;
        child_right_index = node->children.second;

        distance_left = distance(child_left_index);
        distance_right = distance(child_right_index);

        if (distance_left < radius)
        {
          auto const *child_left = bvh_.getNodePtr(child_left_index);
          if (child_left->isLeaf())
          {
            auto leaf_pair = Kokkos::make_pair(
                child_left->getLeafPermutationIndex(), distance_left);
            if ((int)heap.size() < k)
              heap.push(leaf_pair);
            else
              heap.popPush(leaf_pair);
            if ((int)heap.size() == k)
              radius = heap.top().second;

            traverse_left = false;
          }
          else
          {
            traverse_left = true;
          }
        }

        // Note: radius may have been already updated here from the left child
        if (distance_right < radius)
        {
          auto const *child_right = bvh_.getNodePtr(child_right_index);
          if (child_right->isLeaf())
          {
            auto leaf_pair = Kokkos::make_pair(
                child_right->getLeafPermutationIndex(), distance_right);
            if ((int)heap.size() < k)
              heap.push(leaf_pair);
            else
              heap.popPush(leaf_pair);
            if ((int)heap.size() == k)
              radius = heap.top().second;

            traverse_right = false;
          }
          else
          {
            traverse_right = true;
          }
        }
      }

      if (!traverse_left && !traverse_right)
      {
        node_index = *--stack_ptr;
#if defined(__CUDA_ARCH__)
        if (node_index != INVALID_INDEX)
        {
          // This is a theoretically unnecessary duplication of distance
          // calculation for stack nodes. However, for Cuda it's better than
          // than putting the distances in stack.
          distance_node = distance(node_index);
        }
#else
        distance_node = *--stack_distance_ptr;
#endif
      }
      else
      {
        bool going_left = (traverse_left && (distance_left <= distance_right ||
                                             !traverse_right));
        node_index = (going_left ? child_left_index : child_right_index);
        distance_node = (going_left ? distance_left : distance_right);
        if (traverse_left && traverse_right)
        {
          *stack_ptr++ = (going_left ? child_right_index : child_left_index);
#if !defined(__CUDA_ARCH__)
          *stack_distance_ptr++ = (going_left ? distance_right : distance_left);
#endif
        }
      }
    } while (node_index != INVALID_INDEX);

    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap(heap.data(), heap.data() + heap.size(), heap.valueComp());
    for (decltype(heap.size()) i = 0; i < heap.size(); ++i)
    {
      int const leaf_index = (heap.data() + i)->first;
      auto const leaf_distance = (heap.data() + i)->second;
      callback_(predicate, leaf_index, leaf_distance);
    }
    return heap.size();
  }
};

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
void traverse(ExecutionSpace const &space, BVH const &bvh,
              Predicates const &predicates, Callback const &callback)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;
  TreeTraversal<BVH, Predicates, Callback, Tag>(space, bvh, predicates,
                                                callback);
}

} // namespace Details
} // namespace ArborX

#endif
