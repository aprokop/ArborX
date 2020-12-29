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

#ifndef ARBORX_LINEAR_BVH_HPP
#define ARBORX_LINEAR_BVH_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPermutedData.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_DetailsTreeTraversal.hpp>
#include <ArborX_TraversalPolicy.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

namespace Details
{
template <typename DeviceType>
struct TreeVisualization;
template <typename BVH>
struct DistributedTreeNearestUtils;
} // namespace Details

template <typename MemorySpace, typename Enable = void>
class BoundingVolumeHierarchy
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = Box;

  BoundingVolumeHierarchy() = default; // build an empty tree

  template <typename ExecutionSpace, typename Primitives>
  BoundingVolumeHierarchy(ExecutionSpace const &space,
                          Primitives const &primitives);

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback,
             Experimental::TraversalPolicy const &policy =
                 Experimental::TraversalPolicy()) const;

  template <typename ExecutionSpace, typename Predicates,
            typename CallbackOrView, typename View, typename... Args>
  std::enable_if_t<Kokkos::is_view<std::decay_t<View>>{}>
  query(ExecutionSpace const &space, Predicates const &predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&... args) const
  {
    ArborX::query(*this, space, predicates,
                  std::forward<CallbackOrView>(callback_or_view),
                  std::forward<View>(view), std::forward<Args>(args)...);
  }

private:
  template <typename BVH, typename Predicates, typename Callback,
            typename /*Enable*/>
  friend struct Details::TreeTraversal;
  template <typename Tree, typename Primitives>
  friend class Details::TreeConstruction::GenerateHierarchy;
  template <typename DeviceType>
  friend struct Details::TreeVisualization;
  template <typename BVH>
  friend struct Details::DistributedTreeNearestUtils;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // Ropes based traversal is only used for CUDA, as it was found to be slower
  // than regular one for Power9 on Summit.  It is also used with HIP.
  using internal_node_type =
      std::conditional_t<std::is_same<MemorySpace,
#if defined(KOKKOS_ENABLE_CUDA)
                                      Kokkos::CudaSpace
#else
                                      Kokkos::Experimental::HIPSpace
#endif
                                      >{},
                         Details::NodeWithLeftChildAndRopeInternal,
                         Details::NodeWithTwoChildrenInternal>;
  using leaf_node_type =
      std::conditional_t<std::is_same<MemorySpace,
#if defined(KOKKOS_ENABLE_CUDA)
                                      Kokkos::CudaSpace
#else
                                      Kokkos::Experimental::HIPSpace
#endif
                                      >{},
                         Details::NodeWithLeftChildAndRopeLeaf,
                         Details::NodeWithTwoChildrenLeaf>;
#else
  using internal_node_type = Details::NodeWithTwoChildrenInternal;
  using leaf_node_type = Details::NodeWithTwoChildrenLeaf;
#endif

  KOKKOS_FUNCTION
  bool isLeaf(int node_index) const
  {
    return (size_t)(node_index + 1) >= size();
  }

  KOKKOS_FUNCTION
  auto const &getInternalNode(int node_index) const
  {
    assert(!isLeaf(node_index));
    return _internal_nodes(node_index);
  }

  KOKKOS_FUNCTION
  auto &getInternalNode(int node_index)
  {
    assert(!isLeaf(node_index));
    return _internal_nodes(node_index);
  }

  KOKKOS_FUNCTION
  auto const &getLeafNode(int node_index) const
  {
    assert(isLeaf(node_index));
    return _leaf_nodes(node_index - (size() - 1));
  }

  KOKKOS_FUNCTION
  auto &getLeafNode(int node_index)
  {
    assert(isLeaf(node_index));
    return _leaf_nodes(node_index - (size() - 1));
  }

  KOKKOS_FUNCTION constexpr std::size_t
  getLeafPermutationIndex(int node_index) const noexcept
  {
    return getLeafNode(node_index).getLeafPermutationIndex();
  }

  KOKKOS_FUNCTION
  int leftChildIndex(int node_index) const
  {
    return getInternalNode(node_index).left_child;
  }

  template <typename Tag = typename leaf_node_type::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, Details::NodeWithTwoChildrenTag>{}>
      makeLeafNode(int i, int original_index, int /*rope_index*/, Box box) const
  {
    assert(isLeaf(i));
    auto &leaf_node = _leaf_nodes(i - (size() - 1));
    leaf_node = Details::makeLeafNode(typename leaf_node_type::Tag{},
                                      original_index, box);
  }

  template <typename Tag = typename leaf_node_type::Tag>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<Tag, Details::NodeWithLeftChildAndRopeTag>{}>
  // makeLeafNode(int i, int original_index, int rope_index, Box box) const
  makeLeafNode(int i, int original_index, int rope_index, Point point) const
  {
    assert(isLeaf(i));
    auto &leaf_node = _leaf_nodes(i - (size() - 1));
    leaf_node = Details::makeLeafNode(typename leaf_node_type::Tag{},
                                      original_index, point);
    leaf_node.rope = rope_index;
  }

  template <typename Tag = typename internal_node_type::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, Details::NodeWithTwoChildrenTag>{}>
      makeInternalNode(int i, int child_left_index, int child_right_index,
                       int /*rope_index*/, Box box) const
  {
    assert(!isLeaf(i));
    auto &internal_node = _internal_nodes(i);
    internal_node.left_child = child_left_index;
    internal_node.right_child = child_right_index;
    internal_node.bounding_box = box;
  }

  template <typename Tag = typename internal_node_type::Tag>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<Tag, Details::NodeWithLeftChildAndRopeTag>{}>
  makeInternalNode(int i, int child_left_index, int /*child_right_index*/,
                   int rope_index, Box box) const
  {
    assert(!isLeaf(i));
    auto &internal_node = _internal_nodes(i);
    internal_node.left_child = child_left_index;
    internal_node.rope = rope_index;
    internal_node.bounding_box = box;
  }

  template <typename Tag = typename internal_node_type::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, Details::NodeWithTwoChildrenTag>{},
                       int>
      rightChildIndex(int node_index) const
  {
    return getInternalNode(node_index).right_child;
  }

  template <typename Tag = typename internal_node_type::Tag>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<Tag, Details::NodeWithLeftChildAndRopeTag>{}, int>
  rightChildIndex(int node_index) const
  {
    auto left_child_index = leftChildIndex(node_index);
    return getRope(left_child_index);
  }

  template <typename Tag = typename internal_node_type::Tag>
  KOKKOS_FUNCTION std::enable_if_t<
      std::is_same<Tag, Details::NodeWithLeftChildAndRopeTag>{}, int>
  getRope(int node_index) const
  {
    return (isLeaf(node_index) ? getLeafNode(node_index).rope
                               : getInternalNode(node_index).rope);
  }

  Kokkos::View<internal_node_type *, MemorySpace> getInternalNodes()
  {
    assert(!empty());
    return _internal_nodes;
  }

  Kokkos::View<leaf_node_type *, MemorySpace> getLeafNodes()
  {
    assert(!empty());
    return _leaf_nodes;
  }

  Kokkos::View<leaf_node_type const *, MemorySpace> getLeafNodes() const
  {
    assert(!empty());
    return _leaf_nodes;
  }

  KOKKOS_FUNCTION
  int getRootIndex() const { return 0; }

  KOKKOS_FUNCTION
  bounding_volume_type const &getBoundingVolume(int node_index) const
  {
    if (isLeaf(node_index))
    {
      auto &p = getLeafNode(node_index).bounding_box;
      return {p, p};
    }
    else
    {
      return getInternalNode(node_index).bounding_box;
    }
  }

  template <typename Predicate>
  KOKKOS_FUNCTION bool intersects(Predicate const &predicate, int i) const
  {
    if (isLeaf(i))
      return predicate(getLeafNode(i).bounding_box);
    else
      return predicate(getInternalNode(i).bounding_box);
  }

  KOKKOS_FUNCTION
  bounding_volume_type &getBoundingVolume(int node_index)
  {
    assert(!isLeaf(node_index));
    return getInternalNode(node_index).bounding_box;
  }

  size_t _size;
  bounding_volume_type _bounds;
  Kokkos::View<internal_node_type *, MemorySpace> _internal_nodes;
  Kokkos::View<leaf_node_type *, MemorySpace> _leaf_nodes;
};

template <typename DeviceType>
class BoundingVolumeHierarchy<
    DeviceType, std::enable_if_t<Kokkos::is_device<DeviceType>::value>>
    : public BoundingVolumeHierarchy<typename DeviceType::memory_space>
{
public:
  using device_type = DeviceType;

  // clang-format off
  [[deprecated("ArborX::BoundingVolumeHierarchy templated on a device type "
               "is deprecated, use it templated on a memory space instead.")]]
  BoundingVolumeHierarchy() = default;
  template <typename Primitives>
  [[deprecated("ArborX::BoundingVolumeHierarchy templated on a device type "
               "is deprecated, use it templated on a memory space instead.")]]
  BoundingVolumeHierarchy(Primitives const &primitives)
      : BoundingVolumeHierarchy<typename DeviceType::memory_space>(
            typename DeviceType::execution_space{}, primitives)
  {
  }
  // clang-format on
  template <typename FirstArgumentType, typename... Args>
  std::enable_if_t<!Kokkos::is_execution_space<FirstArgumentType>::value>
  query(FirstArgumentType &&arg1, Args &&... args) const
  {
    BoundingVolumeHierarchy<typename DeviceType::memory_space>::query(
        typename DeviceType::execution_space{},
        std::forward<FirstArgumentType>(arg1), std::forward<Args>(args)...);
  }

private:
  template <typename Tree, typename ExecutionSpace, typename Predicates,
            typename CallbackOrView, typename View, typename... Args>
  friend void ArborX::query(Tree const &tree, ExecutionSpace const &space,
                            Predicates const &predicates,
                            CallbackOrView &&callback_or_view, View &&view,
                            Args &&... args);

  template <typename FirstArgumentType, typename... Args>
  std::enable_if_t<Kokkos::is_execution_space<FirstArgumentType>::value>
  query(FirstArgumentType const &space, Args &&... args) const
  {
    BoundingVolumeHierarchy<typename DeviceType::memory_space>::query(
        space, std::forward<Args>(args)...);
  }
};

template <typename MemorySpace>
using BVH = BoundingVolumeHierarchy<MemorySpace>;

template <typename MemorySpace, typename Enable>
template <typename ExecutionSpace, typename Primitives>
BoundingVolumeHierarchy<MemorySpace, Enable>::BoundingVolumeHierarchy(
    ExecutionSpace const &space, Primitives const &primitives)
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _internal_nodes(Kokkos::ViewAllocateWithoutInitializing(
                          "ArborX::BVH::internal_nodes"),
                      _size > 0 ? _size - 1 : 0)

    , _leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing("ArborX::BVH::leaf_nodes"),
          _size > 0 ? _size : 0)
{
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH");

  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution space");

  if (empty())
  {
    Kokkos::Profiling::popRegion();
    return;
  }

  Kokkos::Profiling::pushRegion(
      "ArborX::BVH::BVH::calculate_scene_bounding_box");

  // determine the bounding box of the scene
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            _bounds);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    Details::TreeConstruction::initializeSingleLeafNode(space, primitives,
                                                        getLeafNodes());
    Kokkos::Profiling::popRegion();
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::assign_morton_codes");

  // calculate Morton codes of all objects
  Kokkos::View<unsigned int *, MemorySpace> morton_indices(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::BVH::BVH::morton"),
      size());
  Details::TreeConstruction::assignMortonCodes(space, primitives,
                                               morton_indices, _bounds);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::sort_morton_codes");

  // compute the ordering of primitives along Z-order space-filling curve
  auto permutation_indices = Details::sortObjects(space, morton_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::generate_hierarchy");

  // generate bounding volume hierarchy
  auto &tree = *this;
  Details::TreeConstruction::generateHierarchy(
      space, tree, primitives, permutation_indices, morton_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace, typename Enable>
template <typename ExecutionSpace, typename Predicates, typename Callback>
void BoundingVolumeHierarchy<MemorySpace, Enable>::query(
    ExecutionSpace const &space, Predicates const &predicates,
    Callback const &callback, Experimental::TraversalPolicy const &policy) const
{
  Details::check_valid_access_traits(PredicatesTag{}, predicates);

  using Access = AccessTraits<Predicates, Traits::PredicatesTag>;
  using Tag = typename Details::AccessTraitsHelper<Access>::tag;

  auto profiling_prefix =
      std::string("ArborX::BVH::query::") +
      (std::is_same<Tag, Details::SpatialPredicateTag>{} ? "spatial"
                                                         : "nearest");

  Kokkos::Profiling::pushRegion(profiling_prefix);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bounds(), predicates);
    Kokkos::Profiling::popRegion();

    using PermutedPredicates =
        Details::PermutedData<Predicates, decltype(permute)>;
    Details::traverse(space, *this, PermutedPredicates{predicates, permute},
                      callback);
  }
  else
  {
    Details::traverse(space, *this, predicates, callback);
  }

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
