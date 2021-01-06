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

#ifndef ARBORX_KDTREE_HPP
#define ARBORX_KDTREE_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsKDTreeConstruction.hpp>
#include <ArborX_DetailsKDTreeNode.hpp>
#include <ArborX_DetailsKDTreeTraversal.hpp>
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_DetailsPermutedData.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_TraversalPolicy.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename MemorySpace>
class KDTree
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename MemorySpace::size_type;

  KDTree() = default; // build an empty tree

  template <typename ExecutionSpace, typename Primitives>
  KDTree(ExecutionSpace const &space, Primitives const &primitives);

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  Box bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback,
             Experimental::TraversalPolicy const &policy =
                 Experimental::TraversalPolicy()) const;

private:
  template <typename Tree, typename Predicates, typename Callback, typename Tag>
  friend struct Details::KDTreeTraversal;

  using internal_node_type = Details::KDTreeNodeInternal;
  using leaf_node_type = Details::KDTreeNodeLeaf;

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

  KOKKOS_FUNCTION
  bool isLeaf(size_t i) const { return i >= _leaf_nodes_shift; }

  KOKKOS_FUNCTION
  Details::KDTreeNode const *getNodePtr(int i) const
  {
    if (isLeaf(i))
      return static_cast<Details::KDTreeNode const *>(
          &(_leaf_nodes(i - _leaf_nodes_shift)));
    else
      return static_cast<Details::KDTreeNode const *>(&(_internal_nodes(i)));
  }

  size_t _size;
  size_t _leaf_nodes_shift;
  Box _bounds;
  Kokkos::View<leaf_node_type *, MemorySpace> _leaf_nodes;
  Kokkos::View<internal_node_type *, MemorySpace> _internal_nodes;
};

template <typename MemorySpace>
template <typename ExecutionSpace, typename Primitives>
KDTree<MemorySpace>::KDTree(ExecutionSpace const &space,
                            Primitives const &primitives)
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _leaf_nodes_shift(_size > 0 ? _size - 1 : 0)
    , _leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing("ArborX::KDTree::leaf_nodes"),
          _size)
    , _internal_nodes(Kokkos::ViewAllocateWithoutInitializing(
                          "ArborX::KDTree::internal_nodes"),
                      _size > 0 ? _size - 1 : 0)
{
  Kokkos::Profiling::pushRegion("ArborX::KDTree::KDTree");

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
      "ArborX::KDTree::KDTree::calculate_scene_bounding_box");

  // determine the bounding box of the scene
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            _bounds);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    // FIXME!!!
    // Details::TreeConstruction::initializeSingleLeafNode(
    // space, primitives, _internal_and_leaf_nodes);
    Kokkos::Profiling::popRegion();
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX::KDTree::KDTree::assign_morton_codes");

  // calculate Morton codes of all objects
  Kokkos::View<unsigned int *, MemorySpace> morton_indices(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::KDTree::KDTree::morton"),
      size());
  Details::TreeConstruction::assignMortonCodes(space, primitives,
                                               morton_indices, _bounds);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::KDTree::KDTree::sort_morton_codes");

  // compute the ordering of primitives along Z-order space-filling curve
  auto permutation_indices = Details::sortObjects(space, morton_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::KDTree::KDTree::generate_hierarchy");

  // generate bounding volume hierarchy
  Details::KDTreeConstruction::generateHierarchy(
      space, primitives, permutation_indices, morton_indices, getLeafNodes(),
      getInternalNodes(), bounds());

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace>
template <typename ExecutionSpace, typename Predicates, typename Callback>
void KDTree<MemorySpace>::query(
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
    Details::KDtraverse(space, *this, PermutedPredicates{predicates, permute},
                        callback);
  }
  else
  {
    Details::KDtraverse(space, *this, predicates, callback);
  }

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
