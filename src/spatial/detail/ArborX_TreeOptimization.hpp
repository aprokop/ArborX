/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_TREE_OPTIMIZATION_HPP
#define ARBORX_TREE_OPTIMIZATION_HPP

#include <ArborX_GeometryTraits.hpp>
#include <algorithms/ArborX_Expand.hpp>
#include <detail/ArborX_Node.hpp>
#include <detail/ArborX_TreeNodeLabeling.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace ArborX::Details
{

struct SurfaceArea
{
  // For a full-dimensional box, sum the measures of its 2*dim boundary faces.
  // For a collapsed box, return twice the product of its nonzero extents;
  // a point has cost zero. In 3D this preserves 2*a*b for a plane, gives 2*a
  // for a line, and avoids assigning every collinear subtree the same zero
  // cost. This extension is an optimization heuristic, not intrinsic boundary
  // area: a plane uses its area rather than its perimeter. Full-dimensional 1D
  // boxes retain the usual boundary measure of two endpoints.
  template <class BoundingVolume>
  KOKKOS_FUNCTION auto operator()(BoundingVolume const &volume) const
  {
    static_assert(GeometryTraits::is_box_v<BoundingVolume>,
                  "SurfaceArea: not implemented for this bounding volume");
    if constexpr (GeometryTraits::is_box_v<BoundingVolume>)
    {
      using Coordinate = GeometryTraits::coordinate_type_t<BoundingVolume>;
      constexpr int dim = GeometryTraits::dimension_v<BoundingVolume>;
      int active_dimensions = 0;
      for (int d = 0; d < dim; ++d)
        active_dimensions += volume.maxCorner()[d] != volume.minCorner()[d];
      if (active_dimensions == 0)
        return decltype(2 * Coordinate{}){0};

      if (active_dimensions < dim)
      {
        Coordinate measure = 1;
        for (int d = 0; d < dim; ++d)
        {
          auto const extent = volume.maxCorner()[d] - volume.minCorner()[d];
          // Exact zeros are omitted. A tolerance would change the objective
          // for thin but nondegenerate boxes and depend on coordinate scaling.
          if (extent != 0)
            measure *= extent;
        }
        return 2 * measure;
      }

      // Build the sum of face products incrementally, without division or an
      // extent array: after adding x, S' = S*x + P, where P is the product of
      // previous extents. The final full-dimensional product is unnecessary
      // and could overflow even when the surface area is representable.
      Coordinate area = 0;
      Coordinate product = 1;
      for (int d = 0; d < dim; ++d)
      {
        auto const extent = volume.maxCorner()[d] - volume.minCorner()[d];
        area = area * extent + product;
        if (d + 1 < dim)
          product *= extent;
      }
      return 2 * area;
    }
  }
};

namespace TreeOptimization
{
// Only the lower triangle is stored; the diagonal is never accessed.
KOKKOS_FUNCTION constexpr int pairIndex(int i, int j)
{
  return i > j ? i * (i - 1) / 2 + j : j * (j - 1) / 2 + i;
}

template <class LeafNode, class InternalNode, class IndexableGetter>
KOKKOS_FUNCTION auto getBoundingVolume(int node, int n, LeafNode const *leaves,
                                       InternalNode const *nodes,
                                       IndexableGetter const &indexable_getter)
{
  using BoundingVolume = typename InternalNode::bounding_volume_type;
  BoundingVolume volume;
  if (node < n)
    expand(volume, indexable_getter(leaves[node].value));
  else
    volume = nodes[node - n].bounding_volume;
  return volume;
}

template <int TreeletSize, class LeafNode, class InternalNode,
          class IndexableGetter, class CostFunction>
KOKKOS_FUNCTION void
restructure(int root, int n, LeafNode const *leaves, InternalNode *nodes,
            IndexableGetter const &indexable_getter, int *right_children,
            CostFunction const &cost_function)
{
  using BoundingVolume = typename InternalNode::bounding_volume_type;
  using Cost = std::decay_t<
      std::invoke_result_t<CostFunction const &, BoundingVolume const &>>;
  if (nodes[root - n].left_child < n && right_children[root - n] < n)
    return; // Two leaves have no alternative topology.
  int frontier[TreeletSize];
  int internal[TreeletSize - 1];
  int count = 1;
  frontier[0] = root;
  Cost old_cost = 0;

  // Expand the largest internal frontier node, retaining opaque subtrees at
  // the boundary. Every expanded internal node will be reused exactly once.
  while (count < TreeletSize)
  {
    int selected = -1;
    Cost largest = 0;
    for (int i = 0; i < count; ++i)
      if (frontier[i] >= n)
      {
        auto const cost = cost_function(nodes[frontier[i] - n].bounding_volume);
        if (selected == -1 || cost > largest)
        {
          selected = i;
          largest = cost;
        }
      }
    if (selected == -1)
      break;
    int const node = frontier[selected];
    internal[count - 1] = node;
    old_cost += largest;
    frontier[selected] = nodes[node - n].left_child;
    frontier[count] = right_children[node - n];
    ++count;
  }

  int left[TreeletSize - 1];
  int right[TreeletSize - 1];
  Cost new_cost = 0;
  {
    // Frontier selection needs only node indices. Keep bounding volumes live
    // only during clustering, when the mutable cluster bounds are needed.
    BoundingVolume volumes[TreeletSize];
    for (int i = 0; i < count; ++i)
      volumes[i] =
          getBoundingVolume(frontier[i], n, leaves, nodes, indexable_getter);

    // The costs of the two selected clusters are constant across candidate
    // pairs, so minimizing the union's cost gives the cheapest next merge.
    // Cache these union costs; only pairs involving the merged cluster change.
    // Entries involving inactive frontier slots may be stale and are ignored.
    Cost pairs[TreeletSize * (TreeletSize - 1) / 2];
    for (int i = 1; i < count; ++i)
      for (int j = 0; j < i; ++j)
      {
        auto volume = volumes[i];
        expand(volume, volumes[j]);
        pairs[pairIndex(i, j)] = cost_function(volume);
      }

    for (int step = 0; step < count - 1; ++step)
    {
      int a = -1;
      int b = -1;
      Cost best = 0;
      for (int i = 1; i < count; ++i)
        if (frontier[i] != -1)
          for (int j = 0; j < i; ++j)
            if (frontier[j] != -1)
            {
              auto const cost = pairs[pairIndex(i, j)];
              if (a == -1 || cost < best)
              {
                a = i;
                b = j;
                best = cost;
              }
            }
      left[step] = frontier[a];
      right[step] = frontier[b];
      new_cost += best;
      if (step == count - 2)
        break; // The root's merged volume is never stored or used again.
      expand(volumes[a], volumes[b]);
      // Reverse reuse leaves the root at its original index. This is essential
      // for the original parent links to remain valid for scheduling ancestors.
      frontier[a] = internal[count - 2 - step];
      frontier[b] = -1; // Retire this slot; its cluster is now part of slot a.
      for (int i = 0; i < count; ++i)
        if (i != a && frontier[i] != -1)
        {
          auto volume = volumes[a];
          expand(volume, volumes[i]);
          pairs[pairIndex(a, i)] = cost_function(volume);
        }
    }

  } // Cluster volumes and pair costs are dead before bounds reconstruction.

  // Greedy clustering need not beat the input topology. Frontier subtree costs
  // are unchanged, so comparing these sums compares the entire subtree cost.
  if (new_cost < old_cost)
    for (int step = 0; step < count - 1; ++step)
    {
      int const node = internal[count - 2 - step] - n;
      right_children[node] = right[step];
      nodes[node].left_child = left[step];
      // Recompute accepted bounds in merge order instead of retaining another
      // TreeletSize-1 bounding volumes throughout clustering. Original frontier
      // nodes are never overwritten; merged children have already been updated.
      // Preserve the subtree root's volume exactly, including the global root.
      if (step != count - 2)
      {
        auto volume =
            getBoundingVolume(left[step], n, leaves, nodes, indexable_getter);
        expand(volume, getBoundingVolume(right[step], n, leaves, nodes,
                                         indexable_getter));
        nodes[node].bounding_volume = volume;
      }
    }
}
} // namespace TreeOptimization

template <int TreeletSize, class ExecutionSpace, class BVH, class LeafNodes,
          class InternalNodes, class IndexableGetter, class CostFunction>
void optimizeHierarchy(ExecutionSpace const &space, BVH const &bvh,
                       LeafNodes const &leaves, InternalNodes const &nodes,
                       IndexableGetter const &indexable_getter,
                       CostFunction const &cost_function)
{
  static_assert(TreeletSize >= 2, "A treelet must have at least two leaves");
  int const n = leaves.extent_int(0);
  if (n < 3 || TreeletSize == 2)
    return;
  using MemorySpace = typename LeafNodes::memory_space;
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::optimize::parents"),
      2 * n - 1);
  // These are the original parents, used only to schedule completion. They
  // must not follow the changing topology: ancestor restructuring can reuse
  // internal node indices after the original child subtrees have completed.
  findParents(space, bvh, parents);
  Kokkos::View<int *, MemorySpace> right_children(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::optimize::right_children"),
      n - 1);
  // Snapshot right children before changing any left links: in the compact
  // node representation, a right child is stored as its left sibling's rope.
  // Views own the temporary allocations; kernels need only their data pointers.
  auto *leaf_data = leaves.data();
  auto *node_data = nodes.data();
  auto *right_child_data = right_children.data();
  auto const *parent_data = parents.data();
  Kokkos::parallel_for(
      "ArborX::BVH::optimize::extract_children",
      Kokkos::RangePolicy(space, 0, n - 1), KOKKOS_LAMBDA(int i) {
        int const left = node_data[i].left_child;
        right_child_data[i] =
            left < n ? leaf_data[left].rope : node_data[left - n].rope;
      });
  Kokkos::View<int *, MemorySpace> arrivals(
      Kokkos::view_alloc(space, "ArborX::BVH::optimize::arrivals"), n - 1);
  auto *arrival_data = arrivals.data();
  Kokkos::parallel_for(
      "ArborX::BVH::optimize::restructure", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        while (i != n)
        {
          int const parent = parent_data[i];
          // Publish all subtree writes before announcing completion. The first
          // arrival exits; the second acquires the completed sibling subtree.
          // Original parents are immutable during this kernel. Disjoint ready
          // subtrees have disjoint writers, and an ancestor starts only after
          // both original children have stopped accessing their subtrees.
          Kokkos::memory_fence();
          if (Kokkos::atomic_compare_exchange(&arrival_data[parent - n], 0,
                                              1) == 0)
            return;
          Kokkos::memory_fence();
          TreeOptimization::restructure<TreeletSize>(
              parent, n, leaf_data, node_data, indexable_getter,
              right_child_data, cost_function);
          i = parent;
        }
      });

  // Partition the tree into right-child chains, starting at each left child
  // and at the root. All nodes on a chain have the same rope: its starting
  // node's right sibling, or the sentinel for the root chain. Every node lies
  // on exactly one chain, giving disjoint writers and linear total work even
  // for skewed trees. Read only stable topology, never other nodes' ropes.
  Kokkos::parallel_for(
      "ArborX::BVH::optimize::rebuild_ropes", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        int node = i < n - 1 ? node_data[i].left_child : n;
        int const rope = i < n - 1 ? right_child_data[i] : ROPE_SENTINEL;
        while (node >= n)
        {
          node_data[node - n].rope = rope;
          node = right_child_data[node - n];
        }
        leaf_data[node].rope = rope;
      });
}
} // namespace ArborX::Details

#endif
