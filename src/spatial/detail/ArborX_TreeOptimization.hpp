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

namespace ArborX::Experimental
{
// Probability surrogate for fixed-radius searches with uniformly distributed
// query centers in 3D: a ball intersects a box precisely when its center lies
// in the box's Minkowski sum with a radius-r ball. Nonuniform workloads may
// need a metric estimated from sampled queries instead.
template <class Coordinate>
struct UniformRadiusSearchCost
{
  Coordinate radius;
  template <class Box>
  KOKKOS_FUNCTION auto operator()(Box const &box) const
  {
    static_assert(GeometryTraits::is_box_v<Box> &&
                  GeometryTraits::dimension_v<Box> == 3);
    KOKKOS_ASSERT(radius >= 0);
    using Scalar =
        std::common_type_t<Coordinate, GeometryTraits::coordinate_type_t<Box>>;
    constexpr Scalar pi = 3.1415926535897932384626433832795;
    Scalar const a = box.maxCorner()[0] - box.minCorner()[0];
    Scalar const b = box.maxCorner()[1] - box.minCorner()[1];
    Scalar const c = box.maxCorner()[2] - box.minCorner()[2];
    Scalar const r = radius;
    // Face slabs, quarter cylinders along edges, and spherical corner pieces.
    // This formula also handles planes, lines, and points continuously.
    return a * b * c + 2 * r * (a * b + a * c + b * c) +
           pi * r * r * (a + b + c) + (4 * pi / 3) * r * r * r;
  }
};
} // namespace ArborX::Experimental

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

// One shared workspace per team, rather than a complete treelet per lane.
// GPU teams contain one warp; host teams use one thread with the same
// algorithm.
template <int K, class Volume, class Cost>
struct TreeletWorkspace
{
  int frontier[K], internal[K - 1], left[K - 1], right[K - 1];
  Volume volumes[K], merged[K - 1];
  Cost pairs[K * K], old_cost, new_cost;
  int count;
};

template <int K, class Team, class LeafNode, class InternalNode, class Getter,
          class Metric, class Workspace>
KOKKOS_FUNCTION void restructure(Team const &team, Workspace &w, int root,
                                 int n, LeafNode const *leaves,
                                 InternalNode *nodes, Getter const &getter,
                                 int *right_children, Metric const &metric)
{
  using Cost = std::decay_t<std::invoke_result_t<
      Metric const &, typename InternalNode::bounding_volume_type const &>>;
  // Frontier selection is small and sequential; all expensive pair geometry
  // and minimum searches below are distributed across the team's lanes.
  Kokkos::single(Kokkos::PerTeam(team), [&] {
    w.count = 1;
    w.frontier[0] = root;
    w.old_cost = 0;
    w.new_cost = 0;
    while (w.count < K)
    {
      int selected = -1;
      Cost largest = 0;
      for (int i = 0; i < w.count; ++i)
        if (w.frontier[i] >= n)
        {
          auto cost = metric(nodes[w.frontier[i] - n].bounding_volume);
          if (selected == -1 || cost > largest)
          {
            selected = i;
            largest = cost;
          }
        }
      if (selected == -1)
        break;
      int node = w.frontier[selected] - n;
      w.internal[w.count - 1] = node + n;
      w.old_cost += largest;
      w.frontier[selected] = nodes[node].left_child;
      w.frontier[w.count++] = right_children[node];
    }
  });
  team.team_barrier();
  int const count = w.count;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, count), [&](int i) {
    w.volumes[i] = getBoundingVolume(w.frontier[i], n, leaves, nodes, getter);
  });
  team.team_barrier();
  // A square matrix avoids index decoding and gives every candidate a stable
  // tie-breaking index. Only its strict lower triangle is read or written.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, K * K), [&](int ij) {
    int i = ij / K, j = ij % K;
    if (i < count && j < i)
    {
      auto v = w.volumes[i];
      expand(v, w.volumes[j]);
      w.pairs[ij] = metric(v);
    }
  });
  team.team_barrier();
  for (int step = 0; step < count - 1; ++step)
  {
    using Reduction = Kokkos::MinLoc<Cost, int>;
    typename Reduction::value_type best;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, K * K),
        [&](int ij, typename Reduction::value_type &update) {
          int i = ij / K, j = ij % K;
          if (i < count && j < i && w.frontier[i] != -1 && w.frontier[j] != -1)
            if (w.pairs[ij] < update.val ||
                (w.pairs[ij] == update.val && ij < update.loc))
            {
              update.val = w.pairs[ij];
              update.loc = ij;
            }
        },
        Reduction(best));
    int const a = best.loc / K, b = best.loc % K;
    Kokkos::single(Kokkos::PerTeam(team), [&] {
      w.left[step] = w.frontier[a];
      w.right[step] = w.frontier[b];
      w.new_cost += best.val;
      expand(w.volumes[a], w.volumes[b]);
      w.merged[step] = w.volumes[a];
      w.frontier[a] = w.internal[count - 2 - step];
      w.frontier[b] = -1;
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, count), [&](int i) {
      if (i != a && w.frontier[i] != -1)
      {
        auto v = w.volumes[a];
        expand(v, w.volumes[i]);
        w.pairs[a > i ? a * K + i : i * K + a] = metric(v);
      }
    });
    team.team_barrier();
  }
  if (w.new_cost < w.old_cost)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, count - 1),
                         [&](int step) {
                           int node = w.internal[count - 2 - step] - n;
                           nodes[node].left_child = w.left[step];
                           right_children[node] = w.right[step];
                           if (step != count - 2)
                             nodes[node].bounding_volume = w.merged[step];
                         });
  // Publish all node writes before any owner announces completion to its
  // parent.
  team.team_barrier();
}
} // namespace TreeOptimization

template <int TreeletSize, class ExecutionSpace, class BVH, class LeafNodes,
          class InternalNodes, class IndexableGetter, class CostFunction>
void optimizeHierarchy(ExecutionSpace const &space, BVH const &bvh,
                       LeafNodes const &leaves, InternalNodes const &nodes,
                       IndexableGetter const &indexable_getter,
                       CostFunction const &cost_function, int iterations = 2)
{
  static_assert(TreeletSize >= 2, "A treelet must have at least two leaves");
  int const n = leaves.extent_int(0);
  if (n < TreeletSize || n < 3 || TreeletSize == 2 || iterations <= 0)
    return;
  using MemorySpace = typename LeafNodes::memory_space;
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::optimize::parents"),
      2 * n - 1);
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
  Kokkos::View<int *, MemorySpace> arrivals(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::optimize::arrivals"),
      n - 1);
  auto *arrival_data = arrivals.data();
  using Volume = typename InternalNodes::value_type::bounding_volume_type;
  using Cost =
      std::decay_t<std::invoke_result_t<CostFunction const &, Volume const &>>;
  using Workspace =
      TreeOptimization::TreeletWorkspace<TreeletSize, Volume, Cost>;
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  int team_size = 1;
#ifdef KOKKOS_ENABLE_THREADS
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Threads>)
    team_size = space.concurrency() >= 2 ? 2 : 1;
#endif
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
    team_size = 32;
#endif
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
    team_size = 64;
#endif
  // Reuse global workspace between sweeps. gamma doubles, so later sweeps
  // target larger subtrees rather than repeatedly optimizing tiny ones.
  int gamma = TreeletSize;
  for (int sweep = 0; sweep < iterations && gamma <= n; ++sweep)
  {
    findParents(space, bvh, parents);
    Kokkos::deep_copy(space, arrivals, 0);
    Kokkos::parallel_for(
        "ArborX::BVH::optimize::extract_children",
        Kokkos::RangePolicy(space, 0, n - 1), KOKKOS_LAMBDA(int i) {
          int left = node_data[i].left_child;
          right_child_data[i] =
              left < n ? leaf_data[left].rope : node_data[left - n].rope;
        });
    Policy policy(space, (n - 1) / team_size + 1, team_size);
    policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(Workspace)));
    Kokkos::parallel_for(
        "ArborX::BVH::optimize::restructure", policy,
        KOKKOS_LAMBDA(typename Policy::member_type const &team) {
          auto &w = *static_cast<Workspace *>(
              team.team_shmem().get_shmem(sizeof(Workspace)));
          int i = team.league_rank() * team.team_size() + team.team_rank();
          if (i >= n)
            i = -1;
          int count = 1;
          while (true)
          {
            int ready = 2 * n - 1;
            if (i >= 0 && i != n)
            {
              int parent = parent_data[i];
              // The CAS carries the first child's descendant count as well as
              // its completion flag. No separate O(n) count array is needed.
              Kokkos::memory_fence();
              int other = Kokkos::atomic_compare_exchange(
                  &arrival_data[parent - n], 0, count);
              if (other == 0)
                i = -1;
              else
              {
                Kokkos::memory_fence();
                count += other;
                i = parent;
                if (count >= gamma)
                  ready = parent;
              }
            }
            else
              i = -1;
            // Retired lanes remain available as helpers. Drain every ready root
            // before any owner advances; only that owner continues after
            // commit.
            while (true)
            {
              int selected = ready;
              team.team_reduce(Kokkos::Min<int>(selected));
              if (selected == 2 * n - 1)
                break;
              TreeOptimization::restructure<TreeletSize>(
                  team, w, selected, n, leaf_data, node_data, indexable_getter,
                  right_child_data, cost_function);
              if (ready == selected)
                ready = 2 * n - 1;
            }
            int active;
            // Each team thread contributes its own continuation state.
            active = i >= 0 && i != n ? 1 : 0;
            team.team_reduce(Kokkos::Sum<int>(active));
            if (active == 0)
              break;
          }
        });
    // Partition the tree into right-child chains, starting at each left child
    // and at the root. All nodes on a chain have the same rope: its starting
    // node's right sibling, or the sentinel for the root chain. Every node lies
    // on exactly one chain, giving disjoint writers and linear total work even
    // for skewed trees. Read only stable topology, never other nodes' ropes.
    Kokkos::parallel_for(
        "ArborX::BVH::optimize::rebuild_ropes",
        Kokkos::RangePolicy(space, 0, n), KOKKOS_LAMBDA(int i) {
          int node = i < n - 1 ? node_data[i].left_child : n;
          int const rope = i < n - 1 ? right_child_data[i] : ROPE_SENTINEL;
          while (node >= n)
          {
            node_data[node - n].rope = rope;
            node = right_child_data[node - n];
          }
          leaf_data[node].rope = rope;
        });
    if (gamma > n / 2)
      break;
    gamma *= 2;
  }
}
} // namespace ArborX::Details

#endif
