/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_MINIMUM_SPANNING_TREE_HPP
#define ARBORX_DETAILS_MINIMUM_SPANNING_TREE_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp> // isfinite, signbit
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX::Details
{

enum class BoruvkaMode
{
  MST,
  HDBSCAN
};

constexpr int ROOT_CHAIN_VALUE = -2;
constexpr int FOLLOW_CHAIN_VALUE = -3;

class DirectedEdge
{
public:
  unsigned long long directed_edge = ULLONG_MAX;
  float weight = KokkosExt::ArithmeticTraits::infinity<float>::value;

private:
  static_assert(sizeof(unsigned long long) == 8);
  static constexpr int source_shift = 32;
  static constexpr int target_shift = 1;
  static constexpr unsigned long long mask_source =
      static_cast<unsigned long long>(UINT_MAX >> 1) << source_shift;
  static constexpr unsigned long long mask_target =
      static_cast<unsigned long long>(UINT_MAX >> 1) << target_shift;
  // clang-format off
  // | unused bit | 31 bits for the smallest of source and target | 31 bits for the largest of source and target | flag |
  // clang-format on
  static_assert((mask_source & mask_target) == 0, "implementation bug");
  // direction must be stored in the least significant bit
  static constexpr unsigned long long reverse_direction = 1;

  // performs lexicographical comparison by comparing first the weights and then
  // the unordered pair of vertices
  friend KOKKOS_FUNCTION constexpr bool operator<(DirectedEdge const &lhs,
                                                  DirectedEdge const &rhs)
  {
    return (lhs.weight != rhs.weight) ? (lhs.weight < rhs.weight)
                                      : (lhs.directed_edge < rhs.directed_edge);
  }

  KOKKOS_FUNCTION constexpr bool reverse() const
  {
    return (directed_edge & reverse_direction) == reverse_direction;
  }

public:
  KOKKOS_FUNCTION constexpr int source() const
  {
    return reverse() ? (directed_edge & mask_target) >> target_shift
                     : (directed_edge & mask_source) >> source_shift;
  }
  KOKKOS_FUNCTION constexpr int target() const
  {
    return reverse() ? (directed_edge & mask_source) >> source_shift
                     : (directed_edge & mask_target) >> target_shift;
  }
  KOKKOS_FUNCTION constexpr DirectedEdge(int source, int target, float weight)
      : directed_edge{(source < target)
                          ? (mask_source &
                             (static_cast<unsigned long long>(source)
                              << source_shift)) +
                                (mask_target &
                                 (static_cast<unsigned long long>(target)
                                  << target_shift))
                          : reverse_direction +
                                (mask_source &
                                 (static_cast<unsigned long long>(target)
                                  << source_shift)) +
                                (mask_target &
                                 (static_cast<unsigned long long>(source))
                                     << target_shift)}
      , weight{weight}
  {}
  KOKKOS_DEFAULTED_FUNCTION constexpr DirectedEdge() = default;
  KOKKOS_FUNCTION explicit constexpr operator WeightedEdge()
  {
    return {source(), target(), weight};
  }
};

template <class BVH, class Labels, class Weights, class Edges, class Metric,
          class Radii, class LowerBounds, bool UseSharedRadii>
struct FindComponentNearestNeighbors
{
  BVH _bvh;
  Labels _labels;
  Weights _weights;
  Edges _edges;
  Metric _metric;
  Radii _radii;
  LowerBounds _lower_bounds;

  struct WithLowerBounds
  {};

  template <class ExecutionSpace>
  FindComponentNearestNeighbors(ExecutionSpace const &space, BVH const &bvh,
                                Labels const &labels, Weights const &weights,
                                Edges const &edges, Metric const &metric,
                                Radii const &radii,
                                LowerBounds const &lower_bounds,
                                std::bool_constant<UseSharedRadii>)
      : _bvh(bvh)
      , _labels(labels)
      , _weights(weights)
      , _edges(edges)
      , _metric{metric}
      , _radii(radii)
      , _lower_bounds(lower_bounds)
  {
    int const n = bvh.size();
    ARBORX_ASSERT(labels.extent_int(0) == 2 * n - 1);
    ARBORX_ASSERT(edges.extent_int(0) == n);
    ARBORX_ASSERT(radii.extent_int(0) == n);

#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<ExecutionSpace, Kokkos::Serial>{})
    {
      Kokkos::parallel_for(
          "ArborX::MST::find_component_nearest_neighbors_with_lower_bounds",
          Kokkos::RangePolicy<ExecutionSpace, WithLowerBounds>(space, 0, n),
          *this);
    }
    else
#endif
    {
      Kokkos::parallel_for("ArborX::MST::find_component_nearest_neighbors",
                           Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                           *this);
    }
  }

  KOKKOS_FUNCTION void operator()(WithLowerBounds, int i) const
  {
    auto const component = _labels(i);
    if (_lower_bounds(i) <= _radii(component))
    {
      this->operator()(i);
    }
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;

    auto const distance = [bounding_volume_i =
                               HappyTreeFriends::getIndexable(_bvh, i),
                           &bvh = _bvh](int j) {
      using Details::distance;
      return HappyTreeFriends::isLeaf(bvh, j)
                 ? distance(bounding_volume_i,
                            HappyTreeFriends::getIndexable(bvh, j))
                 : distance(
                       bounding_volume_i,
                       HappyTreeFriends::getInternalBoundingVolume(bvh, j));
    };

    auto const component = _labels(i);
    auto const predicate = [label_i = component, &labels = _labels](int j) {
      return label_i != labels(j);
    };
    auto const leaf_permutation_i = HappyTreeFriends::getValue(_bvh, i).index;

    DirectedEdge current_best{};

    // Use a reference for shared radii, and a copy otherwise.
    std::conditional_t<UseSharedRadii, float &, float> radius =
        _radii(component);

    constexpr int SENTINEL = -1;
    int stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = SENTINEL;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    float stack_distance[64];
    auto *stack_distance_ptr = stack_distance;
    *stack_distance_ptr++ = 0;
#endif

    int node = HappyTreeFriends::getRoot(_bvh);
    float distance_node = 0;

    // Important! The truncation radius is computed using the provided metric,
    // rather than just assigning the Euclidean distance. This only works for
    // metrics that return a value greater or equal to Euclidean distance
    // (e.g., mutual reachability metric). Metrics that do not satisfy this
    // criteria may return wrong results.
    do
    {
      bool traverse_left = false;
      bool traverse_right = false;

      int left_child;
      int right_child;
      float distance_left = inf;
      float distance_right = inf;

      // Note it is <= instead of < when comparing with radius here and below.
      // The reason is that in Boruvka it matters which of the equidistant
      // points we take so that they don't create a cycle among component
      // connectivity. This requires us to uniquely resolve equidistant
      // neighbors, so we cannot skip any of them.
      if (distance_node <= radius)
      {
        // Insert children into the stack and make sure that the closest one
        // ends on top.
        left_child = HappyTreeFriends::getLeftChild(_bvh, node);
        right_child = HappyTreeFriends::getRightChild(_bvh, node);
        distance_left = distance(left_child);
        distance_right = distance(right_child);

        if (predicate(left_child) && distance_left <= radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, left_child))
          {
            float const candidate_dist =
                _metric(leaf_permutation_i,
                        HappyTreeFriends::getValue(_bvh, left_child).index,
                        distance_left);
            DirectedEdge const candidate_edge{i, left_child, candidate_dist};
            if (candidate_edge < current_best)
            {
              current_best = candidate_edge;
              if constexpr (UseSharedRadii)
                Kokkos::atomic_min(&radius, candidate_dist);
              else
                radius = candidate_dist;
            }
          }
          else
          {
            traverse_left = true;
          }
        }

        // Note: radius may have been already updated here from the left child
        if (predicate(right_child) && distance_right <= radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, right_child))
          {
            float const candidate_dist =
                _metric(leaf_permutation_i,
                        HappyTreeFriends::getValue(_bvh, right_child).index,
                        distance_right);
            DirectedEdge const candidate_edge{i, right_child, candidate_dist};
            if (candidate_edge < current_best)
            {
              current_best = candidate_edge;
              if constexpr (UseSharedRadii)
                Kokkos::atomic_min(&radius, candidate_dist);
              else
                radius = candidate_dist;
            }
          }
          else
          {
            traverse_right = true;
          }
        }
      }

      if (!traverse_left && !traverse_right)
      {
        node = *--stack_ptr;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (node != SENTINEL)
        {
          // This is a theoretically unnecessary duplication of distance
          // calculation for stack nodes. However, for Cuda it's better than
          // putting the distances in stack.
          distance_node = distance(node);
        }
#else
        distance_node = *--stack_distance_ptr;
#endif
      }
      else
      {
        node = (traverse_left &&
                (distance_left <= distance_right || !traverse_right))
                   ? left_child
                   : right_child;
        distance_node = (node == left_child ? distance_left : distance_right);
        if (traverse_left && traverse_right)
        {
          *stack_ptr++ = (node == left_child ? right_child : left_child);
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
          *stack_distance_ptr++ =
              (node == left_child ? distance_right : distance_left);
#endif
        }
      }
    } while (node != SENTINEL);

    // This check is only here to reduce hammering the atomics for large
    // components. Otherwise, for a large number of points and a small number of
    // components it becomes extremely expensive.
    auto &component_weight = _weights(component);
    if (current_best.weight < inf && current_best.weight <= component_weight)
    {
      if (Kokkos::atomic_min_fetch(&component_weight, current_best.weight) ==
          current_best.weight)
      {
        _edges(i) = current_best;
      }
    }
  }
};

// For every component C, find the shortest edge (v, w) such that v is in C
// and w is not in C. The found edge is stored in component_out_edges(C).
template <class ExecutionSpace, class BVH, class Labels, class Weights,
          class Edges, class Metric, class Radii, class LowerBounds,
          bool UseSharedRadii>
FindComponentNearestNeighbors(ExecutionSpace, BVH, Labels, Weights, Edges,
                              Metric, Radii, LowerBounds,
                              std::bool_constant<UseSharedRadii>)
    -> FindComponentNearestNeighbors<BVH, Labels, Weights, Edges, Metric, Radii,
                                     LowerBounds, UseSharedRadii>;

template <class ExecutionSpace, class Labels, class ComponentOutEdges,
          class LowerBounds>
void updateLowerBounds(ExecutionSpace const &space, Labels const &labels,
                       ComponentOutEdges const &component_out_edges,
                       LowerBounds lower_bounds)
{
  auto const n = lower_bounds.extent(0);
  Kokkos::parallel_for(
      "ArborX::MST::update_lower_bounds",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        using KokkosExt::max;
        auto component = labels(i);
        auto const &edge = component_out_edges(component);
        lower_bounds(i) = max(lower_bounds(i), edge.weight);
      });
}

// workaround slow atomic min operations on edge type
template <class ExecutionSpace, class Labels, class Weights, class Edges>
void retrieveEdges(ExecutionSpace const &space, Labels const &labels,
                   Weights const &weights, Edges const &edges)
{
  auto const n = weights.extent(0);
  Kokkos::parallel_for(
      "ArborX::MST::reset_component_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto const component = labels(i);
        if (i != component)
          return;
        auto const component_weight = weights(component);
        auto &component_edge = edges(component);
        // replace stale values by neutral element for min reduction
        if (component_edge.weight != component_weight)
        {
          component_edge = {};
          component_edge.weight = component_weight;
        }
      });
  Kokkos::parallel_for(
      "ArborX::MST::reduce_component_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto const component = labels(i);
        auto const component_weight = weights(component);
        auto const &edge = edges(i);
        if (edge.weight == component_weight)
        {
          auto &component_edge = edges(component);
          Kokkos::atomic_min(&(component_edge.directed_edge),
                             edge.directed_edge);
        }
      });
}

struct LabelsTag
{};
struct UnidirectionalEdgesTag
{};
struct BidirectionalEdgesTag
{};

template <class Labels, class OutEdges, class Edges, class EdgesMapping,
          class EdgesCount, BoruvkaMode Mode>
struct UpdateComponentsAndEdges
{
  Labels _labels;
  OutEdges _out_edges;
  Edges _edges;
  EdgesMapping _edge_mapping;
  EdgesCount _num_edges;

  KOKKOS_FUNCTION auto computeNextComponent(int component) const
  {
    int next_component = _labels(_out_edges(component).target());
    int next_next_component = _labels(_out_edges(next_component).target());

    if (next_next_component != component)
    {
      // The component's edge is unidirectional
      return next_component;
    }
    // The component's edge is bidirectional, uniquely resolve the bidirectional
    // edge
    return KokkosExt::min(component, next_component);
  }

  KOKKOS_FUNCTION auto computeFinalComponent(int component) const
  {
    int prev_component = component;
    int next_component;
    while ((next_component = computeNextComponent(prev_component)) !=
           prev_component)
      prev_component = next_component;

    return next_component;
  }

  KOKKOS_FUNCTION void operator()(LabelsTag, int i) const
  {
    auto const component = _labels(i);
    auto const final_component = computeFinalComponent(component);
    _labels(i) = final_component;
  }

  KOKKOS_FUNCTION void operator()(UnidirectionalEdgesTag, int i) const
  {
    auto const component = _labels(i);
    if (i != component || computeNextComponent(component) == component)
      return;

    // append new edge at the "end" of the array (akin to
    // std::vector::push_back)
    auto const edge = static_cast<WeightedEdge>(_out_edges(i));
    auto const back =
        Kokkos::atomic_fetch_inc(&_num_edges()); // atomic post-increment
    _edges(back) = edge;

    if constexpr (Mode == BoruvkaMode::HDBSCAN)
      _edge_mapping(i) = back;
  }

  KOKKOS_FUNCTION void operator()(BidirectionalEdgesTag, int i) const
  {
    auto const component = _labels(i);
    if (i != component || computeNextComponent(component) != component)
      return;

    auto const &edge = _out_edges(i);
    _edge_mapping(i) = _edge_mapping(_labels(edge.target()));
  }
};

// Reverse node leaf permutation order back to original indices
template <class ExecutionSpace, class BVH, class Edges>
void finalizeEdges(ExecutionSpace const &space, BVH const &bvh,
                   Edges const &edges)
{
  int const n = bvh.size();
  ARBORX_ASSERT(edges.extent_int(0) == n - 1);
  Kokkos::parallel_for(
      "ArborX::MST::finalize_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i) {
        edges(i).source =
            HappyTreeFriends::getValue(bvh, edges(i).source).index;
        edges(i).target =
            HappyTreeFriends::getValue(bvh, edges(i).target).index;
      });
}

template <class ExecutionSpace, class Labels, class Edges, class EdgesMapping,
          class SidedParents>
void updateSidedParents(ExecutionSpace const &space, Labels const &labels,
                        Edges const &edges, EdgesMapping const &edges_mapping,
                        SidedParents &sided_parents, int edges_start,
                        int edges_end)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::MST::update_sided_parents");

  // Same as dendrogram alpha's standalone "updateSidedParents"
  Kokkos::parallel_for(
      "ArborX::MST::update_sided_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, edges_start, edges_end),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);

        // As the edge is within the same alpha vertex, labels of its vertices
        // are the same, so can take either
        int component = labels(edge.source);

        int const alpha_edge_index = edges_mapping(component);

        auto const &alpha_edge = edges(alpha_edge_index);

        if (edge < alpha_edge)
        {
          bool is_left_side = (labels(alpha_edge.source) == component);
          sided_parents(e) =
              2 * alpha_edge_index + static_cast<int>(is_left_side);
        }
        else
        {
          sided_parents(e) = FOLLOW_CHAIN_VALUE - alpha_edge_index;
        }
      });
}

template <class ExecutionSpace, class Labels, class OutEdges,
          class EdgesMapping, class BVH, class Parents>
void assignVertexParents(ExecutionSpace const &space, Labels const &labels,
                         OutEdges const &out_edges,
                         EdgesMapping const &edges_mapping, BVH const &bvh,
                         Parents parents)
{
  auto const n = edges_mapping.extent_int(0) + 1;
  int const vertices_offset = n - 1;

  Kokkos::parallel_for(
      "ArborX::MST::compute_vertex_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int e) {
        auto const &edge = out_edges(e);

        int i = labels(edge.source());
        parents(HappyTreeFriends::getValue(bvh, i).index + vertices_offset) =
            edges_mapping(i);
      });
}

template <typename ExecutionSpace, typename Edges,
          typename EdgeHierarchyOffsets, typename SidedParents,
          typename Parents, typename ChainOffsets, typename ChainLevels>
void computeParentsAndReorderEdges(
    ExecutionSpace const &space, Edges &edges,
    EdgeHierarchyOffsets const &edge_hierarchy_offsets,
    SidedParents const &sided_parents, Parents &parents,
    ChainOffsets &chain_offsets, ChainLevels &chain_levels)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::MST::compute_edge_parents");

  using MemorySpace = typename SidedParents::memory_space;

  int num_edges = edges.size();

  // Encode both a sided parent and an edge weight into long long.
  // This way, once we sort based on this value, edges with the same sided
  // parent will already be sorted in increasing order.
  // The main reason for using long long values is the performance when
  // compared with sorting pairs. The second reason is that Kokkos's BinSort
  // does not support custom comparison operators.
  static_assert(sizeof(long long) >= sizeof(int) + sizeof(float));
  Kokkos::View<long long *, MemorySpace> keys(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MST::keys"),
      num_edges);

  constexpr int shift = sizeof(int) * CHAR_BIT;

  Kokkos::parallel_for(
      "ArborX::MST::compute_sided_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        long long key = sided_parents(e);
        auto const &edge = edges(e);
        if (key <= FOLLOW_CHAIN_VALUE)
        {
          int next = FOLLOW_CHAIN_VALUE - key;
          do
          {
            key = sided_parents(next);
            if (key <= FOLLOW_CHAIN_VALUE)
              next = FOLLOW_CHAIN_VALUE - key;
            else if (key >= 0)
            {
              next = key / 2;
              auto const &next_edge = edges(next);
              if (edge < next_edge)
                break;
            }
            else if (key == ROOT_CHAIN_VALUE)
              break;
          } while (true);
        }
        if (key == ROOT_CHAIN_VALUE)
          key = INT_MAX;

        // Comparison of weights as ints is the same as their comparison as
        // floats as long as they are positive and are not NaNs or inf. We use
        // signbit instead of >= 0 just as an extra precaution against negative
        // floating zeros.
        static_assert(sizeof(int) == sizeof(float));
        KOKKOS_ASSERT(Kokkos::isfinite(edge.weight) &&
                      Kokkos::signbit(edge.weight) == 0);
        keys(e) = (key << shift) + Kokkos::bit_cast<int>(edge.weight);
      });

  auto permute = sortObjects(space, keys);

  // Make sure we produce a binary dendrogram
  //
  // The issue is that the edges are sorted above, edges that are in the same
  // chain and have same weight may be in unpredictable order. For the most
  // part, it does not matter, as we don't care about some minor dendrogram
  // perturbations. However, there is one situation that needs to be addressed.
  //
  // Specifically, what could happen is that during edge construction, an edge
  // can already be set as having two children. That could happen either for
  // leaf edges (an edge having two vertex children), or an alpha edge. Then,
  // during the sort, if that edge is not the first one in the chain, it gains
  // a third child, breaking the binary nature of the dendrogram. Note that
  // this can only happen to one edge in the chain, and it's going to be the
  // smallest one there.
  //
  // So, we identify the smallest edge in the chain, and put it first. We don't
  // need to scan the whole chain, just the smallest part of it.
  //
  // Note that this issue could have been avoided if we sorted the edges based
  // on their rank. But obtaining the rank would require another sort that we
  // want to avoid.
  Kokkos::parallel_for(
      "ArborX::MST::fix_same_weight_order",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges - 1),
      KOKKOS_LAMBDA(int const i) {
        auto key = keys(i);

        if (i == 0 || ((keys(i - 1) >> shift) != (key >> shift)))
        {
          // i is at the start of a chain

          // Find the index of the smallest edge with the same weight in
          // this chain
          int m = i;
          for (int k = i + 1; k < num_edges && keys(k) == key; ++k)
            if (edges(permute(k)) < edges(permute(m)))
              m = k;

          // Place the smallest edge at the beginning of the chain
          if (m != i)
            KokkosExt::swap(permute(i), permute(m));
        }
      });

// #define PRINT

#ifdef PRINT
  printf("Keys:\n");
  auto keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
  auto permute_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, permute);
  for (int i = 0; i < num_edges; ++i)
  {
    int key = (keys_host(i) >> shift);
    if (key == INT_MAX)
      printf("[%d]: -1\n", permute_host(i));
    else
      printf("[%d]: %d(%c)\n", permute_host(i), key / 2, (key % 2 ? 'l' : 'r'));
  }
#endif

#ifdef PRINT
  printf("Hierarchy offsets:");
  auto edge_hierarchy_offsets_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, edge_hierarchy_offsets);
  for (int i = 0; i < (int)edge_hierarchy_offsets_host.size(); ++i)
    printf(" %d", edge_hierarchy_offsets_host(i));
  printf("\n");
#endif

  auto rev_permute =
      KokkosExt::cloneWithoutInitializingNorCopying(space, permute);
  Kokkos::parallel_for(
      "ArborX::MST::compute_rev_permute",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int const i) { rev_permute(permute(i)) = i; });

  Kokkos::parallel_for(
      "ArborX::MST::update_vertex_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, num_edges, parents.size()),
      KOKKOS_LAMBDA(int const i) { parents(i) = rev_permute(parents(i)); });

  KokkosExt::reallocWithoutInitializing(space, chain_offsets, num_edges + 1);
  int num_chains;
  Kokkos::parallel_scan(
      "ArborX::MST::compute_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        if (i == num_edges - 1 || (keys(i) >> shift) != (keys(i + 1) >> shift))
          ++update;

        if (final_pass)
        {
          if (i == num_edges - 1)
          {
            // The parent of the root node is set to -1
            parents(i) = -1;
            chain_offsets(update) = num_edges;
            chain_offsets(0) = 0;
          }
          else if ((keys(i) >> shift) == (keys(i + 1) >> shift))
          {
            // For the edges belonging to the same chain, assign the parent of
            // an edge to the edge with the next larger value
            parents(i) = i + 1;
          }
          else
          {
            // For an edge which points to the root of a chain, assign edge's
            // parent to be that root
            parents(i) = rev_permute((keys(i) >> shift) / 2);
            chain_offsets(update) = i + 1;
          }
        }
      },
      num_chains);
  Kokkos::resize(Kokkos::WithoutInitializing, chain_offsets, num_chains + 1);
#ifdef PRINT
  printf("#chains: %d\n", num_chains);
  printf("Chain offsets:");
  auto chain_offsets_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chain_offsets);
  for (int i = 0; i < (int)chain_offsets_host.size(); ++i)
    printf(" %d", chain_offsets_host(i));
  printf("\n");
#endif
#ifdef PRINT
  printf("Parents:\n");
  auto parents_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, parents);
  for (int i = 0; i < (int)parents_host.size(); ++i)
    printf("[%d] %d\n", i, parents_host(i));
#endif

  {
    auto edges_clone =
        KokkosExt::cloneWithoutInitializingNorCopying(space, edges);
    Kokkos::parallel_for(
        "ArborX::MST::reorder_edges",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
        KOKKOS_LAMBDA(int const i) { edges_clone(i) = edges(permute(i)); });
    edges = edges_clone;
  }

  Kokkos::resize(Kokkos::WithoutInitializing, chain_levels, num_chains + 1);
  int num_levels = 0;
  Kokkos::parallel_scan(
      "ArborX::MST::compute_chain_levels",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_chains),
      KOKKOS_LAMBDA(int i, int &level, bool final_pass) {
        auto upper_bound = [&v = edge_hierarchy_offsets](int x) {
          int first = 0;
          int last = v.extent_int(0);
          int count = last - first;
          while (count > 0)
          {
            int step = count / 2;
            int mid = first + step;
            if (!(x < v(mid)))
            {
              first = ++mid;
              count -= step + 1;
            }
            else
            {
              count = step;
            }
          }
          return first;
        };

#ifdef PRINT
        printf("-- [%d]: level %d\n", i,
               upper_bound(permute(chain_offsets(i))) - 1);
#endif

        if (i == num_chains - 1 ||
            upper_bound(permute(chain_offsets(i))) !=
                upper_bound(permute(chain_offsets(i + 1))))
        {
          ++level;
          if (final_pass)
            chain_levels(level) = i + 1;
        }
      },
      num_levels);
  ++num_levels;
  Kokkos::resize(Kokkos::WithoutInitializing, chain_levels, num_levels);
#ifdef PRINT
  printf("Chain levels:");
  auto chain_levels_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chain_levels);
  for (int i = 0; i < (int)chain_levels_host.size(); ++i)
    printf(" %d", chain_levels_host(i));
  printf("\n");
#endif
}

// Compute upper bound on the shortest edge of each component.
template <class ExecutionSpace, class BVH, class Labels, class Metric,
          class Radii>
void resetSharedRadii(ExecutionSpace const &space, BVH const &bvh,
                      Labels const &labels, Metric const &metric,
                      Radii const &radii)
{
  //  We will search for the shortest outgoing edge of a component. The better
  //  we initialize the upper bound on the distance (i.e., the smaller it is),
  //  the less traversal we will do and the faster it will be.
  //
  // Here, we use the knowledge that it is a self-collision problem. In other
  // words, we only have a single point cloud. We further use the fact that if
  // we sort predicates based on the Morton codes, it will match the order of
  // primitives (or be close enough, as some points with the same Morton codes
  // may be in a different order due to the unstable sort that we use). Thus, if
  // we take an index of a query, we assume that it matches the corresponding
  // primitive. If a label of that primitive is different from a label of its
  // neighbor (which is in fact its Morton neighbor), we compute the distance
  // between the two. The upper bound for a component is set to the minimum
  // distance between all such pairs. Given that the Morton neighbors are
  // typically close to each other, this should provide a reasonably low bound.
  auto const n = bvh.size();
  Kokkos::parallel_for(
      "ArborX::MST::reset_shared_radii",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i) {
        int const j = i + 1;
        auto const label_i = labels(i);
        auto const label_j = labels(j);
        if (label_i != label_j)
        {
          auto const r =
              metric(HappyTreeFriends::getValue(bvh, i).index,
                     HappyTreeFriends::getValue(bvh, j).index,
                     distance(HappyTreeFriends::getIndexable(bvh, i),
                              HappyTreeFriends::getIndexable(bvh, j)));
          Kokkos::atomic_min(&radii(label_i), r);
          Kokkos::atomic_min(&radii(label_j), r);
        }
      });
}

} // namespace ArborX::Details

#endif
