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

#ifndef ARBORX_HDBSCAN_HPP
#define ARBORX_HDBSCAN_HPP

#include <ArborX_LinearBVH.hpp>

#include <map>

namespace ArborX
{

template <typename Primitives>
struct PrimitivesWithK
{
  Primitives _primitives;
  int _k;
};

template <typename Primitives>
auto buildPredicates(Primitives v, int k)
{
  return PrimitivesWithK<Primitives>{v, k};
}

template <typename Primitives>
struct AccessTraits<PrimitivesWithK<Primitives>, PredicatesTag>
{
  using memory_space = typename Primitives::memory_space;
  using Predicates = PrimitivesWithK<Primitives>;
  static size_t size(Predicates const &w) { return w._primitives.extent(0); }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(nearest<Point>(w._primitives(i), w._k), (int)i);
  }
};

namespace HDBSCAN
{

template <typename MemorySpace, typename Primitives>
struct CoreDistancesCallback
{
  Primitives _primitives;
  Kokkos::View<float *, MemorySpace> _core_distances;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    auto i = getData(query);

    // TODO: we could avoid some of the distance calculations if we assume that
    // - all points have at least k neighbors
    // - the callback will be called in increasing distance
    using namespace Details;
    float d = distance(_primitives(i), _primitives(j));
    _core_distances(i) = KokkosExt::max(_core_distances(i), d);
  }
};

bool compareEdgesLess(std::tuple<int, int, double> const &edge1,
                      std::tuple<int, int, double> const &edge2)
{
  double dist1 = std::get<2>(edge1);
  double dist2 = std::get<2>(edge2);
  if (dist1 < dist2)
    return true;
  if (dist1 > dist2)
    return false;

  auto vmin1 = std::min(std::get<0>(edge1), std::get<1>(edge1));
  auto vmin2 = std::min(std::get<0>(edge2), std::get<1>(edge2));
  if (vmin1 < vmin2)
    return true;
  if (vmin1 > vmin2)
    return false;

  auto vmax1 = std::max(std::get<0>(edge1), std::get<1>(edge1));
  auto vmax2 = std::max(std::get<0>(edge2), std::get<1>(edge2));
  if (vmax1 < vmax2)
    return true;
  return false;
}

template <typename ExecutionSpace, typename Primitives, typename Labels,
          typename ClosestNeighbors, typename ComponentOutEdges>
void determineComponentEdges(typename ExecutionSpace const &exec_space,
                             Primitives const &primitives, Labels const &labels,
                             ClosestNeighbors cached_closest_neighbors,
                             ComponentOutEdges component_out_edges)
{
  auto const n = primitives.size();

  // Update closest neighbors if necessary
  using PredicateType =
      decltype(ArborX::attach(ArborX::Nearest<ArborX::Point>, int));
  Kokkos::View<PredicateType *, DeviceType> nearest_queries(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::hdbscan::boruvka::queries"),
      n);
  int num_queries;
  Kokkos::parallel_scan(
      "ArborX::hdbscan::boruvka::setup_queries",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        if (labels(cached_closest_neighbors(i)) == labels(i))
        {
          if (final_pass)
            queries(i) = attach(nearest<ArborX::Point>(primitives(i), 1), i);
          ++update;
        }
      },
      num_queries);
  Kokkos::resize(queries, num_queries);

  // FIXME: Use Damien's traversal
  bvh.query(exec_space, queries);

  // Find outgoing edge for each component
  const double infty = std::numeric_limits<double>::infinity();
  Kokkos::View<double *, MemorySpace> closest_component_dist(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::hdbscan::closest_component_dist"),
      n);
  Kokkos::deep_copy(exec_space, closest_component_dist, infty);

  using ArborX::Details;

  Kokkos::parallel_for(
      "ArborX::hdbscan::find_components_out_edges",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n)(int const i) {
        int const component = labels(i);
        auto &component_out_edge = component_out_edges(component);

        auto dist =
            distance(primitives(i), primitives(cached_closest_neighbors(i)));
        if (compare_edges_less(
                std::make_tuple(i, cached_closest_neighbors[i], dist),
                std::make_tuple(component_out_edge.first,
                                component_out_edge.second,
                                closest_component_dist[component])))
        {
          component_out_edge =
              Kokkos::make_pair(i, cached_closest_neighbors(i));
          closest_component_dist(component) = dist;
        }
      });
}

template <typename ExecutionSpace, typename ComponentOutEdges,
          typename Components, typename Labels>
void updateComponents(ExecutionSpace const &exec_space,
                      ComponentOutEdges const &component_out_edges,
                      Components &components, Labels labels)
{
  auto computeNext = [&](int component) {
    int next_component = labels(component_out_edges(component).second);
    int next_next_component =
        labels(component_out_edges(next_component).second);

    if (next_next_component != component)
    {
      // The component's edge is unidirectional
      return next_component;
    }
    // The component's edge is bidirectional, uniquely resolve the bidirectional
    // edge
    return KokkosExt::min(component, next_component);
  };

  auto const n = labels.size();

  Kokkos::View<int, MemorySpace> num_components(
      "ArborX::hdbscan::num_components");

  Kokkos::View<int *, MemorySpace> final_component(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::hdbscan::final_component"),
      n);
  Kokkos::parallel_for(
      "ArborX::hdbscan::update_components",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0,
                                          n)(int const component) {
        int next_component = computeNext(component);

        if (next_component == component)
        {
          final_component(component) = component;
          components(Kokkos::atomic_fetch_add(&num_components(), 1)) =
              component;
          continue;
        }

        int prev_component;
        do
        {
          prev_component = next_component;
          next_component = compute_next(prev_component);
        } while (next_component != prev_component);

        final_component(component) = next_component;
      });
  auto num_components_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_components);
  Kokkos::resize(components, num_components_host());

  // Update component labels
  Kokkos::parallel_for(
      "ArborX::hdbscan::update_component_labels",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n)(int const i) {
        labels(i) = final_component(labels(i));
      });
}

struct Parameters
{
  // Print timers to standard output
  bool _print_timers = false;

  Parameters &setPrintTimers(bool print_timers)
  {
    _print_timers = print_timers;
    return *this;
  }
};
} // namespace HDBSCAN

template <typename ExecutionSpace, typename Primitives>
Kokkos::View<int *, typename Primitives::memory_space>
hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
        int core_min_size,
        HDBSCAN::Parameters const &parameters = HDBSCAN::Parameters())
{
  Kokkos::Profiling::pushRegion("ArborX::hdbscan");

  using MemorySpace = typename Primitives::memory_space;

  ARBORX_ASSERT(core_min_size >= 2);

  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = parameters._print_timers;
  auto timer_start = [&exec_space, verbose](Kokkos::Timer &timer) {
    if (verbose)
      exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space, verbose](Kokkos::Timer const &timer) {
    if (verbose)
      exec_space.fence();
    return timer.seconds();
  };

  int const n = primitives.extent_int(0);

  // Build the tree
  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::hdbscan::tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed["construction"] = timer_seconds(timer);

  // Compute core distances
  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::hdbscan::core_distances");
  auto const predicates = buildPredicates(primitives, core_min_size);
  Kokkos::View<float *, MemorySpace> core_distances(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::hdbscan::core_distances"),
      n);
  bvh.query(exec_space, predicates,
            HDBSCAN::CoreDistancesCallback<MemorySpace, Primitives>{
                primitives, core_distances});
  Kokkos::Profiling::popRegion();
  elapsed["core distances"] = timer_seconds(timer);

  Kokkos::View<Kokkos::Pair<int, int> *, MemorySpace> component_out_edges(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::hdbscan::component_out_edges"),
      n);
  Kokkos::View<int *, MemorySpace> cached_closest_neighbors(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::hdbscan::neighbors"), n);
  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::hdbscan::labels"), n);
  Kokkos::View<int *, MemorySpace> cached_closest_neighbors(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::hdbscan::components"),
      n);

  ArborX::iota(exec_space, labels);
  ArborX::iota(exec_space, cached_closest_neighbors);
  ArborX::iota(exec_space, components);

  Kokkos::Profiling::pushRegion("ArborX::hdbscan::boruvka");
  while (num_components > 1)
  {
    HDBSCAN::determineComponentEdges(exec_space, primitives, labels,
                                     cached_closest_neighbors,
                                     component_out_edges);

    HDBSCAN::updateComponents(exec_space, component_out_edges, components,
                              labels);
  }
  Kokkos::Profiling::popRegion();

  if (verbose)
  {
    printf("-- construction     : %10.3f\n", elapsed["construction"]);
    printf("-- core distances   : %10.3f\n", elapsed["core distances"]);
  }

  Kokkos::Profiling::popRegion();

  return labels;
}

} // namespace ArborX

#endif
