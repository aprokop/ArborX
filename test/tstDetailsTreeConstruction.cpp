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
#include "ArborXTest_LegacyTree.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_KDOP.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <detail/ArborX_IndexableGetter.hpp>
#include <detail/ArborX_MortonCode.hpp> // expandBits, morton32
#include <detail/ArborX_Node.hpp>       // ROPE SENTINEL
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TreeConstruction.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <misc/ArborX_SortUtils.hpp> // sortObjects

#include <boost/test/unit_test.hpp>

#include <array>
#include <bitset>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(assign_morton_codes, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // N is the number of Morton grid cells in each dimension for 64-bit Morton
  // codes.
  constexpr unsigned long long N = 1 << 21;
  std::vector<ArborX::Point<3>> points = {{{0.0, 0.0, 0.0}},
                                          {{0.25, 0.75, 0.25}},
                                          {{0.75, 0.25, 0.25}},
                                          {{0.75, 0.75, 0.25}},
                                          {{1.33, 2.33, 3.33}},
                                          {{1.66, 2.66, 3.66}},
                                          {{(float)N, (float)N, (float)N}}};
  int const n = points.size();
  // lower left front corner of the octant the points fall in
  std::vector<std::array<unsigned long long, 3>> anchors = {
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{1, 2, 3}},
      {{1, 2, 3}},
      {{N - 1, N - 1, N - 1}}};
  auto fun = [](std::array<unsigned long long, 3> const &anchor) {
    using ArborX::Details::expandBitsBy;
    auto i = std::get<0>(anchor);
    auto j = std::get<1>(anchor);
    auto k = std::get<2>(anchor);
    return 4 * expandBitsBy<2>(i) + 2 * expandBitsBy<2>(j) + expandBitsBy<2>(k);
  };
  std::vector<unsigned long long> ref(
      n, std::numeric_limits<unsigned long long>::max());
  for (int i = 0; i < n; ++i)
    ref[i] = fun(anchors[i]);
  // using points rather than boxes for convenience here but still have to
  // build the axis-aligned bounding boxes around them
  Kokkos::View<ArborX::Box<3> *, DeviceType> boxes("boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    ArborX::Details::expand(boxes_host(i), points[i]);
  Kokkos::deep_copy(boxes, boxes_host);

  typename DeviceType::execution_space space{};
  ArborX::Box<3> scene_host;
  ArborX::Details::TreeConstruction::calculateBoundingBoxOfTheScene(
      space, boxes, scene_host);

  BOOST_TEST(ArborX::Details::equals(
      scene_host, {{{0.0, 0.0, 0.0}}, {{(float)N, (float)N, (float)N}}}));

  LegacyValues<decltype(boxes), ArborX::Box<3>> values{boxes};
  ArborX::Details::Indexables indexables{
      values, ArborX::Experimental::DefaultIndexableGetter{}};

  // Test for a bug where missing move ref operator() in DefaultIndexableGetter
  // results in a default initialized indexable used in scene box calucation.
  scene_host = {};
  ArborX::Details::TreeConstruction::calculateBoundingBoxOfTheScene(
      space, indexables, scene_host);
  BOOST_TEST(ArborX::Details::equals(
      scene_host, {{{0.0, 0.0, 0.0}}, {{(float)N, (float)N, (float)N}}}));

  Kokkos::View<unsigned long long *, DeviceType> morton_codes("morton_codes",
                                                              n);
  ArborX::Details::projectOntoSpaceFillingCurve(
      space, boxes, ArborX::Experimental::Morton64(), scene_host, morton_codes);
  auto morton_codes_host = Kokkos::create_mirror_view(morton_codes);
  Kokkos::deep_copy(morton_codes_host, morton_codes);
  BOOST_TEST(morton_codes_host == ref, tt::per_element());
}

template <typename DeviceType>
class FillK
{
public:
  FillK(Kokkos::View<unsigned int *, DeviceType> k)
      : _k(k)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const { _k[i] = 4 - i; }

private:
  Kokkos::View<unsigned int *, DeviceType> _k;
};

BOOST_AUTO_TEST_CASE_TEMPLATE(indirect_sort, DeviceType, ARBORX_DEVICE_TYPES)
{
  // need a functionality that sort objects based on their Morton code and
  // also returns the indices in the original configuration

  // dummy unsorted Morton codes and corresponding sorted indices as reference
  // solution
  //
  using ExecutionSpace = typename DeviceType::execution_space;
  unsigned int const n = 4;
  Kokkos::View<unsigned int *, DeviceType> k("k", n);
  // Fill K with 4, 3, 2, 1
  FillK<DeviceType> fill_k_functor(k);
  Kokkos::parallel_for("fill_k", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       fill_k_functor);

  std::vector<size_t> ref = {3, 2, 1, 0};
  // sort Morton codes and object ids
  auto ids = ArborX::Details::sortObjects(ExecutionSpace{}, k);

  auto k_host = Kokkos::create_mirror_view(k);
  Kokkos::deep_copy(k_host, k);
  auto ids_host = Kokkos::create_mirror_view(ids);
  Kokkos::deep_copy(ids_host, ids);

  // check that they are sorted
  for (unsigned int i = 0; i < n; ++i)
    BOOST_TEST(k_host[i] == i + 1);
  // check that ids are properly ordered
  BOOST_TEST(ids_host == ref, tt::per_element());
}

template <typename Primitives, typename MortonCodes, typename LeafNodes,
          typename InternalNodes>
void generateHierarchy(Primitives primitives, MortonCodes sorted_morton_codes,
                       LeafNodes &leaf_nodes, InternalNodes &internal_nodes)
{
  using ArborX::Details::makeLeafNode;
  using DeviceType = typename MortonCodes::device_type;

  int const n = sorted_morton_codes.extent(0);

  Kokkos::realloc(leaf_nodes, n);
  Kokkos::realloc(internal_nodes, n - 1);

  typename DeviceType::execution_space space{};

  Kokkos::View<unsigned int *, DeviceType> permutation_indices(
      "Testing::indices", n);
  ArborX::Details::KokkosExt::iota(space, permutation_indices);

  using BoundingVolume =
      typename InternalNodes::value_type::bounding_volume_type;
  BoundingVolume bounds;
  ArborX::Details::TreeConstruction::generateHierarchy(
      space, LegacyValues<Primitives, BoundingVolume>{primitives},
      ArborX::Experimental::DefaultIndexableGetter{}, permutation_indices,
      sorted_morton_codes, leaf_nodes, internal_nodes, bounds);
}

template <typename LeafNodes, typename InternalNodes>
void traverse(LeafNodes leaf_nodes, InternalNodes internal_nodes, int root,
              std::ostringstream &sol)
{
  int n = leaf_nodes.extent(0);

  using ArborX::Details::ROPE_SENTINEL;
  std::function<void(int, std::ostream &)> traverseRopes;
  traverseRopes = [&leaf_nodes, &internal_nodes, n,
                   &traverseRopes](int node, std::ostream &os) {
    if (node < n)
    {
      os << "L" << node;
      int rope = leaf_nodes(node).rope;
      if (rope != ROPE_SENTINEL)
        traverseRopes(rope, os);
    }
    else
    {
      node = node - n;
      os << "I" << node;
      traverseRopes(internal_nodes(node).left_child, os);
    }
  };

  traverseRopes(root, sol);
}

namespace Test
{
struct FakePrimitive
{};
struct FakeBoundingVolume
{};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakePrimitive) {}
} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(example_tree_construction, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  if (!Kokkos::SpaceAccessibility<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible)
    return;

  // This is the example from the articles by Karras.
  // See
  // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
  int const n = 8;
  Kokkos::View<unsigned long long *, DeviceType> sorted_morton_codes(
      "sorted_morton_codes", n);
  std::vector<std::string> s{
      "00001", "00010", "00100", "00101", "10011", "11000", "11001", "11110",
  };
  for (int i = 0; i < n; ++i)
  {
    std::bitset<6> b(s[i]);
    BOOST_TEST_MESSAGE(b << "  " << b.to_ullong());
    sorted_morton_codes(i) = b.to_ullong();
  }

  Kokkos::View<Test::FakePrimitive *, DeviceType> primitives(
      "Testing::primitives", n);

  // Reference solution for the depth first search
  std::ostringstream ref;
  // clang-format off
  ref << "I0" << "I3" << "I1" << "L0" << "L1" << "I2" << "L2" << "L3"
      << "I4" << "L4" << "I5" << "I6" << "L5" << "L6" << "L7";
  // clang-format on
  BOOST_TEST_MESSAGE("ref = " << ref.str());

  using LeafNode = ArborX::Details::LeafNode<
      ArborX::PairValueIndex<Test::FakeBoundingVolume>>;
  using InternalNode = ArborX::Details::InternalNode<Test::FakeBoundingVolume>;

  Kokkos::View<LeafNode *, DeviceType> leaf_nodes("Testing::leaf_nodes", 0);
  Kokkos::View<InternalNode *, DeviceType> internal_nodes(
      "Testing::internal_nodes", 0);
  generateHierarchy(primitives, sorted_morton_codes, leaf_nodes,
                    internal_nodes);

  int const root = n;

  std::ostringstream sol;
  traverse(leaf_nodes, internal_nodes, root, sol);

  BOOST_TEST_MESSAGE("sol(node_with_left_child_and_rope) = " << sol.str());

  BOOST_TEST(sol.str() == ref.str());
}

namespace
{
struct BoxVolumeCost
{
  template <class Box>
  KOKKOS_FUNCTION auto operator()(Box const &box) const
  {
    float volume = 1;
    for (int d = 0; d < ArborX::GeometryTraits::dimension_v<Box>; ++d)
      volume *= box.maxCorner()[d] - box.minCorner()[d];
    return volume;
  }
};

// Snapshot topology on the device, then check it independently on the host.
// The recursive walk uses explicit children; expected ropes are derived from
// that walk rather than from the optimizer's parent-climbing algorithm.
template <class ExecutionSpace, class BVH, class CostFunction>
double checkOptimizedHierarchy(ExecutionSpace const &space, BVH const &tree,
                               CostFunction const &cost)
{
  using Friends = ArborX::Details::HappyTreeFriends;
  using Box = typename BVH::bounding_volume_type;
  int const n = tree.size();
  if (n < 2)
    return 0;
  Kokkos::View<int *[3], typename BVH::memory_space> topology("topology",
                                                              2 * n - 1);
  Kokkos::View<Box *, typename BVH::memory_space> bounds("bounds", 2 * n - 1);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, 2 * n - 1), KOKKOS_LAMBDA(int i) {
        topology(i, 2) = Friends::getRope(tree, i);
        if (i < n)
          ArborX::Details::expand(bounds(i), Friends::getIndexable(tree, i));
        else
        {
          topology(i, 0) = Friends::getLeftChild(tree, i);
          topology(i, 1) = Friends::getRightChild(tree, i);
          bounds(i) = Friends::getInternalBoundingVolume(tree, i);
        }
      });
  auto host_topology =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, topology);
  auto host_bounds =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bounds);
  std::vector<bool> visited(2 * n - 1, false);
  double total_cost = 0;
  std::function<Box(int, int)> visit = [&](int node, int rope) -> Box {
    BOOST_REQUIRE(node >= 0 && node < 2 * n - 1);
    BOOST_REQUIRE(!visited[node]);
    visited[node] = true;
    BOOST_TEST(host_topology(node, 2) == rope);
    if (node < n)
      return host_bounds(node);
    int const left = host_topology(node, 0);
    int const right = host_topology(node, 1);
    auto box = visit(left, right);
    ArborX::Details::expand(box, visit(right, rope));
    BOOST_TEST(ArborX::Details::equals(box, host_bounds(node)));
    total_cost += cost(box);
    return box;
  };
  auto root_bounds = visit(n, ArborX::Details::ROPE_SENTINEL);
  BOOST_TEST(ArborX::Details::equals(root_bounds, tree.bounds()));
  for (bool seen : visited)
    BOOST_TEST(seen);
  return total_cost;
}

// Use memory-space-generic callbacks, including when device memory is enabled.
template <class View>
struct RecordNeighborIn
{
  View found;
  int query;

  template <class Predicate, class Value>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value) const
  {
    ++found(query, value.index);
  }
};

template <int TreeletSize, int Dim, class DeviceType, class CostFunction>
void checkOptimization(int n, CostFunction const &cost, bool coincident = false,
                       int active_dimensions = Dim)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Box = ArborX::Box<Dim>;
  ExecutionSpace space;
  Kokkos::View<Box *, DeviceType> boxes("boxes", n);
  auto host_boxes = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < Dim; ++d)
    {
      // Deterministic irregular inputs, with overlapping and duplicate boxes.
      float const x = coincident || d >= active_dimensions
                          ? 0.f
                          : ((i * (17 + d * 12)) % 101) / 10.f;
      host_boxes(i).minCorner()[d] = x;
      host_boxes(i).maxCorner()[d] = x + (d < active_dimensions ? .5f : 0.f);
    }
  Kokkos::deep_copy(space, boxes, host_boxes);
  ArborX::BoundingVolumeHierarchy tree(
      space, ArborX::Experimental::attach_indices(boxes));
  auto const initial_bounds = tree.bounds();
  double previous = checkOptimizedHierarchy(space, tree, cost);
  for (int pass = 0; pass < 3; ++pass)
  {
    if constexpr (TreeletSize == 7 &&
                  std::is_same_v<CostFunction, ArborX::Details::SurfaceArea>)
      tree.optimize(space);
    else
      tree.template optimize<TreeletSize>(space, cost);
    double const current = checkOptimizedHierarchy(space, tree, cost);
    BOOST_TEST(current <= previous * (1 + 1e-6));
    BOOST_TEST(ArborX::Details::equals(initial_bounds, tree.bounds()));
    previous = current;
  }
  // Per-thread traversal currently requires at least two leaves.
  if (n < 2)
    return;
  // Radius queries must produce exactly the brute-force set, including no
  // duplicates. This exercises both rope skipping and descent after rewiring.
  Kokkos::View<int **, DeviceType> found("found", n, n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, n), KOKKOS_LAMBDA(int i) {
        ArborX::Sphere<Dim> sphere{boxes(i).minCorner(), 1.25f};
        tree.query(ArborX::Experimental::PerThread{},
                   ArborX::intersects(sphere),
                   RecordNeighborIn<decltype(found)>{found, i});
      });
  auto host_found =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, found);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    {
      ArborX::Sphere<Dim> sphere{host_boxes(i).minCorner(), 1.25f};
      BOOST_TEST(host_found(i, j) ==
                 (ArborX::intersects(sphere, host_boxes(j)) ? 1 : 0));
    }
}
} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(optimize_hierarchy, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  for (int n : {0, 1, 2, 3, 7, 8, 31, 257})
    checkOptimization<7, 3, DeviceType>(n, ArborX::Details::SurfaceArea{});
  checkOptimization<2, 3, DeviceType>(31, ArborX::Details::SurfaceArea{});
  checkOptimization<4, 2, DeviceType>(31, ArborX::Details::SurfaceArea{});
  checkOptimization<12, 4, DeviceType>(31, BoxVolumeCost{});
  checkOptimization<7, 3, DeviceType>(31, ArborX::Details::SurfaceArea{}, true);
  // Exercise optimization and radius traversal on planar, collinear, and
  // coincident point boxes, whose bounds remain degenerate after merging.
  for (int active_dimensions : {0, 1, 2})
    checkOptimization<7, 3, DeviceType>(31, ArborX::Details::SurfaceArea{},
                                        false, active_dimensions);
}

BOOST_AUTO_TEST_CASE(surface_area)
{
  auto const surface_area = ArborX::Details::SurfaceArea{};
  BOOST_TEST(surface_area(ArborX::Box<1>{{{0}}, {{3}}}) == 2);
  BOOST_TEST(surface_area(ArborX::Box<2>{{{0, 0}}, {{2, 3}}}) == 10);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{0, 0, 0}}, {{2, 3, 4}}}) == 52);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{0, 0, 0}}, {{0, 3, 4}}}) == 24);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 1, 1}}, {{1, 1, 1}}}) == 0);
  // The collapsed axes may occur anywhere; translating the box is immaterial.
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 2, 3}}, {{4, 2, 7}}}) == 24);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 2, 3}}, {{4, 6, 3}}}) == 24);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 2, 3}}, {{4, 2, 3}}}) == 6);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 2, 3}}, {{1, 5, 3}}}) == 6);
  BOOST_TEST(surface_area(ArborX::Box<3>{{{1, 2, 3}}, {{1, 2, 6}}}) == 6);
  BOOST_TEST(surface_area(ArborX::Box<2>{{{0, 0}}, {{0, 3}}}) == 6);
  BOOST_TEST(surface_area(ArborX::Box<2>{{{0, 0}}, {{0, 0}}}) == 0);
  BOOST_TEST(surface_area(ArborX::Box<1>{{{3}}, {{3}}}) == 0);
  BOOST_TEST(surface_area(ArborX::Box<4>{{{0, 0, 0, 0}}, {{2, 3, 4, 5}}}) ==
             308);
  BOOST_TEST(surface_area(ArborX::Box<4>{{{0, 0, 0, 0}}, {{2, 3, 4, 0}}}) ==
             48);
  BOOST_TEST(surface_area(ArborX::Box<4>{{{0, 0, 0, 0}}, {{0, 3, 0, 4}}}) ==
             24);
  // A nonzero thickness must not be classified as a collapsed dimension.
  BOOST_TEST(
      surface_area(ArborX::Box<3, double>{{{0, 0, 0}}, {{1, 1, 1e-12}}}) > 2);
  // The volume overflows float, but the surface area does not.
  auto const large_area =
      surface_area(ArborX::Box<3>{{{0, 0, 0}}, {{1e13f, 1e13f, 1e13f}}});
  BOOST_TEST(std::isfinite(large_area));
  BOOST_TEST(large_area > 5e26f);
}

namespace
{
template <class MemorySpace>
struct OptimizationFixture
{
  using memory_space = MemorySpace;
  using bounding_volume_type = ArborX::Box<3>;
  Kokkos::View<ArborX::Details::LeafNode<bounding_volume_type> *, MemorySpace>
      _leaf_nodes;
  Kokkos::View<ArborX::Details::InternalNode<bounding_volume_type> *,
               MemorySpace>
      _internal_nodes;
  ArborX::Experimental::DefaultIndexableGetter _indexable_getter;
  bounding_volume_type scene;

  KOKKOS_FUNCTION int size() const { return _leaf_nodes.extent_int(0); }
  auto bounds() const { return scene; }
};
} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(optimize_treelet_cost, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  typename DeviceType::execution_space space;
  for (bool crossed : {true, false})
  {
    OptimizationFixture<MemorySpace> tree;
    tree._leaf_nodes = decltype(tree._leaf_nodes)("leaves", 4);
    tree._internal_nodes = decltype(tree._internal_nodes)("nodes", 3);
    auto leaves = Kokkos::create_mirror_view(tree._leaf_nodes);
    auto nodes = Kokkos::create_mirror_view(tree._internal_nodes);
    // Start with crossed clusters {0, 100} and {1, 101}. Agglomeration should
    // replace them with {0, 1} and {100, 101}, strictly reducing the cost.
    for (int i = 0; i < 4; ++i)
    {
      float const x = (i / 2) * 100 + i % 2;
      if (crossed)
        leaves(i).value = {{{x, 0, 0}}, {{x + .5f, .5f, .5f}}};
      else
      {
        // Greedy merging costs 470 here, exceeding the original cost of 446.
        float const positions[4][3] = {
            {4, 1, 9}, {3, 9, 9}, {5, 4, 6}, {4, 8, 0}};
        for (int d = 0; d < 3; ++d)
        {
          leaves(i).value.minCorner()[d] = positions[i][d];
          leaves(i).value.maxCorner()[d] = positions[i][d] + 1;
        }
      }
      ArborX::Details::expand(tree.scene, leaves(i).value);
    }
    nodes(0).left_child = 5;
    nodes(1).left_child = 0;
    nodes(2).left_child = 1;
    nodes(0).bounding_volume = tree.scene;
    nodes(1).bounding_volume = leaves(0).value;
    ArborX::Details::expand(nodes(1).bounding_volume, leaves(2).value);
    nodes(2).bounding_volume = leaves(1).value;
    ArborX::Details::expand(nodes(2).bounding_volume, leaves(3).value);
    nodes(1).rope = 6;
    leaves(0).rope = 2;
    leaves(2).rope = 6;
    leaves(1).rope = 3;
    Kokkos::deep_copy(space, tree._leaf_nodes, leaves);
    Kokkos::deep_copy(space, tree._internal_nodes, nodes);
    auto const cost = ArborX::Details::SurfaceArea{};
    double const before = checkOptimizedHierarchy(space, tree, cost);
    ArborX::Details::optimizeHierarchy<4>(space, tree, tree._leaf_nodes,
                                          tree._internal_nodes,
                                          tree._indexable_getter, cost);
    double const after = checkOptimizedHierarchy(space, tree, cost);
    if (crossed)
      BOOST_TEST(after < before);
    else
    {
      BOOST_TEST(before == 446);
      BOOST_TEST(after == before);
      auto after_nodes = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, tree._internal_nodes);
      BOOST_TEST(after_nodes(0).left_child == 5);
      BOOST_TEST(after_nodes(1).left_child == 0);
      BOOST_TEST(after_nodes(2).left_child == 1);
    }
    // A second pass must preserve the result in both fixtures.
    ArborX::Details::optimizeHierarchy<4>(space, tree, tree._leaf_nodes,
                                          tree._internal_nodes,
                                          tree._indexable_getter, cost);
    BOOST_TEST(checkOptimizedHierarchy(space, tree, cost) == after);
  }
}

namespace
{
struct ConstantNodeCost
{
  template <class Volume>
  KOKKOS_FUNCTION int operator()(Volume const &) const
  {
    return 1;
  }
};
} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(optimize_skewed_tree, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using Box = ArborX::Box<3>;
  typename DeviceType::execution_space space;
  constexpr int n = 64;
  for (bool right_spine : {false, true})
  {
    OptimizationFixture<MemorySpace> tree;
    tree._leaf_nodes = decltype(tree._leaf_nodes)("leaves", n);
    tree._internal_nodes = decltype(tree._internal_nodes)("nodes", n - 1);
    auto leaves = Kokkos::create_mirror_view(tree._leaf_nodes);
    auto nodes = Kokkos::create_mirror_view(tree._internal_nodes);
    std::vector<int> right(n - 1);
    for (int i = 0; i < n; ++i)
      leaves(i).value = {{{float(i), 0, 0}}, {{float(i) + .5f, 1, 1}}};
    for (int i = 0; i < n - 1; ++i)
    {
      int const continuation = i == n - 2 ? n - 1 : n + i + 1;
      nodes(i).left_child = right_spine ? i : continuation;
      right[i] = right_spine ? continuation : i;
    }
    std::function<Box(int, int)> initialize = [&](int node, int rope) -> Box {
      if (node < n)
      {
        leaves(node).rope = rope;
        return leaves(node).value;
      }
      auto &internal = nodes(node - n);
      internal.rope = rope;
      auto box = initialize(internal.left_child, right[node - n]);
      ArborX::Details::expand(box, initialize(right[node - n], rope));
      return internal.bounding_volume = box;
    };
    tree.scene = initialize(n, ArborX::Details::ROPE_SENTINEL);
    Kokkos::deep_copy(space, tree._leaf_nodes, leaves);
    Kokkos::deep_copy(space, tree._internal_nodes, nodes);
    ConstantNodeCost cost;
    BOOST_TEST(checkOptimizedHierarchy(space, tree, cost) == n - 1);
    // Equal costs preserve the skewed topology, exercising long rope chains.
    ArborX::Details::optimizeHierarchy<7>(space, tree, tree._leaf_nodes,
                                          tree._internal_nodes,
                                          tree._indexable_getter, cost);
    BOOST_TEST(checkOptimizedHierarchy(space, tree, cost) == n - 1);
  }
}

namespace
{
struct KDOPCost
{
  KOKKOS_FUNCTION float
  operator()(ArborX::Experimental::KDOP<3, 14> const &volume) const
  {
    float cost = 0;
    for (int d = 0; d < volume.n_directions; ++d)
      cost += volume._max_values[d] - volume._min_values[d];
    return cost;
  }
};
} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(optimize_custom_bounding_volume, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  typename DeviceType::execution_space space;
  using Point = ArborX::Point<3>;
  constexpr int n = 17;
  Kokkos::View<Point *, DeviceType> points("points", n);
  auto host_points = Kokkos::create_mirror_view(points);
  for (int i = 0; i < n; ++i)
    host_points(i) = {{float(i), float(i % 3), 0}};
  Kokkos::deep_copy(space, points, host_points);
  ArborX::BoundingVolumeHierarchy<MemorySpace, ArborX::PairValueIndex<Point>,
                                  ArborX::Experimental::DefaultIndexableGetter,
                                  ArborX::Experimental::KDOP<3, 14>>
      tree(space, ArborX::Experimental::attach_indices(points));
  tree.optimize(space, KDOPCost{});
  Kokkos::View<int **, DeviceType> found("found", n, n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, n), KOKKOS_LAMBDA(int i) {
        ArborX::Box<3> box;
        for (int d = 0; d < 3; ++d)
        {
          box.minCorner()[d] = points(i)[d] - 2.f;
          box.maxCorner()[d] = points(i)[d] + 2.f;
        }
        tree.query(ArborX::Experimental::PerThread{}, ArborX::intersects(box),
                   RecordNeighborIn<decltype(found)>{found, i});
      });
  auto host_found =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, found);
  for (int i = 0; i < n; ++i)
  {
    ArborX::Box<3> box;
    for (int d = 0; d < 3; ++d)
    {
      box.minCorner()[d] = host_points(i)[d] - 2.f;
      box.maxCorner()[d] = host_points(i)[d] + 2.f;
    }
    for (int j = 0; j < n; ++j)
      BOOST_TEST(host_found(i, j) ==
                 (ArborX::intersects(box, host_points(j)) ? 1 : 0));
  }
}
