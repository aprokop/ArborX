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

#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

namespace Test
{
// NOTE only supporting Point and Box
using PrimitivePointOrBox = ArborX::Point<3>;
// using PrimitivePointOrBox = ArborX::Box;

// clang-format off
struct FakeBoundingVolume
{
};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, PrimitivePointOrBox) {}
template<int DIM>
KOKKOS_FUNCTION void expand(ArborX::Box<DIM> &, FakeBoundingVolume) { }

struct FakePredicateGeometry {};
KOKKOS_FUNCTION ArborX::Point<3> returnCentroid(FakePredicateGeometry) { return {}; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, FakeBoundingVolume) { return true; }
KOKKOS_FUNCTION float distance(FakePredicateGeometry, FakeBoundingVolume) { return 0.f; }
KOKKOS_FUNCTION bool intersects(FakePredicateGeometry, PrimitivePointOrBox) { return true; }
KOKKOS_FUNCTION float distance(FakePredicateGeometry, PrimitivePointOrBox) { return 0.f; }
// clang-format on

struct PoorManLambda
{
  template <class Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate, Value) const
  {}
};
} // namespace Test

template <>
struct ArborX::GeometryTraits::dimension<Test::FakeBoundingVolume>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<Test::FakeBoundingVolume>
{
  using type = float;
};

// Compile-only
void check_bounding_volume_and_predicate_geometry_type_requirements()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;
  using Tree = ArborX::BoundingVolumeHierarchy<
      MemorySpace, Test::PrimitivePointOrBox,
      ArborX::Experimental::DefaultIndexableGetter, Test::FakeBoundingVolume>;

  Kokkos::View<Test::PrimitivePointOrBox *, MemorySpace> primitives(
      "primitives", 0);
  Tree tree(ExecutionSpace{}, primitives);

  using SpatialPredicate =
      decltype(ArborX::intersects(Test::FakePredicateGeometry{}));
  Kokkos::View<SpatialPredicate *, MemorySpace> spatial_predicates(
      "spatial_predicates", 0);
  tree.query(ExecutionSpace{}, spatial_predicates, Test::PoorManLambda{});
#ifndef __NVCC__
  tree.query(ExecutionSpace{}, spatial_predicates,
             KOKKOS_LAMBDA(SpatialPredicate, auto){});
#endif

  using NearestPredicate =
      decltype(ArborX::nearest(Test::FakePredicateGeometry{}));
  Kokkos::View<NearestPredicate *, MemorySpace> nearest_predicates(
      "nearest_predicates", 0);
  tree.query(ExecutionSpace{}, nearest_predicates, Test::PoorManLambda{});
#ifndef __NVCC__
  tree.query(ExecutionSpace{}, nearest_predicates,
             KOKKOS_LAMBDA(NearestPredicate, auto){});
#endif
}

namespace Test
{

// clang-format off
struct FakePrimitiveGeometry {};

KOKKOS_FUNCTION void expand(ArborX::Box<3> &, FakePrimitiveGeometry) {}
KOKKOS_FUNCTION ArborX::Point<3> returnCentroid(FakePrimitiveGeometry) { return {}; }
// clang-format on

} // namespace Test

template <>
struct ArborX::GeometryTraits::dimension<Test::FakePrimitiveGeometry>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<Test::FakePrimitiveGeometry>
{
  using type = float;
};

// Compile-only
void check_hierarchy_for_custom_types()
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::View<Test::FakePrimitiveGeometry *, MemorySpace> primitives(
      "primitives", 0);
  ArborX::BoundingVolumeHierarchy tree(ExecutionSpace{}, primitives);
}
