/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_HyperTriangle.hpp>
#include <ArborX_TypeErasedGeometry.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <iostream>
#include <random>
#include <vector>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct FirstOctant
{};

struct NearestToOrigin
{
  int k;
};

template <>
struct ArborX::AccessTraits<FirstOctant, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(FirstOctant) { return 1; }
  static KOKKOS_FUNCTION auto get(FirstOctant, std::size_t)
  {
    return ArborX::intersects(ArborX::Box{{{0, 0, 0}}, {{1, 1, 1}}});
  }
  using memory_space = MemorySpace;
};

template <>
struct ArborX::AccessTraits<NearestToOrigin, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(NearestToOrigin) { return 1; }
  static KOKKOS_FUNCTION auto get(NearestToOrigin d, std::size_t)
  {
    return ArborX::nearest(ArborX::Point{0, 0, 0}, d.k);
  }
  using memory_space = MemorySpace;
};

struct PrintfCallback
{
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, int primitive,
                                  OutputFunctor const &out) const
  {
    Kokkos::printf("Found %d from functor\n", primitive);
    out(primitive);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
  using Box = ArborX::ExperimentalHyperGeometry::Box<3>;
  using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  ExecutionSpace exec;

  int const n = 100;
  Kokkos::View<ArborX::Experimental::Geometry<64> *, ExecutionSpace> geometries(
      Kokkos::view_alloc("geometries", Kokkos::WithoutInitializing, exec), n);
  { // scope so that random number generation resources are released
    using RandomPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    RandomPool random_pool(123456);
    Kokkos::parallel_for(
        "Example::random_fill", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
          RandomPool::generator_type generator = random_pool.get_state();
          switch (generator.urand(3))
          {
          case 0:
            geometries(i) = Point{(float)i, (float)i, (float)i};
            break;
          case 1:
            geometries(i) = Box{{(float)i, (float)i, (float)i},
                                {(float)i + 1, (float)i + 1, (float)i + 1}};
            break;
          case 2:
            geometries(i) = Triangle{
                {(float)i + 1, (float)i, (float)i},
                {(float)i, (float)i + 1, (float)i},
                {(float)i, (float)i, (float)i},
            };
            break;
          default:
            Kokkos::abort("bug");
          }
          random_pool.free_state(generator);
        });
  }

  Box b;
  Kokkos::parallel_reduce(
      "Example::reduce_bounds", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(int i, Box &u) {
        using ArborX::Details::expand;
        expand(u, geometries(i));
      },
      b);

  return 0;
}
