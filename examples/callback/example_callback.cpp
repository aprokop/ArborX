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

#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

using Geometry = ArborX::Experimental::Geometry<64>;
using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
using Box = ArborX::ExperimentalHyperGeometry::Box<3>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

std::ostream &operator<<(std::ostream &os, Point const &p)
{
  os << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
  return os;
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace exec;

  int const n = 100;
  Kokkos::View<Geometry *, ExecutionSpace> geometries(
      Kokkos::view_alloc("geometries", Kokkos::WithoutInitializing, exec), n);
  Kokkos::parallel_for(
      "Example::fill", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(int i) {
        switch (i % 3)
        {
        case 0:
          ::new (&geometries(i)) Geometry(Point{(float)i, (float)i, (float)i});
          break;
        case 1:
          ::new (&geometries(i))
              Geometry(Box{{(float)i, (float)i, (float)i},
                           {(float)i + 1, (float)i + 1, (float)i + 1}});
          break;
        case 2:
          ::new (&geometries(i)) Geometry(Triangle{
              {(float)i + 1, (float)i, (float)i},
              {(float)i, (float)i + 1, (float)i},
              {(float)i, (float)i, (float)i},
          });
          break;
        default:
          Kokkos::abort("bug");
        }
      });
  assert(n % 3 == 1); // last geometry is a point

  Box bounds;
  Kokkos::parallel_reduce(
      "Example::reduce_bounds", Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(int i, Box &u) {
        using ArborX::Details::expand;
        expand(u, geometries(i));
      },
      bounds);

  std::cout << "min_corner=" << bounds.minCorner()
            << ", max_corner=" << bounds.maxCorner() << '\n';
  // expected output: min_corner=(0,0,0), max_corner=(n-1,n-1,n-1)

  return 0;
}
