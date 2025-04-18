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
#ifndef ARBORX_SEGMENT_HPP
#define ARBORX_SEGMENT_HPP

#include <ArborX_GeometryTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Clamp.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX::Experimental
{
template <int DIM, typename Coordinate = float>
struct Segment
{
  ArborX::Point<DIM, Coordinate> a;
  ArborX::Point<DIM, Coordinate> b;
};

template <int DIM, typename Coordinate>
KOKKOS_DEDUCTION_GUIDE Segment(Point<DIM, Coordinate>, Point<DIM, Coordinate>)
    -> Segment<DIM, Coordinate>;

template <int N, typename T>
KOKKOS_DEDUCTION_GUIDE Segment(T const (&)[N], T const (&)[N]) -> Segment<N, T>;

} // namespace ArborX::Experimental

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  using type = SegmentTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
