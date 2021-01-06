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
#ifndef ARBORX_DETAILS_AAPLANE_HPP
#define ARBORX_DETAILS_AAPLANE_HPP

#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
struct AAPlane
{
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AAPlane() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr AAPlane(int axis, float location)
      : _axis(axis)
      , _location(location)
  {
  }

  KOKKOS_FUNCTION
  int axis() const { return _axis; }

  KOKKOS_FUNCTION
  int &axis() { return _axis; }

  KOKKOS_FUNCTION
  float location() const { return _location; }

  KOKKOS_FUNCTION
  float &location() { return _location; }

  int _axis = -1;
  float _location = 0.f;
};

} // namespace Details
} // namespace ArborX

#endif
