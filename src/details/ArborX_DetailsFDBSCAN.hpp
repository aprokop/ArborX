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

#ifndef ARBORX_DETAILSFDBSCAN_HPP
#define ARBORX_DETAILSFDBSCAN_HPP

#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <typename MemorySpace>
struct CountUpToN
{
  Kokkos::View<int *, MemorySpace> _counts;
  int _n;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int) const
  {
    auto i = getData(query);
    Kokkos::atomic_fetch_add(&_counts(i), 1);

    if (_counts(i) < _n)
      return ArborX::CallbackTreeTraversalControl::normal_continuation;

    // Once count reaches threshold, terminate the traversal.
    return ArborX::CallbackTreeTraversalControl::early_exit;
  }
};

template <typename MemorySpace, typename CorePointsType>
struct FDBSCANCallback
{
  UnionFind<MemorySpace> _union_find;
  CorePointsType _is_core_point;

  FDBSCANCallback(Kokkos::View<int *, MemorySpace> const &view,
                  CorePointsType is_core_point)
      : _union_find(view)
      , _is_core_point(is_core_point)
  {
  }

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int j) const
  {
    int const i = ArborX::getData(query);

    bool const is_border_point = !_is_core_point(i);
    bool const is_neighbor_core_point = _is_core_point(j);

    if (is_border_point && _union_find.representative(i) != i)
      return ArborX::CallbackTreeTraversalControl::early_exit;

    if (is_neighbor_core_point)
    {
      if (!is_border_point)
      {
        // Both points are core points
        _union_find.merge(i, j);
      }
      else
      {
        // A border point is connected to a core point
        _union_find.merge_into(i, j);
        return ArborX::CallbackTreeTraversalControl::early_exit;
      }
    }
    else if (!is_border_point)
    {
      // A core point is connected to a border point
      _union_find.merge_into(j, i);
    }
    else
    {
      // Both points are border points, do nothing
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};
} // namespace Details
} // namespace ArborX

#endif
