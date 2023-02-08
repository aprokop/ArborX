/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_NEIGHBOR_LIST_HPP
#define ARBORX_NEIGHBOR_LIST_HPP

#include <ArborX_DetailsHalfTraversal.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp> // reallocWithoutInitializing
#include <ArborX_DetailsUtils.hpp>                // exclusivePrefixSum
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Experimental
{

struct NeighborListPredicateGetter
{
  float _radius;

  KOKKOS_FUNCTION auto operator()(Box b) const
  {
    return intersects(Sphere{b.minCorner(), _radius});
  }
};

template <class ExecutionSpace, class Primitives, class Indices, class Counts>
int findHalfNeighborList2D(ExecutionSpace const &space,
                           Primitives const &primitives, float radius,
                           Indices &indices, Counts &counts)
{
  KokkosExt::ScopedProfileRegion guard(
      "ArborX::Experimental::HalfNeighborList2D");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  int const buffer_size =
      indices.extent_int(0) == n ? indices.extent_int(1) : 0;
  if (buffer_size > 0)
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::Experimental::HalfNeighborList::Count");

    KokkosExt::reallocWithoutInitializing(space, counts, n);
    Kokkos::deep_copy(space, counts, 0);
    HalfTraversal(
        space, bvh,
        KOKKOS_LAMBDA(int, int j) { Kokkos::atomic_increment(&counts(j)); },
        NeighborListPredicateGetter{radius});

    Kokkos::Profiling::popRegion();
  }
  else
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::Experimental::HalfNeighborList::CountAndFill");

    KokkosExt::reallocWithoutInitializing(space, counts, n);
    Kokkos::deep_copy(space, counts, 0);
    HalfTraversal(
        space, bvh,
        KOKKOS_LAMBDA(int i, int j) {
          int const pos = Kokkos::atomic_fetch_inc(&counts(j));
          if (pos < buffer_size)
          {
            indices(j, pos) = i;
          }
        },
        NeighborListPredicateGetter{radius});

    Kokkos::Profiling::popRegion();
  }
  auto const max_neighbors = max(space, counts);
  if (max_neighbors <= buffer_size)
  {
    return max_neighbors;
  }

  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList::Fill");

  KokkosExt::reallocWithoutInitializing(space, indices, n, max_neighbors);
  Kokkos::deep_copy(space, counts, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(j, Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();

  return max_neighbors;
}

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void findHalfNeighborList(ExecutionSpace const &space,
                          Primitives const &primitives, float radius,
                          Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::HalfNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int, int j) { Kokkos::atomic_increment(&offsets(j)); },
      NeighborListPredicateGetter{radius});
  exclusivePrefixSum(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::HalfNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

template <class ExecutionSpace, class Primitives, class Indices, class Counts>
void findFullNeighborList2D(ExecutionSpace const &space,
                            Primitives const &primitives, float radius,
                            Indices &indices, Counts &counts)
{
  KokkosExt::ScopedProfileRegion guard(
      "ArborX::Experimental::FullNeighborList2D");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  int const buffer_size =
      indices.extent_int(0) == n ? indices.extent_int(1) : 0;
  if (buffer_size > 0)
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::Experimental::FullNeighborList2D::Count");

    KokkosExt::reallocWithoutInitializing(space, counts, n);
    Kokkos::deep_copy(space, counts, 0);
    HalfTraversal(
        space, bvh,
        KOKKOS_LAMBDA(int i, int j) {
          Kokkos::atomic_increment(&counts(i));
          Kokkos::atomic_increment(&counts(j));
        },
        NeighborListPredicateGetter{radius});

    Kokkos::Profiling::popRegion();
  }
  else
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::Experimental::FullNeighborList2D::CountAndFill");

    KokkosExt::reallocWithoutInitializing(space, counts, n);
    Kokkos::deep_copy(space, counts, 0);
    HalfTraversal(
        space, bvh,
        KOKKOS_LAMBDA(int i, int j) {
          Kokkos::atomic_increment(&counts(i));
          int const pos = Kokkos::atomic_fetch_inc(&counts(j));
          if (pos < buffer_size)
          {
            indices(j, pos) = i;
          }
        },
        NeighborListPredicateGetter{radius});

    Kokkos::Profiling::popRegion();
  }
  auto max_neighbors = max(space, counts);
  if (max_neighbors <= buffer_size)
  {
    return;
  }
  // NOTE can do better if counting half fit in the buffer

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::FullNeighbor2DList::Fill");

  KokkosExt::reallocWithoutInitializing(space, indices, n, max_neighbors);
  Kokkos::deep_copy(space, counts, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(j, Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::FullNeighborList2D::Copy");

  auto counts_copy = KokkosExt::clone(space, counts, counts.label() + "_copy");
  Kokkos::parallel_for(
      "ArborX::Experimental::FullNeighborList2D::Copy",
      Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          typename Kokkos::TeamPolicy<ExecutionSpace>::member_type const
              &member) {
        auto const i = member.league_rank();
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, counts_copy(i)), [&](int j) {
              int const k = indices(i, j);
              indices(k, Kokkos::atomic_fetch_inc(&counts(k))) = i;
            });
      });

  Kokkos::Profiling::popRegion();
}

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void findFullNeighborList(ExecutionSpace const &space,
                          Primitives const &primitives, float radius,
                          Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::FullNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        Kokkos::atomic_increment(&offsets(i));
        Kokkos::atomic_increment(&offsets(j));
      },
      NeighborListPredicateGetter{radius});
  exclusivePrefixSum(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::FullNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Copy");

  auto counts_copy = KokkosExt::clone(space, counts, counts.label() + "_copy");
  Kokkos::parallel_for(
      "ArborX::Experimental::FullNeighborList::Copy",
      Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          typename Kokkos::TeamPolicy<ExecutionSpace>::member_type const
              &member) {
        auto const i = member.league_rank();
        auto const first = offsets(i);
        auto const last = counts_copy(i);
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, last - first), [&](int j) {
              int const k = indices(first + j);
              indices(Kokkos::atomic_fetch_inc(&counts(k))) = i;
            });
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Experimental

#endif
