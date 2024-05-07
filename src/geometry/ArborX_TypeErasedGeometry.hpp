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

#ifndef ARBORX_TYPE_ERASED_GEOMETRY_HPP
#define ARBORX_TYPE_ERASED_GEOMETRY_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperBox.hpp>

#include <Kokkos_Macros.hpp>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

namespace ArborX::Experimental
{

template <size_t Capacity = 32, size_t Alignment = alignof(void *)>
class Geometry
{
  using Point = ExperimentalHyperGeometry::Point<3>;
  using Box = ExperimentalHyperGeometry::Box<3>;

  struct Concept
  {
    virtual KOKKOS_DEFAULTED_FUNCTION ~Concept() = default;
    virtual KOKKOS_FUNCTION void expand(Box &) const = 0;
    virtual KOKKOS_FUNCTION bool intersects(Box const &) const = 0;
    virtual KOKKOS_FUNCTION float distance(Point const &) const = 0;
    virtual KOKKOS_FUNCTION float distance(Box const &) const = 0;
    virtual KOKKOS_FUNCTION Point returnCentroid() const = 0;
    virtual KOKKOS_FUNCTION void clone(Concept *) const = 0;
    virtual KOKKOS_FUNCTION void move(Concept *) = 0;
  };

  template <class GeometryT>
  struct OwningModel : Concept
  {
    GeometryT geometry_;

    KOKKOS_FUNCTION OwningModel(GeometryT geometry)
        : geometry_(std::move(geometry))
    {}
    KOKKOS_FUNCTION void expand(Box &other) const override
    {
      using Details::expand;
      expand(other, geometry_);
    }
    KOKKOS_FUNCTION bool intersects(Box const &other) const override
    {
      using Details::intersects;
      return intersects(geometry_, other);
    }
    KOKKOS_FUNCTION float distance(Box const &other) const override
    {
      using Details::distance;
      return distance(other, geometry_);
    }
    KOKKOS_FUNCTION float distance(Point const &other) const override
    {
      using Details::distance;
      return distance(other, geometry_);
    }
    KOKKOS_FUNCTION Point returnCentroid() const override
    {
      using Details::returnCentroid;
      return returnCentroid(geometry_);
    }
    KOKKOS_FUNCTION void clone(Concept *memory) const override
    {
#ifdef KOKKOS_ENABLE_CXX17
      // ::new (static_cast<void *>(memory)) OwningModel(*this);
      ::new (const_cast<void*>(static_cast<const volatile void*>(memory)))
              OwningModel(*this);
#else
      std::construct_at(static_cast<OwningModel *>(memory), *this);
#endif
    }
    KOKKOS_FUNCTION void move(Concept *memory) override
    {
#ifdef KOKKOS_ENABLE_CXX17
      // ::new (static_cast<void *>(memory)) OwningModel(std::move(*this));
      ::new (const_cast<void*>(static_cast<const volatile void*>(memory)))
              OwningModel(std::move(*this));
#else
      std::construct_at(static_cast<OwningModel *>(memory), std::move(*this));
#endif
    }
  };

  KOKKOS_FUNCTION Concept *pimpl()
  {
    return reinterpret_cast<Concept *>(buffer_);
  }
  KOKKOS_FUNCTION Concept const *pimpl() const
  {
    return reinterpret_cast<Concept const *>(buffer_);
  }
  alignas(Alignment) std::byte buffer_[Capacity] = {};

  static KOKKOS_FUNCTION void poor_mans_raw_memory_swap(std::byte *a,
                                                        std::byte *b) noexcept
  {
    for (size_t i = 0; i < Capacity; ++i)
    {
      auto tmp(a[i]);
      a[i] = b[i];
      b[i] = tmp;
    }
  }

  friend KOKKOS_FUNCTION void expand(Box &other, Geometry const &geometry)
  {
    geometry.pimpl()->expand(other);
  }
  friend KOKKOS_FUNCTION void intersects(Geometry const &geometry,
                                         Box const &other)
  {
    geometry.pimpl()->intersects(other);
  }
  friend KOKKOS_FUNCTION auto distance(Geometry const &geometry,
                                       Box const &other)
  {
    return geometry.pimpl()->distance(other);
  }
  friend KOKKOS_FUNCTION auto distance(Geometry const &geometry,
                                       Point const &other)
  {
    return geometry.pimpl()->distance(other);
  }

  friend KOKKOS_FUNCTION Point returnCentroid(Geometry const &geometry)
  {
    return geometry.pimpl()->returnCentroid();
  }

public:
  KOKKOS_FUNCTION Geometry(Geometry const &other)
  {
    other.pimpl()->clone(pimpl());
  }
  KOKKOS_FUNCTION Geometry &operator=(Geometry const &other)
  {
    Geometry copy(other);
    poor_mans_raw_memory_swap(buffer_, copy.buffer_);
    return *this;
  }
  KOKKOS_FUNCTION Geometry(Geometry &&other) noexcept
  {
    other.pimpl()->move(pimpl());
  }
  KOKKOS_FUNCTION Geometry &operator=(Geometry &&other) noexcept
  {
    Geometry copy(std::move(other));
    poor_mans_raw_memory_swap(buffer_, copy.buffer_);
    return *this;
  }
  KOKKOS_FUNCTION ~Geometry()
  {
#ifdef KOKKOS_ENABLE_CXX17
    // pimpl()->~Concept();
#else
    std::destroy_at(pimpl());
#endif
  }

  template <class GeometryT>
  KOKKOS_FUNCTION Geometry(GeometryT geometry)
  {
    using Model = OwningModel<GeometryT>;
    static_assert(sizeof(Model) <= Capacity, "Given type is too large");
    static_assert(alignof(Model) <= Alignment, "Given type is misaligned");
#ifdef KOKKOS_ENABLE_CXX17
    // ::new (static_cast<void *>(pimpl())) Model(std::move(geometry));
    ::new (const_cast<void*>(static_cast<const volatile void*>(pimpl())))
            Model(std::move(geometry));
#else
    std::construct_at(static_cast<Model *>(pimpl()), std::move(geometry));
#endif
  }
};

} // namespace ArborX::Experimental

template <size_t Capacity, size_t Alignment>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Geometry<Capacity, Alignment>>
{
  static constexpr int value = 3;
};
template <size_t Capacity, size_t Alignment>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::Geometry<Capacity, Alignment>>
{
  using type = float;
};

#endif
