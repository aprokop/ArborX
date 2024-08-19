/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_QUARTENION_HPP
#define ARBORX_DETAILS_QUARTENION_HPP

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <type_traits>

namespace ArborX::Details
{

template <typename RealType>
class quaternion
{
  static_assert(std::is_floating_point_v<RealType> &&
                    std::is_same_v<RealType, std::remove_cv_t<RealType>>,
                "ArborX::quaternion can only be instantiated for a "
                "cv-unqualified floating point type");

private:
  RealType _re{};
  RealType _imi{};
  RealType _imj{};
  RealType _imk{};

public:
  using value_type = RealType;

  KOKKOS_DEFAULTED_FUNCTION quaternion() = default;
  KOKKOS_DEFAULTED_FUNCTION quaternion(quaternion const &) noexcept = default;
  KOKKOS_DEFAULTED_FUNCTION quaternion &
  operator=(quaternion const &) noexcept = default;

  template <typename RType,
            std::enable_if_t<std::is_convertible_v<RType, RealType>, int> = 0>
  KOKKOS_FUNCTION quaternion(quaternion<RType> const &other) noexcept
      // Intentionally do the conversions implicitly here so that users don't
      // get any warnings about narrowing, etc., that they would expect to get
      // otherwise
      : _re(other.real())
      , _imi(other.imag_i())
      , _imj(other.imag_j())
      , _imk(other.imag_k())
  {}

  /// Constructor that takes just the real part, and sets the imaginary part to
  /// zero
  KOKKOS_FUNCTION quaternion(RealType const &val) noexcept
      : _re(val)
      , _imi(RealType(0))
      , _imj(RealType(0))
      , _imk(RealType(0))
  {}

  //! Constructor that takes the real and imaginary parts
  KOKKOS_FUNCTION
  quaternion(RealType const &re, RealType const &imi, RealType const &imj,
             RealType const &imk) noexcept
      : _re(re)
      , _imi(imi)
      , _imj(imj)
      , _imk(imk)
  {}

  // Assignment operator (from a real number)
  KOKKOS_FUNCTION quaternion &operator=(RealType const &val) noexcept
  {
    _re = val;
    _imi = RealType(0);
    _imj = RealType(0);
    _imk = RealType(0);
    return *this;
  }

  KOKKOS_FUNCTION
  constexpr RealType &real() noexcept { return _re; }
  KOKKOS_FUNCTION
  constexpr RealType &imag_i() noexcept { return _imi; }
  KOKKOS_FUNCTION
  constexpr RealType &imag_j() noexcept { return _imj; }
  KOKKOS_FUNCTION
  constexpr RealType &imag_k() noexcept { return _imk; }

  KOKKOS_FUNCTION
  constexpr RealType real() const noexcept { return _re; }
  KOKKOS_FUNCTION
  constexpr RealType imag_i() const noexcept { return _imi; }
  KOKKOS_FUNCTION
  constexpr RealType imag_j() const noexcept { return _imj; }
  KOKKOS_FUNCTION
  constexpr RealType imag_k() const noexcept { return _imk; }

  KOKKOS_FUNCTION
  constexpr void real(RealType v) noexcept { _re = v; }
  KOKKOS_FUNCTION
  constexpr void imag_i(RealType v) noexcept { _imi = v; }
  KOKKOS_FUNCTION
  constexpr void imag_j(RealType v) noexcept { _imj = v; }
  KOKKOS_FUNCTION
  constexpr void imag_k(RealType v) noexcept { _imk = v; }

  constexpr KOKKOS_FUNCTION quaternion &
  operator+=(quaternion<RealType> const &src) noexcept
  {
    _re += src._re;
    _imi += src._imi;
    _imj += src._imj;
    _imk += src._imk;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION quaternion &
  operator+=(RealType const &src) noexcept
  {
    _re += src;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION quaternion &
  operator-=(quaternion<RealType> const &src) noexcept
  {
    _re -= src._re;
    _imi -= src._imi;
    _imj -= src._imj;
    _imk -= src._imk;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION quaternion &
  operator-=(RealType const &src) noexcept
  {
    _re -= src;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION quaternion &
  operator*=(quaternion<RealType> const &src) noexcept
  {
    RealType const re =
        _re * src._re - _imi * src._imi - _imj * src._imj - _imk * src._imk;
    RealType const imi =
        _re * src._imi + _imi * src._re + _imj * src._imk - _imk * src._imj;
    RealType const imj =
        _re * src._imj - _imi * src._imk + _imj * src._re + _imk * src._imi;
    RealType const imk =
        _re * src._imk + _imi * src._imj - _imj * src._imi + _imk * src._re;
    _re = re;
    _imi = imi;
    _imj = imj;
    _imk = imk;
    return *this;
  }

  constexpr KOKKOS_INLINE_FUNCTION quaternion &
  operator*=(RealType const &src) noexcept
  {
    _re *= src;
    _imi *= src;
    _imj *= src;
    _imk *= src;
    return *this;
  }
};

template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION bool operator==(quaternion<RealType1> const &x,
                                       quaternion<RealType2> const &y) noexcept
{
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y.real()) &&
         common_type(x.imag_i()) == common_type(y.imag_i()) &&
         common_type(x.imag_j()) == common_type(y.imag_j()) &&
         common_type(x.imag_k()) == common_type(y.imag_k());
}

template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION bool operator==(quaternion<RealType1> const &x,
                                       RealType2 const &y) noexcept
{
  using common_type = std::common_type_t<RealType1, RealType2>;
  return common_type(x.real()) == common_type(y.real()) &&
         common_type(x.imag_i()) == common_type(0) &&
         common_type(x.imag_j()) == common_type(0) &&
         common_type(x.imag_k()) == common_type(0);
}

template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION bool operator==(RealType1 const &x,
                                       quaternion<RealType2> const &y) noexcept
{
  return y == x;
}

template <class RealType>
KOKKOS_INLINE_FUNCTION RealType abs(quaternion<RealType> const &x)
{
  auto r = x.real() * x.real() + x.imag_i() * x.imag_i() +
           x.imag_j() * x.imag_j() + x.imag_k() * x.imag_k();
  return (r > RealType(0) ? Kokkos::sqrt(r) : RealType(0));
}

template <class RealType>
KOKKOS_INLINE_FUNCTION quaternion<RealType>
conj(quaternion<RealType> const &x) noexcept
{
  return quaternion<RealType>(x.real(), -x.imag_i(), -x.imag_j(), -x.imag_k());
}

template <class RealType>
KOKKOS_INLINE_FUNCTION quaternion<RealType>
inv(quaternion<RealType> const &x) noexcept
{
  auto r = x.real() * x.real() + x.imag_i() * x.imag_i() +
           x.imag_j() * x.imag_j() + x.imag_k() * x.imag_k();
  KOKKOS_ASSERT(r != 0);

  return quaternion<RealType>(x.real() / r, -x.imag_i() / r, -x.imag_j() / r,
                              -x.imag_k() / r);
}

template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION quaternion<std::common_type_t<RealType1, RealType2>>
operator+(quaternion<RealType1> const &x,
          quaternion<RealType2> const &y) noexcept
{
  return quaternion<std::common_type_t<RealType1, RealType2>>(
      x.real() + y.real(), x.imag_i() + y.imag_i(), x.imag_j() + y.imag_j(),
      x.imag_k() + y.imag_k());
}

template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION quaternion<std::common_type_t<RealType1, RealType2>>
operator*(quaternion<RealType1> const &x,
          quaternion<RealType2> const &y) noexcept
{
  return quaternion<std::common_type_t<RealType1, RealType2>>(
      x.real() * y.real() - x.imag_i() * y.imag_i() - x.imag_j() * y.imag_j() -
          x.imag_k() * y.imag_k(),
      x.real() * y.imag_i() + x.imag_i() * y.real() + x.imag_j() * y.imag_k() -
          x.imag_k() * y.imag_j(),
      x.real() * y.imag_j() - x.imag_i() * y.imag_k() + x.imag_j() * y.real() +
          x.imag_k() * y.imag_i(),
      x.real() * y.imag_k() + x.imag_i() * y.imag_j() -
          x.imag_j() * y.imag_i() + x.imag_k() * y.real());
}

template <class RealType>
KOKKOS_INLINE_FUNCTION quaternion<RealType>
operator-(quaternion<RealType> const &x) noexcept
{
  return quaternion<RealType>(-x.real(), -x.imag_i(), -x.imag_j(), -x.imag_k());
}

} // namespace ArborX::Details

#endif
