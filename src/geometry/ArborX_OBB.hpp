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
#ifndef ARBORX_OBB_HPP
#define ARBORX_OBB_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsContainers.hpp> // StaticVector
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsSymmetricSVD.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_DetailsVector.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX::Experimental
{

namespace
{
template <typename T>
using UnmanagedViewWrapper =
    Kokkos::View<T, Kokkos::AnonymousSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <int DIM, typename Coordinate>
struct Rotation
{
  // Rotation (i.e., ortho-normal) matrix
  // Normalized eigenvectors are stores in columns.
  // The inverse is the transpose.
  Coordinate _matrix[DIM * DIM];

  KOKKOS_FUNCTION
  Rotation()
  {
    auto U = matrix();
    // Set matrix to identity
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        U(i, j) = (i == j ? 1 : 0);
  }

  KOKKOS_FUNCTION
  Rotation(UnmanagedViewWrapper<Coordinate[DIM][DIM]> const &A)
  {
    auto U = matrix();
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        U(i, j) = A(i, j);
  }

  KOKKOS_FUNCTION auto matrix()
  {
    return UnmanagedViewWrapper<Coordinate[DIM][DIM]>(_matrix);
  }

  KOKKOS_FUNCTION auto matrix() const
  {
    return UnmanagedViewWrapper<const Coordinate[DIM][DIM]>(_matrix);
  }

  template <typename Point>
  KOKKOS_FUNCTION auto rotate(Point const &point) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);

    auto const U = matrix();

    // Use matrix transpose
    Point p;
    for (int row = 0; row < DIM; ++row)
    {
      p[row] = 0;
      for (int col = 0; col < DIM; ++col)
        p[row] += U(col, row) * point[col];
    }
    return p;
  }

  template <typename Point>
  KOKKOS_FUNCTION auto rotate_back(Point const &point) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);

    auto const U = matrix();

    Point p;
    for (int row = 0; row < DIM; ++row)
    {
      p[row] = 0;
      for (int col = 0; col < DIM; ++col)
        p[row] += U(row, col) * point[col];
    }
    return p;
  }
};

template <typename Coordinate>
struct Rotation<3, Coordinate>
{
  Coordinate _q[4] = {1};

  KOKKOS_DEFAULTED_FUNCTION Rotation() = default;

  KOKKOS_FUNCTION
  Rotation(UnmanagedViewWrapper<Coordinate[3][3]> const &U)
  {
    auto r11 = U(0, 0);
    auto r12 = U(0, 1);
    auto r13 = U(0, 2);
    auto r21 = U(1, 0);
    auto r22 = U(1, 1);
    auto r23 = U(1, 2);
    auto r31 = U(2, 0);
    auto r32 = U(2, 1);
    auto r33 = U(2, 2);

    using Kokkos::sqrt;
    auto q0 = sqrt(1 + r11 + r22 + r33) / 2;
    auto q1 = sqrt(1 + r11 - r22 - r33) / 2;
    auto q2 = sqrt(1 - r11 + r22 - r33) / 2;
    auto q3 = sqrt(1 - r11 - r22 + r33) / 2;

    if (q0 >= q1 && q0 >= q2 && q0 >= q3)
    {
      auto s = 1 / (4 * q0);
      _q[0] = q0;
      _q[1] = (r32 - r23) * s;
      _q[2] = (r13 - r31) * s;
      _q[3] = (r21 - r12) * s;
    }
    else if (q1 >= q0 && q1 >= q2 && q1 >= q3)
    {
      auto s = 1 / (4 * q1);
      _q[0] = (r32 - r23) * s;
      _q[1] = q1;
      _q[2] = (r12 + r21) * s;
      _q[2] = (r13 + r31) * s;
    }
    else if (q2 >= q0 && q2 >= q1 && q2 >= q3)
    {
      auto s = 1 / (4 * q2);
      _q[0] = (r13 - r31) * s;
      _q[1] = (r12 + r21) * s;
      _q[2] = q2;
      _q[3] = (r23 + r32) * s;
    }
    else
    {
      auto s = 1 / (4 * q3);
      _q[0] = (r21 - r12) * s;
      _q[1] = (r13 + r31) * s;
      _q[2] = (r23 + r32) * s;
      _q[3] = q3;
    }

    _q[1] = -_q[1];
    _q[2] = -_q[2];
    _q[3] = -_q[3];
  }

  template <typename Point>
  KOKKOS_FUNCTION auto rotate(Point point) const
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);
    auto const &v = point;

    // w = real part
    // r = imaginary part vector
    auto const w = _q[0];
    Coordinate r[3] = {_q[1], _q[2], _q[3]};
    // Then, Rodrigues formula is
    // v = v + 2r x (r x v + wv)
    Coordinate z[3] = {(r[1] * v[2] - r[2] * v[1]) + w * v[0],
                       (r[2] * v[0] - r[0] * v[2]) + w * v[1],
                       (r[0] * v[1] - r[1] * v[0]) + w * v[2]};
    return Point{v[0] + 2 * (r[1] * z[2] - r[2] * z[1]),
                 v[1] + 2 * (r[2] * z[0] - r[0] * z[2]),
                 v[2] + 2 * (r[0] * z[1] - r[1] * z[0])};
  }

  template <typename Point>
  KOKKOS_FUNCTION auto rotate_back(Point const &point) const
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);
    auto const &v = point;

    // The difference with rotate() is conjuction of _q
    auto const w = _q[0];
    Coordinate r[3] = {-_q[1], -_q[2], -_q[3]};
    Coordinate z[3] = {(r[1] * v[2] - r[2] * v[1]) + w * v[0],
                       (r[2] * v[0] - r[0] * v[2]) + w * v[1],
                       (r[0] * v[1] - r[1] * v[0]) + w * v[2]};
    return Point{v[0] + 2 * (r[1] * z[2] - r[2] * z[1]),
                 v[1] + 2 * (r[2] * z[0] - r[0] * z[2]),
                 v[2] + 2 * (r[0] * z[1] - r[1] * z[0])};
  }
};

template <int DIM, typename Coordinate>
KOKKOS_INLINE_FUNCTION bool operator==(Rotation<DIM, Coordinate> const &l,
                                       Rotation<DIM, Coordinate> const &r)
{
  if constexpr (DIM != 3)
  {
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        if (l.matrix()(i, j) != r.matrix()(i, j))
          return false;
    return true;
  }
  else
  {
    return l._q == r._q;
  }
}

} // namespace

template <int DIM, typename Coordinate = float>
struct OBB
{
  Rotation<DIM, Coordinate> _rotation;

  // Box in the new coordinate system
  ExperimentalHyperGeometry::Box<DIM, Coordinate> _box;

  KOKKOS_DEFAULTED_FUNCTION OBB() = default;

  template <typename View, typename = std::enable_if_t<Kokkos::is_view_v<View>>>
  KOKKOS_FUNCTION OBB(View const &points)
  {
    static_assert(View::rank == 1);

    using Point = typename View::value_type;
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);
    static_assert(
        std::is_same_v<GeometryTraits::coordinate_type_t<Point>, Coordinate>);

    int const n = points.size();
    KOKKOS_ASSERT(n > 0);

    if (n == 1)
    {
      Details::expand(_box, points[0]);
      return;
    }

    // Compute covariance matrix
    Coordinate cov_data[DIM * DIM];
    UnmanagedViewWrapper<Coordinate[DIM][DIM]> cov(cov_data);
    {
      // There's a way to optimize computing covariance in a single loop,
      // however, it is likely less precise
      Coordinate mean[DIM];
      for (int i = 0; i < DIM; ++i)
      {
        mean[i] = 0;
        for (int k = 0; k < n; ++k)
          mean[i] += points(k)[i];
        mean[i] /= n;
      }
      for (int i = 0; i < DIM; ++i)
        for (int j = i; j < DIM; ++j)
        {
          cov(i, j) = 0;
          for (int k = 0; k < n; ++k)
            cov(i, j) += (points(k)[i] - mean[i]) * (points(k)[j] - mean[j]);
          cov(i, j) /= n;
        }

      // FIXME: not sure we need to do this (depends on SymSVD implementation)
      for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < i; ++j)
          cov(i, j) = cov(j, i);
    }

    // Find the orthonormalized eigenvectors and store them
    Coordinate d_data[DIM];
    Coordinate u_data[DIM * DIM];
    UnmanagedViewWrapper<Coordinate[DIM]> D(d_data);
    UnmanagedViewWrapper<Coordinate[DIM][DIM]> U(u_data);

    Interpolation::Details::symmetricSVDKernel(cov, D, U);

    _rotation = Rotation(U);

    // Find extents by projecting points
    for (int i = 0; i < n; ++i)
      Details::expand(_box, rotate(points(i)));
  }

  template <int N>
  KOKKOS_FUNCTION
  OBB(ExperimentalHyperGeometry::Point<DIM, Coordinate> const (&points)[N])
      : OBB(UnmanagedViewWrapper<
            const ExperimentalHyperGeometry::Point<DIM, Coordinate>[N]>(points))
  {}

  template <typename Point>
  KOKKOS_FUNCTION auto rotate(Point const &point) const
  {
    return _rotation.rotate(point);
  }

  template <typename Point>
  KOKKOS_FUNCTION auto rotate_back(Point const &point) const
  {
    return _rotation.rotate_back(point);
  }

  KOKKOS_FUNCTION auto corners() const
  {
    static_assert(DIM == 2 || DIM == 3);

    KOKKOS_ASSERT(Details::isValid(*this));

    using Point = ExperimentalHyperGeometry::Point<DIM, Coordinate>;

    Coordinate const xs[2] = {_box.minCorner()[0], _box.maxCorner()[0]};
    Coordinate const ys[2] = {_box.minCorner()[1], _box.maxCorner()[1]};
    int const num_unique_xs = (xs[0] != xs[1]) + 1;
    int const num_unique_ys = (ys[0] != ys[1]) + 1;
    if constexpr (DIM == 2)
    {
      Details::StaticVector<Point, 4> points;
      for (int i = 0; i < num_unique_xs; ++i)
        for (int j = 0; j < num_unique_ys; ++j)
          points.emplaceBack(rotate_back(Point{xs[i], ys[j]}));
      return points;
    }
    else
    {
      Coordinate const zs[2] = {_box.minCorner()[2], _box.maxCorner()[2]};
      int const num_unique_zs = (zs[0] != zs[1]) + 1;
      Details::StaticVector<Point, 8> points;
      for (int i = 0; i < num_unique_xs; ++i)
        for (int j = 0; j < num_unique_ys; ++j)
          for (int k = 0; k < num_unique_zs; ++k)
            points.emplaceBack(rotate_back(Point{xs[i], ys[j], zs[k]}));
      return points;
    }
  }

  // FIXME: only necessary to not modify the tests
  KOKKOS_FUNCTION explicit operator Box() const
  {
    Box box;
    Details::expand(box, *this);
    return box;
  }
};

} // namespace ArborX::Experimental

template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::OBB<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::tag<ArborX::Experimental::OBB<DIM, Coordinate>>
{
  using type = OBBTag;
};
template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::OBB<DIM, Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{

template <typename OBB>
struct equals<OBBTag, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(OBB const &l, OBB const &r)
  {
    return l._rotation == r._rotation && Details::equals(l._box, r._box);
  }
};

template <typename OBB>
struct isValid<OBBTag, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(OBB const &obb)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;

    // Slight modification on isValid(Box) in that Box{p, p} would be valid
    // here
    auto const &b = obb._box;
    for (int d = 0; d < DIM; ++d)
    {
      auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
      if (!Kokkos::isfinite(r_d) || r_d < 0)
        return false;
    }
    return true;
  }
};

template <typename OBB, typename Point>
struct expand<OBBTag, PointTag, OBB, Point>
{
  KOKKOS_FUNCTION static void apply(OBB &obb, Point const &point)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;
    using Coordinate = GeometryTraits::coordinate_type_t<OBB>;

    using HyperPoint = ExperimentalHyperGeometry::Point<DIM, Coordinate>;
    auto hyper_point = Kokkos::bit_cast<HyperPoint>(point);

    if (!Details::isValid(obb))
    {
      obb = OBB({hyper_point});
      return;
    }

    auto const corners = obb.corners();
    int const num_corners = corners.size();

    Details::StaticVector<HyperPoint, corners.capacity() + 1> points;
    for (int i = 0; i < num_corners; ++i)
      points.emplaceBack(corners[i]);
    points.emplaceBack(hyper_point);

    obb = OBB(Kokkos::View<HyperPoint *, Kokkos::AnonymousSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        points.data(), points.size()));
  }
};

template <typename OBB, typename Box>
struct expand<OBBTag, BoxTag, OBB, Box>
{
  KOKKOS_FUNCTION static void apply(OBB &obb, Box const &box)
  {
    OBB other;
    // FIXME: doing expand instead of _box = box to accomodate both regular and
    // hyper-dimensional box
    Details::expand(other._box, box);

    if (!Details::isValid(obb))
      obb = other;
    else
      Details::expand(obb, other);
  }
};

template <typename OBB, typename Triangle>
struct expand<OBBTag, TriangleTag, OBB, Triangle>
{
  KOKKOS_FUNCTION static void apply(OBB &obb, Triangle const &triangle)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;
    using Coordinate = GeometryTraits::coordinate_type_t<OBB>;
    using HyperPoint = ExperimentalHyperGeometry::Point<DIM, Coordinate>;

    if (!Details::isValid(obb))
    {
#if 0
      printf("--TRIANGLE--\n");
      printf("[%7.3f, %7.3f, %7.3f] - [%7.3f, %7.3f, %7.3f] - [%7.3f, "
             "%7.3f, %7.3f]\n",
             triangle.a[0], triangle.a[1], triangle.a[2], triangle.b[0],
             triangle.b[1], triangle.b[2], triangle.c[0], triangle.c[1],
             triangle.c[2]);
#endif

      obb = OBB({triangle.a, triangle.b, triangle.c});

#if 0
      printf("--OBB--\n");
      auto const A = obb.matrix();
      printf("| %.2f %.2f %.2f |\n", A(0, 0), A(0, 1), A(0, 2));
      printf("| %.2f %.2f %.2f |\n", A(1, 0), A(1, 1), A(1, 2));
      printf("| %.2f %.2f %.2f |\n", A(2, 0), A(2, 1), A(2, 2));
      auto const &box = obb._box;
      printf("box: [%7.3f, %7.3f, %7.3f] - [%7.3f, %7.3f, %7.3f]\n",
             box.minCorner()[0], box.minCorner()[1], box.minCorner()[2],
             box.maxCorner()[0], box.maxCorner()[1], box.maxCorner()[2]);
#endif

      return;
    }
    auto const corners = obb.corners();
    int const num_corners = corners.size();

    Details::StaticVector<HyperPoint, corners.capacity() + 3> points;
    for (int i = 0; i < num_corners; ++i)
      points.emplaceBack(corners[i]);
    points.emplaceBack(triangle.a);
    points.emplaceBack(triangle.b);
    points.emplaceBack(triangle.c);

    obb = OBB(Kokkos::View<HyperPoint *, Kokkos::AnonymousSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        points.data(), points.size()));
  }
};

template <typename Box, typename OBB>
struct expand<BoxTag, OBBTag, Box, OBB>
{
  KOKKOS_FUNCTION static void apply(Box &box, OBB const &obb)
  {
    if (!Details::isValid(obb))
      return;

    auto const corners = obb.corners();
    int const num_corners = corners.size();
    for (int i = 0; i < num_corners; ++i)
      Details::expand(box, corners[i]);
  }
};

template <typename OBB1, typename OBB2>
struct expand<OBBTag, OBBTag, OBB1, OBB2>
{
  KOKKOS_FUNCTION static void apply(OBB1 &obb, OBB2 const &other)
  {
    KOKKOS_ASSERT(Details::isValid(other));

    if (!Details::isValid(obb))
    {
      obb = other;
      return;
    }

#if 0
    {
      printf("--OBB1--\n");
      auto const A = obb.matrix();
      printf("| %.2f %.2f %.2f |\n", A(0, 0), A(0, 1), A(0, 2));
      printf("| %.2f %.2f %.2f |\n", A(1, 0), A(1, 1), A(1, 2));
      printf("| %.2f %.2f %.2f |\n", A(2, 0), A(2, 1), A(2, 2));
      auto const &box = obb._box;
      printf("box: [%7.3f, %7.3f, %7.3f] - [%7.3f, %7.3f, %7.3f]\n",
             box.minCorner()[0], box.minCorner()[1], box.minCorner()[2],
             box.maxCorner()[0], box.maxCorner()[1], box.maxCorner()[2]);
    }
    {
      printf("--OBB2--\n");
      auto const A = other.matrix();
      printf("| %.2f %.2f %.2f |\n", A(0, 0), A(0, 1), A(0, 2));
      printf("| %.2f %.2f %.2f |\n", A(1, 0), A(1, 1), A(1, 2));
      printf("| %.2f %.2f %.2f |\n", A(2, 0), A(2, 1), A(2, 2));
      auto const &box = other._box;
      printf("box: [%7.3f, %7.3f, %7.3f] - [%7.3f, %7.3f, %7.3f]\n",
             box.minCorner()[0], box.minCorner()[1], box.minCorner()[2],
             box.maxCorner()[0], box.maxCorner()[1], box.maxCorner()[2]);
    }
#endif

    auto corners1 = obb.corners();
    auto corners2 = other.corners();
    int const num_corners1 = corners1.size();
    int const num_corners2 = corners2.size();

    using Point = typename decltype(corners1)::value_type;

    Details::StaticVector<Point, corners1.capacity() + corners2.capacity()>
        points;
    for (int i = 0; i < num_corners1; ++i)
      points.emplaceBack(corners1[i]);
    for (int i = 0; i < num_corners2; ++i)
      points.emplaceBack(corners2[i]);

    obb = OBB1(Kokkos::View<Point *, Kokkos::AnonymousSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        points.data(), points.size()));
#if 0
    {
      printf("--RESULT--\n");
      auto const A = obb.matrix();
      printf("| %.2f %.2f %.2f |\n", A(0, 0), A(0, 1), A(0, 2));
      printf("| %.2f %.2f %.2f |\n", A(1, 0), A(1, 1), A(1, 2));
      printf("| %.2f %.2f %.2f |\n", A(2, 0), A(2, 1), A(2, 2));
      auto const &box = obb._box;
      printf("box: [%7.3f, %7.3f, %7.3f] - [%7.3f, %7.3f, %7.3f]\n",
             box.minCorner()[0], box.minCorner()[1], box.minCorner()[2],
             box.maxCorner()[0], box.maxCorner()[1], box.maxCorner()[2]);
    }
#endif
  }
};

template <typename OBB>
struct centroid<OBBTag, OBB>
{
  KOKKOS_FUNCTION static auto apply(OBB const &obb)
  {
    return obb.rotate_back(Details::returnCentroid(obb._box));
  }
};

template <typename Point, typename OBB>
struct intersects<PointTag, OBBTag, Point, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              OBB const &obb)
  {
    return Details::intersects(obb.rotate(point), obb._box);
  }
};

template <typename Sphere, typename OBB>
struct intersects<SphereTag, OBBTag, Sphere, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &sphere,
                                              OBB const &obb)
  {
    return Details::intersects(
        Sphere{obb.rotate(sphere.centroid()), sphere.radius()}, obb._box);
  }
};

template <typename Point, typename OBB>
struct distance<PointTag, OBBTag, Point, OBB>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, OBB const &obb)
  {
    return Details::distance(obb.rotate(point), obb._box);
  }
};

} // namespace ArborX::Details::Dispatch

#endif
