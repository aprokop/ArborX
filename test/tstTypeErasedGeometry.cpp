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

#include <ArborX_HyperTriangle.hpp>
#include <ArborX_TypeErasedGeometry.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>

using Geometry = ArborX::Experimental::Geometry<64>;
using Box = ArborX::ExperimentalHyperGeometry::Box<3>;
using Point = ArborX::ExperimentalHyperGeometry::Point<3>;
using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3>;

BOOST_AUTO_TEST_SUITE(TypeErased)

BOOST_AUTO_TEST_CASE(expand_point)
{
  using ArborX::Details::expand;

  Geometry g = Point{1, 2, 3};

  Box b;
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 1);
  BOOST_TEST(b.minCorner()[1] == 2);
  BOOST_TEST(b.minCorner()[2] == 3);
  BOOST_TEST(b.maxCorner()[0] == 1);
  BOOST_TEST(b.maxCorner()[1] == 2);
  BOOST_TEST(b.maxCorner()[2] == 3);

  g = Point{4, 5, 6};
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 1);
  BOOST_TEST(b.minCorner()[1] == 2);
  BOOST_TEST(b.minCorner()[2] == 3);
  BOOST_TEST(b.maxCorner()[0] == 4);
  BOOST_TEST(b.maxCorner()[1] == 5);
  BOOST_TEST(b.maxCorner()[2] == 6);

  Point p{0, 0, 0};
  g = p;
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 0);
  BOOST_TEST(b.minCorner()[1] == 0);
  BOOST_TEST(b.minCorner()[2] == 0);
  BOOST_TEST(b.maxCorner()[0] == 4);
  BOOST_TEST(b.maxCorner()[1] == 5);
  BOOST_TEST(b.maxCorner()[2] == 6);
}

BOOST_AUTO_TEST_CASE(expand_box)
{
  using ArborX::Details::expand;

  Geometry g = Box{{0, 0, 0}, {1, 1, 1}};

  Box b;
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 0);
  BOOST_TEST(b.minCorner()[1] == 0);
  BOOST_TEST(b.minCorner()[2] == 0);
  BOOST_TEST(b.maxCorner()[0] == 1);
  BOOST_TEST(b.maxCorner()[1] == 1);
  BOOST_TEST(b.maxCorner()[2] == 1);

  g = Box{{1, 2, 3}, {4, 5, 6}};
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 0);
  BOOST_TEST(b.minCorner()[1] == 0);
  BOOST_TEST(b.minCorner()[2] == 0);
  BOOST_TEST(b.maxCorner()[0] == 4);
  BOOST_TEST(b.maxCorner()[1] == 5);
  BOOST_TEST(b.maxCorner()[2] == 6);
}

BOOST_AUTO_TEST_CASE(expand_triangle)
{
  using ArborX::Details::expand;

  Geometry g = Triangle{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};

  Box b;
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 0);
  BOOST_TEST(b.minCorner()[1] == 0);
  BOOST_TEST(b.minCorner()[2] == 0);
  BOOST_TEST(b.maxCorner()[0] == 1);
  BOOST_TEST(b.maxCorner()[1] == 1);
  BOOST_TEST(b.maxCorner()[2] == 1);

  g = Triangle{{1, 2, 3}, {2, 3, 1}, {3, 1, 2}};
  expand(b, g);

  BOOST_TEST(b.minCorner()[0] == 0);
  BOOST_TEST(b.minCorner()[1] == 0);
  BOOST_TEST(b.minCorner()[2] == 0);
  BOOST_TEST(b.maxCorner()[0] == 3);
  BOOST_TEST(b.maxCorner()[1] == 3);
  BOOST_TEST(b.maxCorner()[2] == 3);
}

BOOST_AUTO_TEST_SUITE_END()
