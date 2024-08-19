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
#include <ArborX_DetailsQuaternion.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(quaternion_constructor)
{
  using quaternion = ArborX::Details::quaternion<float>;

  quaternion x(2);
  BOOST_TEST(x.real() == 2);
  BOOST_TEST(x.imag_i() == 0);
  BOOST_TEST(x.imag_j() == 0);
  BOOST_TEST(x.imag_k() == 0);

  quaternion y(x);
  BOOST_TEST(y.real() == 2);
  BOOST_TEST(y.imag_i() == 0);
  BOOST_TEST(y.imag_j() == 0);
  BOOST_TEST(y.imag_k() == 0);
  BOOST_TEST((y == x));

  quaternion z(1, 2, 3, 4);
  BOOST_TEST(z.real() == 1);
  BOOST_TEST(z.imag_i() == 2);
  BOOST_TEST(z.imag_j() == 3);
  BOOST_TEST(z.imag_k() == 4);

  quaternion q = 3;
  BOOST_TEST(q.real() == 3);
  BOOST_TEST(q.imag_i() == 0);
  BOOST_TEST(q.imag_j() == 0);
  BOOST_TEST(q.imag_k() == 0);
}

BOOST_AUTO_TEST_CASE(quaternion_addition)
{
  using quaternion = ArborX::Details::quaternion<float>;

  quaternion x(1, 2, 3, 4);
  quaternion y(10, 20, 30, 40);
  BOOST_TEST((x + y == quaternion(11, 22, 33, 44)));
}

BOOST_AUTO_TEST_CASE(quaternion_multiplication)
{
  using quaternion = ArborX::Details::quaternion<float>;

  quaternion r(1, 0, 0, 0);
  quaternion i(0, 1, 0, 0);
  quaternion j(0, 0, 1, 0);
  quaternion k(0, 0, 0, 1);

  BOOST_TEST((r * r == r));
  BOOST_TEST((r * i == i));
  BOOST_TEST((r * j == j));
  BOOST_TEST((r * k == k));

  BOOST_TEST((i * r == i));
  BOOST_TEST((i * i == -r));
  BOOST_TEST((i * j == k));
  BOOST_TEST((i * k == -j));

  BOOST_TEST((j * r == j));
  BOOST_TEST((j * i == -k));
  BOOST_TEST((j * j == -r));
  BOOST_TEST((j * k == i));

  BOOST_TEST((k * r == k));
  BOOST_TEST((k * i == j));
  BOOST_TEST((k * j == -i));
  BOOST_TEST((k * k == -r));
}

BOOST_AUTO_TEST_CASE(quaternion_abs)
{
  using quaternion = ArborX::Details::quaternion<float>;

  // 25 = 16 + 4 + 4 + 1
  quaternion q(4, 2, 2, 1);
  BOOST_TEST(abs(q) == 5);
}

BOOST_AUTO_TEST_CASE(quaternion_conj)
{
  using quaternion = ArborX::Details::quaternion<float>;

  quaternion q(1, 2, 3, 4);
  BOOST_TEST((conj(q) == quaternion(1, -2, -3, -4)));
}

BOOST_AUTO_TEST_CASE(quaternion_inverse)
{
  using quaternion = ArborX::Details::quaternion<float>;

  quaternion q(1.6f, 0.8f, 0.8f, 0.4f);
  BOOST_TEST((inv(q) == quaternion(0.4f, -0.2f, -0.2f, -0.1f)));
}
