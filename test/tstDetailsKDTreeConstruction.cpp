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
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKDTreeConstruction.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // clz

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsKDTreeConstruction

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(number_of_leading_zero_bits)
{
  using KokkosExt::clz;
  BOOST_TEST(clz(0) == 32);
  BOOST_TEST(clz(1) == 31);
  BOOST_TEST(clz(2) == 30);
  BOOST_TEST(clz(3) == 30);
  BOOST_TEST(clz(4) == 29);
  BOOST_TEST(clz(5) == 29);
  BOOST_TEST(clz(6) == 29);
  BOOST_TEST(clz(7) == 29);
  BOOST_TEST(clz(8) == 28);
  BOOST_TEST(clz(9) == 28);
  // bitwise exclusive OR operator to compare bits
  BOOST_TEST(clz(1 ^ 0) == 31);
  BOOST_TEST(clz(2 ^ 0) == 30);
  BOOST_TEST(clz(2 ^ 1) == 30);
  BOOST_TEST(clz(3 ^ 0) == 30);
  BOOST_TEST(clz(3 ^ 1) == 30);
  BOOST_TEST(clz(3 ^ 2) == 31);
  BOOST_TEST(clz(4 ^ 0) == 29);
  BOOST_TEST(clz(4 ^ 1) == 29);
  BOOST_TEST(clz(4 ^ 2) == 29);
  BOOST_TEST(clz(4 ^ 3) == 29);
}

BOOST_AUTO_TEST_CASE(compact_bits)
{
  using ArborX::Details::compactBits;
  using ArborX::Details::expandBits;
  BOOST_TEST(compactBits(expandBits(0)) == 0);
  BOOST_TEST(compactBits(expandBits(15)) == 15);
  BOOST_TEST(compactBits(expandBits(33)) == 33);
  BOOST_TEST(compactBits(expandBits(731)) == 731);
  BOOST_TEST(compactBits(expandBits(916)) == 916);
}

BOOST_AUTO_TEST_CASE(compute_reference_plane)
{
  using ArborX::Details::AAPlane;
  using ArborX::Details::isValid;
  using ArborX::Details::KDTreeConstruction::computeReferencePlane;

#define LAZY(i, j, ref_axis, ref_location)                                     \
  computeReferencePlane((i), (j), p);                                          \
  BOOST_TEST(p.axis() == (ref_axis));                                          \
  BOOST_TEST(p.location() == (ref_location));

  AAPlane p;
  computeReferencePlane(0, 0, p);
  BOOST_ASSERT(!isValid(p));
  LAZY(0, 1, 2, 1. / 1024);
  LAZY(0, 2, 1, 1. / 1024);
  LAZY(0, 3, 1, 1. / 1024);
  LAZY(0, 4, 0, 1. / 1024);
  LAZY(0, 5, 0, 1. / 1024);
  LAZY(0, 6, 0, 1. / 1024);
  LAZY(0, 7, 0, 1. / 1024);
  LAZY(0, 8, 2, 2. / 1024);
  LAZY(0, 9, 2, 2. / 1024);
  LAZY(0, 10, 2, 2. / 1024);
  LAZY(0, 15, 2, 2. / 1024);
  LAZY(0, 16, 1, 2. / 1024);
  LAZY(0, 17, 1, 2. / 1024);
  LAZY(0, 32, 0, 2. / 1024);
  // 233698377 = 00001101111011011111010001001001
  // 256522488 = 00001111010010100011100011111000
  //             --xyzxy
  // x coord        0  1  0  0  0  0  0  0  0  0
  LAZY(233698377, 256522488, 1, 256. / 1024);
  // 32242   = 00000000000000000111110111110010
  // 2340203 = 00000000001000111011010101101011
  //           --xyzxyzxyz
  // x coord       0  0  1  0  0  0  0  0  0  0
  LAZY(32242, 2340203, 2, 128. / 1024);
  // 32242   = 00000000111000000111110111110010
  // 2340203 = 00000000111000111011010101101011
  //           --xyzxyzxyzxyzx
  // x coord     0  0  1  0  1  0  0  0  0  0
  LAZY(14712306, 14923115, 0, 160. / 1024);
}
