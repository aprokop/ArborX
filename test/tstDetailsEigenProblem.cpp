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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DetailsEigenProblem.hpp>
#include <ArborX_DetailsVector.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE EigenProblem

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(eigen_problem_2D)
{
  constexpr int DIM = 2;
  using Coordinate = float;

  using Vector = ArborX::Details::Vector<DIM, Coordinate>;
  using Matrix = Kokkos::Array<Kokkos::Array<Coordinate, DIM>, DIM>;

  using ArborX::Details::eigenProblem;

  Coordinate eigs[DIM];
  Vector eigv[DIM];

  // Two vectors are proportional if they line on the same line, meaning the
  // angle between them is 0 or 180.
  auto eigv_compare = [](Vector const &v1, Vector const &v2) {
    return std::abs(std::abs(v1.dot(v2)) - v1.norm() * v2.norm()) <
           std::numeric_limits<Coordinate>::epsilon();
  };

  {
    Matrix A = {{{0, 0}, {0, 0}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0);
    BOOST_TEST(eigs[1] == 0);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }

  {
    Matrix A = {{{2, 0}, {0, 0}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 0);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }
  {
    Matrix A = {{{2, 0}, {0, 3}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 3);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1}));
  }
  {
    Matrix A = {{{1, -1}, {-1, 1}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0);
    BOOST_TEST(eigs[1] == 2);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, -1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, 1}));
  }
  {
    Matrix A = {{{3, 1}, {1, 3}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 4);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, -1}));
  }
}

BOOST_AUTO_TEST_CASE(eigen_problem_3D)
{
  constexpr int DIM = 3;
  using Coordinate = float;

  using Vector = ArborX::Details::Vector<DIM, Coordinate>;
  using Matrix = Kokkos::Array<Kokkos::Array<Coordinate, DIM>, DIM>;

  using ArborX::Details::eigenProblem;

  Coordinate eigs[DIM];
  Vector eigv[DIM];

  // Two vectors are proportional if they line on the same line, meaning the
  // angle between them is 0 or 180.
  auto eigv_compare = [](Vector const &v1, Vector const &v2) {
    return std::abs(std::abs(v1.dot(v2)) - v1.norm() * v2.norm()) <
           std::sqrt(std::numeric_limits<Coordinate>::epsilon());
  };

  {
    // zero matrix
    Matrix A = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0);
    BOOST_TEST(eigs[1] == 0);
    BOOST_TEST(eigs[2] == 0);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1, 0}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{0, 0, 1}));
  }
  {
    // diagonal matrix
    Matrix A = {{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 3);
    BOOST_TEST(eigs[2] == 4);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1, 0}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{0, 0, 1}));
  }
  {
    // f = e = 0
    Matrix A = {{{3, 1, 0}, {1, 3, 0}, {0, 0, 5}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 4);
    BOOST_TEST(eigs[2] == 5);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 1, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, -1, 0}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{0, 0, 1}));
  }
  {
    // f = d = 0
    Matrix A = {{{5, 0, 0}, {0, 3, 1}, {0, 1, 3}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 5);
    BOOST_TEST(eigs[1] == 2);
    BOOST_TEST(eigs[2] == 4);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0, 0}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1, 1}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{0, 1, -1}));
  }
  {
    // d = e = 0
    Matrix A = {{{3, 0, 1}, {0, 5, 0}, {1, 0, 3}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 2);
    BOOST_TEST(eigs[1] == 5);
    BOOST_TEST(eigs[2] == 4);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 0, 1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{0, 1, 0}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{1, 0, -1}));
  }
  {
    // f = 0 (banded matrix)
    Matrix A = {{{2, -2, 0}, {-2, 4, -2}, {0, -2, 2}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == 0, tt::tolerance(1e-6f));
    BOOST_TEST(eigs[1] == 6);
    BOOST_TEST(eigs[2] == 2);
    BOOST_TEST(eigv_compare(eigv[0], Vector{1, 1, 1}));
    BOOST_TEST(eigv_compare(eigv[1], Vector{1, -2, 1}));
    BOOST_TEST(eigv_compare(eigv[2], Vector{1, 0, -1}));
  }
  {
    // repeated eigenvalues (-1, -1, 8)
    Matrix A = {{{3, 2, 4}, {2, 0, 2}, {4, 2, 3}}};
    eigenProblem(A, eigs, eigv);
    BOOST_TEST(eigs[0] == -1);
    BOOST_TEST(eigs[1] == -1);
    BOOST_TEST(eigs[2] == 8);
    // BOOST_TEST(eigv_compare(eigv[0], Vector{1, -2, 0}));
    // BOOST_TEST(eigv_compare(eigv[1], Vector{4, 2, -5}));
    // BOOST_TEST(eigv_compare(eigv[2], Vector{2, 1, 2}));
  }
}
