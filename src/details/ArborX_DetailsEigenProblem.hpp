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

#ifndef ARBORX_DETAILS_EIGENPROBLEM_HPP
#define ARBORX_DETAILS_EIGENPROBLEM_HPP

#include <ArborX_DetailsVector.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

// Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. Closed-form
// expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian
// matrices. [Research Report] Université de Lyon. 2017
// https://hal.science/hal-01501221/
template <typename Coordinate>
KOKKOS_FUNCTION void
eigenProblem(Kokkos::Array<Kokkos::Array<Coordinate, 2>, 2> const &matrix,
             Coordinate eigenvalues[2],
             Details::Vector<2, Coordinate> (&eigenvectors)[2])
{
  // Matrix:
  // | a c |
  // | c b |
  auto const a = matrix[0][0];
  auto const c = matrix[0][1];
  auto const b = matrix[1][1];

  if (c == 0)
  {
    eigenvalues[0] = a;
    eigenvalues[1] = b;
    eigenvectors[0] = {1, 0};
    eigenvectors[1] = {0, 1};
    return;
  }

  auto delta = Kokkos::sqrt(4 * c * c + (a - b) * (a - b));
  eigenvalues[0] = (a + b - delta) / 2;
  eigenvalues[1] = (a + b + delta) / 2;
  eigenvectors[0] = {eigenvalues[1] - b, c};
  eigenvectors[1] = {eigenvalues[0] - b, c};
}

// Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. Closed-form
// expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian
// matrices. [Research Report] Université de Lyon. 2017
// https://hal.science/hal-01501221/
// with some helpful contributions from
// https://gist.github.com/jipolanco/cd0be9867511148345144ddcd517d9ba
template <typename Coordinate>
KOKKOS_FUNCTION void
eigenProblem(Kokkos::Array<Kokkos::Array<Coordinate, 3>, 3> const &matrix,
             Coordinate eigenvalues[3],
             Details::Vector<3, Coordinate> (&eigenvectors)[3])
{

  // Matrix:
  // | a d f |
  // | d b e |
  // | f e c |
  auto const a = matrix[0][0];
  auto const d = matrix[0][1];
  auto const f = matrix[0][2];
  auto const b = matrix[1][1];
  auto const e = matrix[1][2];
  auto const c = matrix[2][2];

  // printf("| %5.2f %5.2f %5.2f |\n", a, d, f);
  // printf("| %5.2f %5.2f %5.2f |\n", d, b, e);
  // printf("| %5.2f %5.2f %5.2f |\n", f, e, c);

  // FIXME: I'm not sure how robust this method is to the repeated eigenvalues

  // Treat corner cases
  if ((f == 0 && e == 0) || (f == 0 && d == 0) || (d == 0 && e == 0))
  {
    Coordinate eigs[2];
    Details::Vector<2, Coordinate> v[2];
    if (f == 0 && e == 0)
    {
      eigenProblem({{{a, d}, {d, b}}}, eigs, v);

      eigenvalues[0] = eigs[0];
      eigenvalues[1] = eigs[1];
      eigenvalues[2] = c;
      eigenvectors[0] = {v[0][0], v[0][1], 0};
      eigenvectors[1] = {v[1][0], v[1][1], 0};
      eigenvectors[2] = {0, 0, 1};
    }
    else if (f == 0 && d == 0)
    {
      eigenProblem({{{b, e}, {e, c}}}, eigs, v);

      eigenvalues[0] = a;
      eigenvalues[1] = eigs[0];
      eigenvalues[2] = eigs[1];
      eigenvectors[0] = {1, 0, 0};
      eigenvectors[1] = {0, v[0][0], v[0][1]};
      eigenvectors[2] = {0, v[1][0], v[1][1]};
    }
    else
    {
      eigenProblem({{{a, f}, {f, c}}}, eigs, v);

      eigenvalues[0] = eigs[0];
      eigenvalues[1] = b;
      eigenvalues[2] = eigs[1];
      eigenvectors[0] = {v[0][0], 0, v[0][1]};
      eigenvectors[1] = {0, 1, 0};
      eigenvectors[2] = {v[1][0], 0, v[1][1]};
    }
    return;
  }

  auto const d2 = d * d;
  auto const e2 = e * e;
  auto const f2 = f * f;

  // Rewrite x1 from the paper equation to ensure its positiveness.
  //   a^2 + b^2 + c^2 - ab - ac - bc =
  //     [(a - b)^2 + (b - c)^2 + (a - c)^2] / 2
  auto x1 = ((a - b) * (a - b) + (b - c) * (b - c) + (a - c) * (a - c)) / 2 +
            3 * (d2 + f2 + e2);
  Coordinate x2;
  {
    auto A = 2 * a - b - c;
    auto B = 2 * b - a - c;
    auto C = 2 * c - a - b;
    x2 = -A * B * C + 9 * (C * d2 + B * f2 + A * e2) + 54 * d * e * f;
  }

  // Use slightly more numerically stable approach. Instead of
  //   4 * x1^3 - x2^2
  // do
  //   (y1 - x2)*(y1 + x2)
  auto y1 = 2 * Kokkos::sqrt(x1 * x1 * x1);
  auto delta = (y1 - x2) * (y1 + x2);
  // FIXME: if delta < 0, it must be really small, so check for that and
  // make it 0?
  KOKKOS_ASSERT(delta >= 0);
  auto const phi = Kokkos::atan2(Kokkos::sqrt(delta), x2);

  {
    auto const pi = Kokkos::numbers::pi_v<Coordinate>;
    auto const A = (a + b + c) / 3;
    auto const B = 2 * Kokkos::sqrt(x1) / 3;
    eigenvalues[0] = A - B * Kokkos::cos(phi / 3);
    eigenvalues[1] = A + B * Kokkos::cos((phi - pi) / 3);
    eigenvalues[2] = A + B * Kokkos::cos((phi + pi) / 3);
  }

  for (int k = 0; k < 3; ++k)
  {
    if (f != 0)
    {
      // Rewrite the paper equations to avoid divisions
      auto const alpha = d * (c - eigenvalues[k]) - e * f;
      auto const beta = f * (b - eigenvalues[k]) - d * e;
      eigenvectors[k] = {beta * (eigenvalues[k] - c) - alpha * e, alpha * f,
                         beta * f};
    }
    else
    {
      eigenvectors[k] = {e * d, e * (eigenvalues[k] - a),
                         (eigenvalues[k] - a) * (eigenvalues[k] - b) - d * d};
    }

    using Vector = Details::Vector<3, Coordinate>;
    KOKKOS_ASSERT(!(eigenvectors[k] == Vector{0, 0, 0}));
  }
}

} // namespace ArborX::Details

#endif
