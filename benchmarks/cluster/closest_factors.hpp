/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_CLOSEST_FACTORS_HPP
#define ARBORX_CLOSEST_FACTORS_HPP

#include <array>
#include <vector>

// Find closest DIM factors for a number. The factors are
// sorted in the descending order.
template <int DIM>
std::array<int, DIM> closestFactors(int n)
{
  static_assert(DIM > 0);

  std::array<int, DIM> result;
  if constexpr (DIM == 1)
  {
    result[0] = n;
    return result;
  }

  std::vector<int> factors;

  // Find all prime factors
  unsigned i = 2;
  while (n > 1)
  {
    if (n % i != 0)
    {
      ++i;
      continue;
    }

    factors.push_back(i);
    n /= i;
  }

  // Reduce the list of factors
  while (factors.size() > DIM)
  {
    // Combine two smallest factors
    factors[1] *= factors[0];
    factors.erase(factors.begin());

    // Re-sort the list
    std::sort(factors.begin(), factors.end());
  }

  int num_factors = factors.size();
  assert(num_factors <= DIM);

  result.fill(1); // for missing factors
  for (int d = 0; d < num_factors; ++d)
    result[d] = factors[num_factors - 1 - d];

  return result;
}

#endif
