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

#ifndef ARBORX_HPP
#define ARBORX_HPP

#include <ArborX_Config.hpp>

// Geometries
#include <ArborX_Box.hpp>
#include <ArborX_KDOP.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Segment.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>

// Indexes
#include <ArborX_BruteForce.hpp>
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_AttachIndices.hpp>
#include <detail/ArborX_NeighborList.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>

// Other files
#include <misc/ArborX_Exception.hpp>

#endif
