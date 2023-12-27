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

#include <ArborX_Config.hpp> // IWYU pragma: export

#include <ArborX_AccessTraits.hpp>       // IWYU pragma: export
#include <ArborX_Box.hpp>                // IWYU pragma: export
#include <ArborX_BruteForce.hpp>         // IWYU pragma: export
#include <ArborX_Callbacks.hpp>          // IWYU pragma: export
#include <ArborX_CrsGraphWrapper.hpp>    // IWYU pragma: export
#include <ArborX_Exception.hpp>          // IWYU pragma: export
#include <ArborX_GeometryTraits.hpp>     // IWYU pragma: export
#include <ArborX_HyperBox.hpp>           // IWYU pragma: export
#include <ArborX_HyperPoint.hpp>         // IWYU pragma: export
#include <ArborX_HyperSphere.hpp>        // IWYU pragma: export
#include <ArborX_IndexableGetter.hpp>    // IWYU pragma: export
#include <ArborX_KDOP.hpp>               // IWYU pragma: export
#include <ArborX_LinearBVH.hpp>          // IWYU pragma: export
#include <ArborX_PairValueIndex.hpp>     // IWYU pragma: export
#include <ArborX_Point.hpp>              // IWYU pragma: export
#include <ArborX_Predicates.hpp>         // IWYU pragma: export
#include <ArborX_SpaceFillingCurves.hpp> // IWYU pragma: export
#include <ArborX_Sphere.hpp>             // IWYU pragma: export
#include <ArborX_TraversalPolicy.hpp>    // IWYU pragma: export

#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp> // IWYU pragma: export
#include <ArborX_PairIndexRank.hpp>   // IWYU pragma: export
#endif

#endif
