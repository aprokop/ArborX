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

#ifndef ARBORX_EXAMPLE_MESH_INTERPOLATION_HELPERS_HPP
#define ARBORX_EXAMPLE_MESH_INTERPOLATION_HELPERS_HPP

#include <string>
#include <vector>

#include <Panzer_STK_Interface.hpp>
#include <Teuchos_RCP.hpp>
#include <mpi.h>

Teuchos::RCP<panzer_stk::STK_Interface>
build_mesh(MPI_Comm comm, std::string const &filename,
           std::vector<std::string> &block_names, int restart_index);

bool check_mesh_file(MPI_Comm comm, std::string const &filename);

#endif
