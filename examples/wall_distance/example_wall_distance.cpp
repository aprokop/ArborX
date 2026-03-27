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

#include "mesh_reader.hpp"
#include "wall_distance.hpp"
#include <Panzer_STK_Interface.hpp>
#include <mpi.h>

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  // using MemorySpace = typename ExecutionSpace::memory_space;

  panzer_stk::STK_Interface mesh(ArborX::Details::readExodusMesh("mesh.exo"));

  constexpr int DIM = 2;
  std::vector<std::string> wall_names = {"left_tri3_edge2", "right_tri3_edge2",
                                         "top_tri3_edge2", "bottom_tri3_edge2"};

  ExecutionSpace space;
  auto index = ArborX::Details::buildIndex<DIM>(space, mesh, wall_names);

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
