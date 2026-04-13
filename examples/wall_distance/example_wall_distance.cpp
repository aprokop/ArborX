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

#include "wall_distance.hpp"
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <mpi.h>

void print_mesh_info(stk::mesh::MetaData const &meta_data)
{
  stk::mesh::BulkData const &bulk_data = meta_data.mesh_bulk_data();
  for (stk::mesh::Part const *part : meta_data.get_parts())
    if (part->id() >= 0)
    {
      auto topology = part->topology();
      auto rank = part->primary_entity_rank();
      std::cout << "Part: name=" << part->name()
                << (rank == meta_data.side_rank() ? " (side)" : "")
                << ", id=" << part->id() << ", rank=" << rank
                << ", topology=" << topology.name()
                << ", n=" << stk::mesh::count_entities(bulk_data, rank, *part)
                << std::endl;
    }

  stk::mesh::FieldVector const &fields = meta_data.get_fields();
  for (stk::mesh::FieldBase *field : fields)
    std::cout << "Field: name=" << field->name()
              << ", rank=" << field->entity_rank() << std::endl;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;

    panzer_stk::STK_ExodusReaderFactory factory("mesh.exo");
    Teuchos::RCP<panzer_stk::STK_Interface> mesh =
        factory.buildMesh(MPI_COMM_WORLD);
    print_mesh_info(*mesh->getMetaData());

    constexpr int DIM = 2;
    std::vector<std::string> wall_names = {"left_tri3_edge2",
                                           "right_tri3_edge2", "top_tri3_edge2",
                                           "bottom_tri3_edge2"};

    ExecutionSpace space;
    auto index = ArborX::Details::buildIndex<DIM>(space, *mesh, wall_names);
  }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
