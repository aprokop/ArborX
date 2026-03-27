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

#include <string>

#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <mpi.h>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>

namespace ArborX::Details
{

void print_mesh_info(stk::mesh::MetaData const &meta_data)
{
  stk::mesh::BulkData const &bulk_data = meta_data.mesh_bulk_data();
  for (stk::mesh::Part const *part : meta_data.get_parts())
    // if (part->id() >= 0 && part->topology() !=
    // stk::topology::INVALID_TOPOLOGY)
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

auto readExodusMesh(std::string const &filename)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  KOKKOS_ASSERT(stk::parallel_machine_size(comm) == 1);

  stk::io::StkMeshIoBroker mesh_reader;

  auto bulk_ptr = stk::mesh::MeshBuilder(comm).create();
  // Inform STK IO which STK Mesh objects to populate later
  mesh_reader.set_bulk_data(*bulk_ptr);

  // Connect the broker to the file
  mesh_reader.add_mesh_database(filename, "exodusII", stk::io::READ_MESH);

  // Read the file header and populate the MetaData (Parts and Fields)
  mesh_reader.create_input_mesh();

  // Populate entities in STK Mesh from Exodus file
  mesh_reader.populate_bulk_data();

  stk::mesh::MetaData const &meta_data = mesh_reader.meta_data();

  print_mesh_info(meta_data);

  unsigned num_elements = stk::mesh::count_entities(mesh_reader.bulk_data(),
                                                    stk::topology::ELEM_RANK,
                                                    meta_data.universal_part());
  printf("Read mesh with %u elements\n", num_elements);

  return Teuchos::rcp(mesh_reader.meta_data_ptr());
}

} // namespace ArborX::Details
