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

#include "ArborX_WallDistance.hpp"

#include <Panzer_IntegrationRule.hpp>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_WorksetContainer.hpp>
#include <Panzer_WorksetNeeds.hpp>
#include <mpi.h>

constexpr int workset_size = 64;

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

auto build_worksets(Teuchos::RCP<panzer_stk::STK_Interface> const &mesh,
                    std::string const &block_name,
                    std::string const &basis_type, int const basis_order,
                    int int_order)
{
  using Teuchos::rcp;

  panzer::CellData cell_data(workset_size, mesh->getCellTopology(block_name));

  panzer::WorksetNeeds needs;
  auto basis = rcp(new panzer::PureBasis(basis_type, basis_order, cell_data));
  auto ir = rcp(new panzer::IntegrationRule(int_order, cell_data));
  needs.bases.push_back(basis);
  needs.int_rules.push_back(ir);
  needs.cellData = cell_data;

  auto workset_factory = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh));

  panzer::WorksetContainer workset_container;
  workset_container.setFactory(workset_factory);
  workset_container.setNeeds(block_name, needs);

  panzer::WorksetDescriptor workset_descriptor(block_name);
  return workset_container.getWorksets(workset_descriptor);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  using Coordinate = double;

  std::string basis_type = "HGrad";
  constexpr int basis_order = 1;
  constexpr int int_order = 2;

  constexpr bool ReplicateSides = true;

  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;

    ExecutionSpace space;

    panzer_stk::STK_ExodusReaderFactory factory("mesh.exo");
    Teuchos::RCP<panzer_stk::STK_Interface> mesh =
        factory.buildMesh(MPI_COMM_WORLD);
    print_mesh_info(*mesh->getMetaData());

    auto worksets =
        build_worksets(mesh, "eblock-0_0", basis_type, basis_order, int_order);

    std::vector<std::string> wall_names = {"left_tri3_edge2",
                                           "right_tri3_edge2", "top_tri3_edge2",
                                           "bottom_tri3_edge2"};

    panzer::CellData cell_data(64, mesh->getCellTopology("eblock-0_0"));
    panzer::IntegrationRule ir(int_order, cell_data);

    // FIXME: there must be a better way to get max_num_cells_per_workset and
    // num_int_points_per_cell without creating an MDField just for that
    PHX::MDField<Coordinate, panzer::Cell, panzer::Point> blah_distances(
        "Example::blah", ir.dl_scalar);
    auto const num_worksets = (*worksets).size();
    auto const max_num_cells_per_workset = blah_distances.extent(0);
    int const num_int_points_per_cell = blah_distances.extent(1);

    Kokkos::View<Coordinate ***, MemorySpace> workset_distances(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "Example::workset_distances"),
        num_worksets, max_num_cells_per_workset, num_int_points_per_cell);

    if (mesh->getDimension() == 2)
    {
      constexpr int DIM = 2;
      ArborX::Experimental::WallDistance<MemorySpace, DIM, Coordinate,
                                         ReplicateSides>
          wall_distance(space, *mesh, wall_names);
      wall_distance.distance(space, *worksets, ir, workset_distances);
    }
    else
    {
      constexpr int DIM = 3;
      ArborX::Experimental::WallDistance<MemorySpace, DIM, Coordinate,
                                         ReplicateSides>
          wall_distance(space, *mesh, wall_names);
      wall_distance.distance(space, *worksets, ir, workset_distances);
    }
  }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
