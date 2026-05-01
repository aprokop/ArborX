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
#include <ArborX_Version.hpp>

#include <boost/program_options.hpp>

#include <Panzer_IntegrationRule.hpp>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_WorksetContainer.hpp>
#include <Panzer_WorksetNeeds.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <mpi.h>

constexpr int workset_size = 64;

class STKMeshFactory : public panzer_stk::STK_ExodusReaderFactory
{
public:
  STKMeshFactory(std::string const &file_name)
      : panzer_stk::STK_ExodusReaderFactory(file_name)
  {}

  virtual Teuchos::RCP<panzer_stk::STK_Interface>
  buildUncommitedMesh(stk::ParallelMachine parallelMach) const override
  {
    PANZER_FUNC_TIME_MONITOR("STKMeshFactory::buildUncomittedMesh()");

    using Teuchos::rcp;

    // read in meta data
    stk::io::StkMeshIoBroker *meshData =
        new stk::io::StkMeshIoBroker(parallelMach);
    meshData->use_simple_fields();
    meshData->property_add(Ioss::Property("LOWER_CASE_VARIABLE_NAMES", false));

    // add in "FAMILY_TREE" entity for doing refinement
    std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
    entity_rank_names.push_back("FAMILY_TREE");
    meshData->set_rank_name_vector(entity_rank_names);

    // NOTE: the only difference with Panzer's STK_ExodusReaderFactory is that
    // we add the DECOMPOSITION_METHOD property
    meshData->property_add(Ioss::Property("DECOMPOSITION_METHOD", "RIB"));

    meshData->add_mesh_database(fileName_,
                                panzer_stk::fileTypeToIOSSType(fileType_),
                                stk::io::READ_MESH);

    meshData->create_input_mesh();
    auto metaData = rcp(meshData->meta_data_ptr());

    auto mesh = rcp(new panzer_stk::STK_Interface(metaData));
    mesh->initializeFromMetaData();
    mesh->instantiateBulkData(parallelMach);
    meshData->set_bulk_data(Teuchos::get_shared_ptr(mesh->getBulkData()));

    // read in other transient fields, these will be useful later when
    // trying to read other fields for use in solve
    meshData->add_all_mesh_fields_as_input_fields();

    // store mesh data pointer for later use in initializing
    // bulk data
    mesh->getMetaData()->declare_attribute_with_delete(meshData);

    // build element blocks
    registerElementBlocks(*mesh, *meshData);
    registerSidesets(*mesh);
    registerNodesets(*mesh);

    buildMetaData(parallelMach, *mesh);

    return mesh;
  }
};

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

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_rank == 0)
  {
    std::cout << "ArborX version    : " << ArborX::version() << std::endl;
    std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
              << std::endl;
    std::cout << "#MPI ranks        : " << comm_size << std::endl;
  }

  using Coordinate = double;

  namespace bpo = boost::program_options;

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiply.
  auto *help_it = std::find_if(argv, argv + argc, [](std::string const &x) {
    return x == "--help" || x == "--kokkos-help";
  });
  bool is_help_present = (help_it != argv + argc);
  if (is_help_present && comm_rank != 0)
  {
    std::swap(*help_it, *(argv + argc - 1));
    --argc;
  }

  Kokkos::ScopeGuard guard(argc, argv);

  std::string basis_type;
  std::string filename;
  int basis_order;
  int int_order;
  std::string block_name;
  std::vector<std::string> wall_names;
  bool verbose;
  bool inspect;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "help message" )
    ("basis-order", bpo::value<int>(&basis_order)->default_value(1), "basis order")
    ("basis-type", bpo::value<std::string>(&basis_type)->default_value("HGrad"), "basis type")
    ("block-name", bpo::value<std::string>(&block_name)->default_value("eblock-0_0"), "block name")
    ("filename", bpo::value<std::string>(&filename)->default_value("mesh.exo"), "mesh filename")
    ("int-order", bpo::value<int>(&int_order)->default_value(2), "integration order")
    ( "inspect", bpo::bool_switch(&inspect), "inspect mesh file")
    ( "verbose", bpo::bool_switch(&verbose), "verbose")
    ("wall-names", bpo::value<std::vector<std::string>>(&wall_names)->multitoken(), "names of walls")
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (is_help_present)
  {
    if (comm_rank == 0)
      std::cout << desc << '\n';
    MPI_Finalize();
    return 0;
  }

  auto vec2string = [](std::vector<std::string> const &names) {
    if (names.empty())
      return std::string("(none)");
    std::string result = names[0];
    for (size_t i = 1; i < names.size(); ++i)
      result += ", " + names[i];
    return result;
  };

  // Print out the runtime parameters
  if (comm_rank == 0)
  {
    printf("basis order       : %d\n", basis_order);
    printf("basis type        : %s\n", basis_type.c_str());
    printf("block name        : %s\n", block_name.c_str());
    printf("filename          : %s\n", filename.c_str());
    printf("inspect           : %s\n", (inspect ? "true" : "false"));
    printf("integration order : %d\n", int_order);
    printf("verbose           : %s\n", (verbose ? "true" : "false"));
    printf("wall names        : %s\n", vec2string(wall_names).c_str());
  }

  constexpr bool ReplicateSides = true;

  {
    ExecutionSpace space;

    STKMeshFactory factory(filename);
    Teuchos::RCP<panzer_stk::STK_Interface> mesh =
        factory.buildMesh(MPI_COMM_WORLD);

    if (inspect)
    {
      if (comm_rank == 0)
        print_mesh_info(*mesh->getMetaData());
      return 0;
    }

    auto worksets =
        build_worksets(mesh, block_name, basis_type, basis_order, int_order);

    panzer::CellData cell_data(64, mesh->getCellTopology(block_name));
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

  MPI_Finalize();

  return 0;
}
