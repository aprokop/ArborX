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

#include "helpers.hpp"

#include <Ionit_Initializer.h>
#include <Ioss_DatabaseIO.h>
#include <Ioss_ElementBlock.h>
#include <Ioss_IOFactory.h>
#include <Ioss_Region.h>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <mpi.h>
#include <sstream>
#include <iterator>

class STKMeshFactory : public panzer_stk::STK_ExodusReaderFactory
{
public:
  STKMeshFactory(std::string const &file_name, int const restart_index)
      : panzer_stk::STK_ExodusReaderFactory(file_name, restart_index)
  {}

  void completeMeshConstruction(panzer_stk::STK_Interface &mesh,
                                 stk::ParallelMachine parallelMach) const override
  {
    if (!mesh.isInitialized())
      mesh.initialize(parallelMach, true, false);

    stk::mesh::MetaData &metaData = *mesh.getMetaData();
    stk::io::StkMeshIoBroker *meshData = const_cast<stk::io::StkMeshIoBroker *>(
        metaData.get_attribute<stk::io::StkMeshIoBroker>());
    TEUCHOS_ASSERT(metaData.remove_attribute(meshData));

    meshData->populate_bulk_data();

    int restartIndex = restartIndex_;
    if (restartIndex < 0)
      restartIndex = 1 + restartIndex +
                     meshData->get_input_ioss_region()->get_max_time().first;

    meshData->read_defined_input_fields(restartIndex);

    mesh.setInitialStateTime(
        restartIndex > 0
            ? meshData->get_input_ioss_region()->get_state_time(restartIndex)
            : 0.0);

    delete meshData;
  }
};

template <typename T>
std::string vec2string(std::vector<T> const &s, std::string const &delim = ", ")
{
  if (s.empty())
    return "[]";
  if (s.size() == 1)
    return "[" + std::to_string(s[0]) + "]";

  std::ostringstream ss;
  std::copy(s.begin(), s.end(),
            std::ostream_iterator<std::string>{ss, delim.c_str()});
  auto delimited_items = ss.str().erase(ss.str().length() - delim.size());
  return "[" + delimited_items + "]";
}

bool check_mesh_file(MPI_Comm comm, std::string const &filename)
{
  Ioss::Init::Initializer ioss_initializer;
  Ioss::DatabaseIO *ioss_db =
      Ioss::IOFactory::create("exodus", filename, Ioss::READ_MODEL, comm);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  if (ioss_db == nullptr || !ioss_db->ok(true))
  {
    if (comm_rank == 0)
      std::cerr << "ERROR: Could not open file " << filename << "\n";
    return false;
  }

  Ioss::Region region(ioss_db, "mesh_region");

  auto const &element_blocks = region.get_element_blocks();
  if (element_blocks.empty() && comm_rank == 0)
  {
    std::cerr << "ERROR: No element blocks found in mesh file " << filename
              << "\n";
    return false;
  }

  return true;
}

Teuchos::RCP<panzer_stk::STK_Interface>
build_mesh(MPI_Comm comm, std::string const &filename,
           std::vector<std::string> &block_names, int restart_index)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  STKMeshFactory factory(filename, restart_index);
  auto mesh = factory.buildUncommitedMesh(comm);

  if (block_names.empty())
    mesh->getElementBlockNames(block_names);

  factory.completeMeshConstruction(*mesh, comm);

  return mesh;
}
