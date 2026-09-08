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

#include "ArborXBenchmark_TimeMonitor.hpp"
#include <ArborX.hpp>
#include <ArborXBenchmark_MPIKokkosScopeGuard.hpp>
#include <ArborX_BarycentricCoordinates.hpp>
#include <ArborX_Triangle.hpp>
#include <ArborX_Version.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <vector>

#include "helpers.hpp"
#include <Panzer_STK_Interface.hpp>
#include <mpi.h>

template <int DIM, typename Coordinate, typename ExecutionSpace>
struct ElementAccessTraits;

template <typename Coordinate, typename ExecutionSpace>
struct ElementAccessTraits<2, Coordinate, ExecutionSpace>
{
  using Triangle = ArborX::Triangle<2, Coordinate>;
  using Triangles = Kokkos::View<Triangle *, ExecutionSpace::memory_space>;

  Triangles triangles;

  KOKKOS_FUNCTION int size(ElementAccessTraits const &) const
  {
    return triangles.extent(0);
  }

  KOKKOS_FUNCTION Triangle get(ElementAccessTraits const &, int i) const
  {
    return triangles(i);
  }
};

template <typename Coordinate, typename ExecutionSpace>
struct ElementAccessTraits<3, Coordinate, ExecutionSpace>
{
  using Tetrahedron =
      ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>;
  using Tetrahedra =
      Kokkos::View<Tetrahedron *, ExecutionSpace::memory_space>;

  Tetrahedra tetrahedra;

  KOKKOS_FUNCTION int size(ElementAccessTraits const &) const
  {
    return tetrahedra.extent(0);
  }

  KOKKOS_FUNCTION Tetrahedron get(ElementAccessTraits const &, int i) const
  {
    return tetrahedra(i);
  }
};

namespace ArborX
{
template <typename Coordinate, typename ExecutionSpace>
struct AccessTraits<ElementAccessTraits<2, Coordinate, ExecutionSpace>>
{
  using Tag = ArborX::Details::PrimitiveTag;
  using self_type = ElementAccessTraits<2, Coordinate, ExecutionSpace>;
  using memory_space = typename ExecutionSpace::memory_space;

  KOKKOS_FUNCTION static int size(self_type const &self)
  {
    return self.size(self);
  }

  KOKKOS_FUNCTION static auto get(self_type const &self, int i)
  {
    return self.get(self, i);
  }
};

template <typename Coordinate, typename ExecutionSpace>
struct AccessTraits<ElementAccessTraits<3, Coordinate, ExecutionSpace>>
{
  using Tag = ArborX::Details::PrimitiveTag;
  using self_type = ElementAccessTraits<3, Coordinate, ExecutionSpace>;
  using memory_space = typename ExecutionSpace::memory_space;

  KOKKOS_FUNCTION static int size(self_type const &self)
  {
    return self.size(self);
  }

  KOKKOS_FUNCTION static auto get(self_type const &self, int i)
  {
    return self.get(self, i);
  }
};
} // namespace ArborX

template <int DIM, typename Coordinate, typename ExecutionSpace,
          typename MemorySpace>
void interpolate_field(
    ExecutionSpace const &space,
    Teuchos::RCP<panzer_stk::STK_Interface> const &source_mesh,
    Teuchos::RCP<panzer_stk::STK_Interface> const &target_mesh,
    std::vector<std::string> const &source_block_names,
    std::vector<std::string> const &target_block_names,
    std::string const &source_field_name, std::string const &target_field_name,
    bool verbose)
{
  using Point = ArborX::Point<DIM, Coordinate>;

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  ArborXBenchmark::TimeMonitor time_monitor;

  // Extract source mesh nodes and elements
  auto timer = time_monitor.getNewTimer("extract_source_mesh");
  timer->start();

  auto source_meta = source_mesh->getMetaData();
  auto source_bulk = source_mesh->getBulkData();

  stk::mesh::Selector source_selector;
  for (auto const &block_name : source_block_names)
    source_selector |= *source_meta->get_part(block_name);
  source_selector &= source_meta->locally_owned_part() |
                     source_meta->globally_shared_part();

  // Get nodes
  stk::mesh::EntityVector source_nodes;
  stk::mesh::get_selected_entities(
      source_selector, source_bulk->buckets(stk::topology::NODE_RANK),
      source_nodes);
  int num_source_nodes = source_nodes.size();

  if (comm_rank == 0)
    std::cout << "Source mesh: " << num_source_nodes << " nodes\n";

  // Extract node coordinates
  Kokkos::View<Point *, MemorySpace> source_coords(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "source_coordinates"),
      num_source_nodes);

  auto source_coords_host = Kokkos::create_mirror_view(source_coords);
  for (int i = 0; i < num_source_nodes; ++i)
  {
    auto const *coords = source_bulk->begin_nodes(source_nodes[i]);
    for (int d = 0; d < DIM; ++d)
      source_coords_host(i)[d] = coords[d];
  }
  Kokkos::deep_copy(source_coords, source_coords_host);

  // Get elements
  std::vector<stk::mesh::Entity> source_elements;
  for (auto const &block_name : source_block_names)
    source_mesh->getMyElements(block_name, source_elements);
  int num_source_elements = source_elements.size();

  if (comm_rank == 0)
    std::cout << "Source mesh: " << num_source_elements << " elements\n";

  // Extract element connectivity
  Kokkos::View<int *, MemorySpace> element_nodes_flat;
  Kokkos::View<int *, MemorySpace> element_offsets;

  if constexpr (DIM == 2)
  {
    // For 2D, we have triangles (3 nodes per element)
    Kokkos::View<int *, MemorySpace> element_nodes_temp(
        "element_nodes", num_source_elements * 3);
    auto element_nodes_host = Kokkos::create_mirror_view(element_nodes_temp);

    for (int el = 0; el < num_source_elements; ++el)
    {
      auto const *nodes = source_bulk->begin_nodes(source_elements[el]);
      int num_nodes = source_bulk->num_nodes(source_elements[el]);
      ARBORX_ASSERT(num_nodes == 3);
      for (int i = 0; i < num_nodes; ++i)
        element_nodes_host(el * 3 + i) =
            source_bulk->identifier(nodes[i]) - 1; // Convert to 0-based
    }

    Kokkos::deep_copy(element_nodes_temp, element_nodes_host);
    element_nodes_flat = element_nodes_temp;

    element_offsets = Kokkos::View<int *, MemorySpace>("element_offsets",
                                                        num_source_elements + 1);
    auto element_offsets_host = Kokkos::create_mirror_view(element_offsets);
    for (int i = 0; i <= num_source_elements; ++i)
      element_offsets_host(i) = i * 3;
    Kokkos::deep_copy(element_offsets, element_offsets_host);
  }
  else
  {
    // For 3D, we have tetrahedra (4 nodes per element)
    Kokkos::View<int *, MemorySpace> element_nodes_temp(
        "element_nodes", num_source_elements * 4);
    auto element_nodes_host = Kokkos::create_mirror_view(element_nodes_temp);

    for (int el = 0; el < num_source_elements; ++el)
    {
      auto const *nodes = source_bulk->begin_nodes(source_elements[el]);
      int num_nodes = source_bulk->num_nodes(source_elements[el]);
      ARBORX_ASSERT(num_nodes == 4);
      for (int i = 0; i < num_nodes; ++i)
        element_nodes_host(el * 4 + i) =
            source_bulk->identifier(nodes[i]) - 1; // Convert to 0-based
    }

    Kokkos::deep_copy(element_nodes_temp, element_nodes_host);
    element_nodes_flat = element_nodes_temp;

    element_offsets = Kokkos::View<int *, MemorySpace>("element_offsets",
                                                        num_source_elements + 1);
    auto element_offsets_host = Kokkos::create_mirror_view(element_offsets);
    for (int i = 0; i <= num_source_elements; ++i)
      element_offsets_host(i) = i * 4;
    Kokkos::deep_copy(element_offsets, element_offsets_host);
  }

  timer->stop();

  // Extract target mesh nodes
  timer = time_monitor.getNewTimer("extract_target_mesh");
  timer->start();

  auto target_meta = target_mesh->getMetaData();
  auto target_bulk = target_mesh->getBulkData();

  stk::mesh::Selector target_selector;
  for (auto const &block_name : target_block_names)
    target_selector |= *target_meta->get_part(block_name);
  target_selector &= target_meta->locally_owned_part() |
                     target_meta->globally_shared_part();

  stk::mesh::EntityVector target_nodes;
  stk::mesh::get_selected_entities(
      target_selector, target_bulk->buckets(stk::topology::NODE_RANK),
      target_nodes);
  int num_target_nodes = target_nodes.size();

  if (comm_rank == 0)
    std::cout << "Target mesh: " << num_target_nodes << " nodes\n";

  // Extract target node coordinates
  Kokkos::View<Point *, MemorySpace> target_coords(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "target_coordinates"),
      num_target_nodes);

  auto target_coords_host = Kokkos::create_mirror_view(target_coords);
  for (int i = 0; i < num_target_nodes; ++i)
  {
    auto const *coords = target_bulk->begin_nodes(target_nodes[i]);
    for (int d = 0; d < DIM; ++d)
      target_coords_host(i)[d] = coords[d];
  }
  Kokkos::deep_copy(target_coords, target_coords_host);

  // Read source field
  timer = time_monitor.getNewTimer("read_source_field");
  timer->start();

  auto *source_field =
      source_mesh->getSolutionField(source_field_name, source_block_names[0]);

  Kokkos::View<Coordinate *, MemorySpace> source_field_vals(
      "source_field_values", num_source_nodes);
  auto source_field_host = Kokkos::create_mirror_view(source_field_vals);

  for (int i = 0; i < num_source_nodes; ++i)
  {
    auto const *field_data = stk::mesh::field_data(*source_field, source_nodes[i]);
    KOKKOS_ASSERT(field_data != nullptr);
    source_field_host(i) = *field_data;
  }

  Kokkos::deep_copy(source_field_vals, source_field_host);

  timer->stop();

  // Create BVH tree
  timer = time_monitor.getNewTimer("build_bvh");
  timer->start();

  ElementAccessTraits<DIM, Coordinate, ExecutionSpace> access_traits;

  if constexpr (DIM == 2)
  {
    // Create triangles from source mesh
    Kokkos::View<ArborX::Triangle<2, Coordinate> *, MemorySpace> triangles(
        "triangles", num_source_elements);
    auto triangles_host = Kokkos::create_mirror_view(triangles);

    for (int el = 0; el < num_source_elements; ++el)
    {
      auto n0 = element_nodes_flat((el + 0) * 3);
      auto n1 = element_nodes_flat((el + 1) * 3);
      auto n2 = element_nodes_flat((el + 2) * 3);
      triangles_host(el) = {source_coords_host(n0), source_coords_host(n1),
                             source_coords_host(n2)};
    }

    Kokkos::deep_copy(triangles, triangles_host);
    access_traits.triangles = triangles;
  }
  else
  {
    // Create tetrahedra from source mesh
    using Tetrahedron = ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>;
    Kokkos::View<Tetrahedron *, MemorySpace> tetrahedra(
        "tetrahedra", num_source_elements);
    auto tetrahedra_host = Kokkos::create_mirror_view(tetrahedra);

    for (int el = 0; el < num_source_elements; ++el)
    {
      auto n0 = element_nodes_flat((el + 0) * 4);
      auto n1 = element_nodes_flat((el + 1) * 4);
      auto n2 = element_nodes_flat((el + 2) * 4);
      auto n3 = element_nodes_flat((el + 3) * 4);
      tetrahedra_host(el) = {source_coords_host(n0), source_coords_host(n1),
                              source_coords_host(n2), source_coords_host(n3)};
    }

    Kokkos::deep_copy(tetrahedra, tetrahedra_host);
    access_traits.tetrahedra = tetrahedra;
  }

  ArborX::BoundingVolumeHierarchy<MemorySpace> tree(space, access_traits);

  timer->stop();

  // Query for containment
  timer = time_monitor.getNewTimer("query");
  timer->start();

  Kokkos::View<decltype(ArborX::intersects(Point{})) *, MemorySpace> queries(
      "queries", num_target_nodes);
  auto queries_host = Kokkos::create_mirror_view(queries);

  for (int i = 0; i < num_target_nodes; ++i)
    queries_host(i) = ArborX::intersects(target_coords_host(i));

  Kokkos::deep_copy(queries, queries_host);

  Kokkos::View<typename decltype(tree)::value_type *, MemorySpace> values;
  Kokkos::View<int *, MemorySpace> offsets;

  tree.query(space, queries, values, offsets);

  auto offsets_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           offsets);
  auto values_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         values);

  timer->stop();

  // Interpolate field values
  timer = time_monitor.getNewTimer("interpolate");
  timer->start();

  Kokkos::View<Coordinate *, MemorySpace> target_field_vals(
      "target_field_values", num_target_nodes);
  auto target_field_host = Kokkos::create_mirror_view(target_field_vals);

  for (int i = 0; i < num_target_nodes; ++i)
  {
    int start = offsets_host(i);
    int end = offsets_host(i + 1);

    if (start == end)
    {
      if (comm_rank == 0)
        std::cout << "Warning: Node " << i << " is not contained in any element\n";
      target_field_host(i) = 0.0;
    }
    else
    {
      // Use the first element found
      auto const &result = values_host(start);
      int element_id = result.index;

      if constexpr (DIM == 2)
      {
        // Get element nodes and coordinates
        auto n0 = element_nodes_flat((element_id + 0) * 3);
        auto n1 = element_nodes_flat((element_id + 1) * 3);
        auto n2 = element_nodes_flat((element_id + 2) * 3);

        ArborX::Triangle<2, Coordinate> triangle{source_coords_host(n0),
                                                  source_coords_host(n1),
                                                  source_coords_host(n2)};

        auto bary = ArborX::Experimental::barycentricCoordinates(
            triangle, target_coords_host(i));

        // Interpolate
        target_field_host(i) =
            bary[0] * source_field_host(n0) + bary[1] * source_field_host(n1) +
            bary[2] * source_field_host(n2);
      }
      else
      {
        // Get element nodes and coordinates
        auto n0 = element_nodes_flat((element_id + 0) * 4);
        auto n1 = element_nodes_flat((element_id + 1) * 4);
        auto n2 = element_nodes_flat((element_id + 2) * 4);
        auto n3 = element_nodes_flat((element_id + 3) * 4);

        ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate> tet{
            source_coords_host(n0), source_coords_host(n1),
            source_coords_host(n2), source_coords_host(n3)};

        auto bary = ArborX::Experimental::barycentricCoordinates(
            tet, target_coords_host(i));

        // Interpolate
        target_field_host(i) =
            bary[0] * source_field_host(n0) + bary[1] * source_field_host(n1) +
            bary[2] * source_field_host(n2) + bary[3] * source_field_host(n3);
      }
    }
  }

  Kokkos::deep_copy(target_field_vals, target_field_host);

  timer->stop();

  // Write interpolated field to target mesh
  timer = time_monitor.getNewTimer("write_output");
  timer->start();

  auto *target_field =
      target_mesh->getSolutionField(target_field_name, target_block_names[0]);

  for (int i = 0; i < num_target_nodes; ++i)
  {
    auto *field_data = stk::mesh::field_data(*target_field, target_nodes[i]);
    KOKKOS_ASSERT(field_data != nullptr);
    *field_data = target_field_host(i);
  }

  timer->stop();

  if (verbose)
    time_monitor.summarize(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  ArborXBenchmark::MPIKokkosScopeGuard guard(argc, argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  using Coordinate = double;

  namespace bpo = boost::program_options;

  std::string source_filename;
  std::string target_filename;
  std::string output_filename;
  std::vector<std::string> source_block_names;
  std::vector<std::string> target_block_names;
  std::string source_field_name;
  std::string target_field_name;
  int restart_index;
  bool verbose;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "help message")
    ("source-filename,s", bpo::value<std::string>(&source_filename)->default_value("source.exo"), "source mesh filename")
    ("target-filename,t", bpo::value<std::string>(&target_filename)->default_value("target.exo"), "target mesh filename")
    ("output-filename,o", bpo::value<std::string>(&output_filename)->default_value("output.exo"), "output filename")
    ("source-block-names,b", bpo::value<std::vector<std::string>>(&source_block_names)->multitoken(), "source block names")
    ("target-block-names,B", bpo::value<std::vector<std::string>>(&target_block_names)->multitoken(), "target block names")
    ("source-field-name,f", bpo::value<std::string>(&source_field_name)->default_value("field"), "source field name")
    ("target-field-name,F", bpo::value<std::string>(&target_field_name)->default_value("interpolated_field"), "target field name")
    ("restart-index,r", bpo::value<int>(&restart_index)->default_value(-1), "restart index")
    ("verbose,v", bpo::bool_switch(&verbose), "verbose")
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv)
                 .options(desc)
                 .positional(bpo::positional_options_description{})
                 .run(),
             vm);
  bpo::notify(vm);

  if (comm_rank == 0)
  {
    std::cout << "ArborX version    : " << ArborX::version() << std::endl;
    std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
              << std::endl;
    std::cout << "#MPI ranks        : " << comm_size << std::endl;
  }

  if (vm.count("help") > 0)
  {
    if (comm_rank == 0)
      std::cout << desc << '\n';
    return 0;
  }

  if (!check_mesh_file(comm, source_filename))
    return 1;

  if (!check_mesh_file(comm, target_filename))
    return 2;

  if (comm_rank == 0)
  {
    printf("source filename    : %s\n", source_filename.c_str());
    printf("target filename    : %s\n", target_filename.c_str());
    printf("output filename    : %s\n", output_filename.c_str());
    printf("source field name  : %s\n", source_field_name.c_str());
    printf("target field name  : %s\n", target_field_name.c_str());
    printf("restart index      : %d%s\n", restart_index,
           (restart_index == -1 ? " (last)" : ""));
    printf("verbose            : %s\n", (verbose ? "true" : "false"));
  }

  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;

    Kokkos::Timer timer;

    auto source_mesh =
        build_mesh(comm, source_filename, source_block_names, restart_index);
    auto target_mesh =
        build_mesh(comm, target_filename, target_block_names, restart_index);

    // Add field to target mesh if it doesn't exist
    for (auto const &block_name : target_block_names)
      target_mesh->addSolutionField(target_field_name, block_name);

    if (comm_rank == 0)
      std::cout << "Mesh construction time: " << timer.seconds()
                << " seconds\n";

    ExecutionSpace space;

    if (source_mesh->getDimension() == 2)
      interpolate_field<2, Coordinate>(
          space, source_mesh, target_mesh, source_block_names,
          target_block_names, source_field_name, target_field_name, verbose);
    else
      interpolate_field<3, Coordinate>(
          space, source_mesh, target_mesh, source_block_names,
          target_block_names, source_field_name, target_field_name, verbose);

    space.fence();

    // Write output
    timer.reset();
    target_mesh->writeToExodus(output_filename);
    if (comm_rank == 0)
      std::cout << "Output write time: " << timer.seconds() << " seconds\n";
  }

  return 0;
}
