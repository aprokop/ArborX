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

#include <ArborX.hpp>

#include <Kokkos_DynRankView.hpp>

#include <Panzer_IntegrationRule.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_Workset.hpp>
#include <Panzer_Workset_Utilities.hpp> // getIntegrationRuleIndex

namespace
{
enum class Topology
{
  TRIANGLE,
  QUADRILATERAL,
  TETRAHEDRON,
  HEXAHEDRON
};

template <Topology T, typename MemorySpace>
struct Geometries
{
  Kokkos::DynRankView<double, MemorySpace> _sides;
};

} // namespace

template <Topology T, typename MemorySpace>
struct ArborX::AccessTraits<Geometries<T, MemorySpace>>
{
  using Self = Geometries<T, MemorySpace>;
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    if constexpr (T == Topology::HEXAHEDRON)
      return 2 * self._sides.extent(0);
    else
      return self._sides.extent(0);
  }
  KOKKOS_FUNCTION static auto get(Self const &self, size_t i)
  {
    auto const &sides = self._sides;
    if constexpr (T == Topology::TRIANGLE || T == Topology::QUADRILATERAL)
    {
      return ArborX::Experimental::Segment{{sides(i, 0, 0), sides(i, 0, 1)},
                                           {sides(i, 1, 0), sides(i, 1, 1)}};
    }
    else if constexpr (T == Topology::TETRAHEDRON)
    {
      return ArborX::Triangle{{sides(i, 0, 0), sides(i, 0, 1), sides(i, 0, 2)},
                              {sides(i, 1, 0), sides(i, 1, 1), sides(i, 1, 2)},
                              {sides(i, 2, 0), sides(i, 2, 1), sides(i, 2, 2)}};
    }
    else if constexpr (T == Topology::HEXAHEDRON)
    {
      auto const &sides = self._sides;
      // Split quad side into two triangles
      bool const odd = (i % 2);
      int j = (odd ? 1 : 2);
      int k = (odd ? 2 : 3);
      i /= 2;
      return ArborX::Triangle{{sides(i, 0, 0), sides(i, 0, 1), sides(i, 0, 2)},
                              {sides(i, j, 0), sides(i, j, 1), sides(i, j, 2)},
                              {sides(i, k, 0), sides(i, k, 1), sides(i, k, 2)}};
    }
  }
};

namespace ArborX::WallDistance
{
// #define WALL_DISTANCE_USE_REPLICATION

#ifdef WALL_DISTANCE_USE_REPLICATION
template <typename LocalSides, typename GlobalSides>
static void gatherGlobalSides(MPI_Comm comm, LocalSides const &local_sides,
                              GlobalSides &global_sides)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::WallDistance::gatherGlobalSides");

  using ExecutionSpace = typename GlobalSides::execution_space;
  using MemorySpace = typename GlobalSides::memory_space;

  ExecutionSpace space;

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  // Compose gather communication pattern.
  std::vector<int> global_counts(comm_size, 0);
  global_counts[comm_rank] = local_sides.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(global_counts.data()), 1, MPI_INT, comm);

  std::vector<int> offsets(comm_size, 0);
  std::exclusive_scan(global_counts.begin(), global_counts.end(),
                      offsets.begin(), 0);
  int num_global_sides = global_counts.back() + offsets.back();

  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), global_sides,
                 num_global_sides, local_sides.extent(1),
                 local_sides.extent(2));

  // Create host-side mirror for sides
  // Have to be careful with layouts
  auto local_sides_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, local_sides);
  auto global_sides_host = create_layout_right_mirror_view_no_init(
      space, Kokkos::HostSpace{}, global_sides);
  Kokkos::parallel_for(
      "ArborX::WallDistance::gatherGlobalSides::copy_local",
      Kokkos::RangePolicy(space, 0, local_sides.extent(0)),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < (int)local_sides.extent(1); ++j)
          for (int k = 0; k < (int)local_sides.extent(2); ++k)
            global_sides_host(offsets[comm_rank] + i, j, k) =
                local_sides_host(i, j, k);
      });
  space.fence();

  auto const side_data_size = local_sides.extent(1) * local_sides.extent(2);
  for (int rank = 0; rank < comm_size; ++rank)
  {
    offsets[rank] *= side_data_size;
    global_counts[rank] *= side_data_size;
  }

  // FIXME: hardcoded to MPI_DOUBLE
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_sides_host.data(),
                 global_counts.data(), offsets.data(), MPI_DOUBLE, comm);

  // For multi-dimensional views, we need to first copy into a separate
  // storage because of a different layout
  auto tmp_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::view_alloc(space, MemorySpace{}), global_sides_host);
  Kokkos::deep_copy(space, global_sides, tmp_view);
  space.fence();
}
#endif

// Check that the topologies in all element blocks are the same, and are ones
// from the list and return the key
static int get_topology_key(panzer_stk::STK_Interface const &mesh)
{
  std::vector<int> accepted_topologies = {
      shards::Tetrahedron<4>::key, shards::Hexahedron<8>::key,
      shards::Triangle<3>::key, shards::Quadrilateral<4>::key};

  std::vector<std::string> elem_block_names;
  mesh.getElementBlockNames(elem_block_names);

  auto equal_topologies = [&mesh](std::string const &a, std::string const &b) {
    return mesh.getCellTopology(a)->getKey() ==
           mesh.getCellTopology(b)->getKey();
  };

  if (std::adjacent_find(elem_block_names.begin(), elem_block_names.end(),
                         equal_topologies) != elem_block_names.end())
    throw std::runtime_error("Different topologies in element blocks");

  auto key = mesh.getCellTopology(elem_block_names[0])->getKey();
  if (std::find(accepted_topologies.begin(), accepted_topologies.end(), key) ==
      accepted_topologies.end())
    throw std::runtime_error(
        "Block topology is not Tet4, Hex8, Tri3, or Quad4");
  return key;
}

struct WallDistanceCallback
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &output) const
  {
    output(distance(ArborX::getGeometry(query), value));
  }
};

template <int DIM, typename ExecutionSpace>
auto buildIndex(ExecutionSpace const &space,
                panzer_stk::STK_Interface const &mesh,
                std::vector<std::string> const &wall_names)
{
  std::string prefix = "ArborX::WallDistance::buildIndex";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename ExecutionSpace::memory_space;

  // Get sideset names of sidesets
  std::vector<std::string> sideset_block_names;
  mesh.getSidesetNames(sideset_block_names);
  KOKKOS_ASSERT(!sideset_block_names.empty());

  // Get the local set of sides declared as walls from all block/sideset
  // combinations.
  std::vector<stk::mesh::Entity> local_side_entities;
  for (auto const &wall_name : wall_names)
  {
    if (std::find(sideset_block_names.begin(), sideset_block_names.end(),
                  wall_name) == sideset_block_names.end())
      continue;

    std::vector<stk::mesh::Entity> sideset_sides;
    mesh.getMySides(wall_name, sideset_sides);
    local_side_entities.insert(local_side_entities.end(), sideset_sides.begin(),
                               sideset_sides.end());
  }

  using Scalar = double;

  Kokkos::DynRankView<Scalar, MemorySpace> local_sides;
  mesh.getElementVertices(local_side_entities, local_sides);

  auto key = get_topology_key(mesh);

#ifdef WALL_DISTANCE_USE_REPLICATION
  Kokkos::View<Scalar ***, MemorySpace> global_sides(prefix + "global_sides", 0,
                                                     0, 0);
  MPI_Comm comm = Teuchos::getRawMpiComm(*mesh.getComm());
  gatherGlobalSides(comm, local_sides, global_sides);
  if constexpr (DIM == 2)
  {
    if (key == shards::Triangle<3>::key)
      return ArborX::DistributedTree(
          comm, space,
          Geometries<Topology::TRIANGLE, MemorySpace>{global_sides});
    else
      return ArborX::DistributedTree(
          comm, space, Geometries<Topology::QUADRILATERAL, MemorySpace>{sides});
  }
  else
  {
    if (key == shards::Tetrahedron<4>::key)
      return ArborX::DistributedTree(
          comm, space,
          Geometries<Topology::TETRAHEDRON, MemorySpace>{global_sides});
    else
      return ArborX::DistributedTree(
          comm, space,
          Geometries<Topology::HEXAHEDRON, MemorySpace>{global_sides});
  }
#else
  if constexpr (DIM == 2)
  {
    if (key == shards::Triangle<3>::key)
      return ArborX::BoundingVolumeHierarchy(
          space, Geometries<Topology::TRIANGLE, MemorySpace>{local_sides});
    else
      return ArborX::BoundingVolumeHierarchy(
          space, Geometries<Topology::QUADRILATERAL, MemorySpace>{local_sides});
  }
  else
  {
    if (key == shards::Tetrahedron<4>::key)
      return ArborX::BoundingVolumeHierarchy(
          space, Geometries<Topology::TETRAHEDRON, MemorySpace>{local_sides});
    else
      return ArborX::BoundingVolumeHierarchy(
          space, Geometries<Topology::HEXAHEDRON, MemorySpace>{local_sides});
  }
#endif
}

template <typename ExecutionSpace, typename Index>
auto distance(ExecutionSpace const &space, Index const &index,
              std::vector<panzer::Workset> const &worksets,
              panzer::IntegrationRule const &int_rule)
{
  std::string prefix = "ArborX::WallDistance::distance";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename Index::memory_space;

  constexpr int DIM =
      ArborX::GeometryTraits::dimension_v<typename Index::value_type>;
  using Scalar = double;
  using Point = ArborX::Point<DIM, Scalar>;

  PHX::MDField<Scalar, panzer::Cell, panzer::Point> blah_distances(
      prefix + "workset_distances", int_rule.dl_scalar);

  auto const num_worksets = worksets.size();
  auto const max_num_cells_per_workset = blah_distances.extent(0);
  int const num_int_points_per_cell = blah_distances.extent(1);

  auto int_rule_index =
      panzer::getIntegrationRuleIndex(int_rule.order(), worksets[0]);

  size_t num_queries = 0;
  std::vector<size_t> workset_sizes(num_worksets);
  for (size_t workset_id = 0; workset_id < num_worksets; ++workset_id)
  {
    auto const &workset = worksets[workset_id];
    auto const num_cells = workset.num_cells;
    workset_sizes[workset_id] = num_cells;
    num_queries += num_cells * num_int_points_per_cell;
  }

  Kokkos::View<Point *, MemorySpace> points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, prefix + "points"),
      num_queries);
  for (size_t workset_id = 0, queries_offset = 0; workset_id < num_worksets;
       ++workset_id)
  {
    auto const &workset = worksets[workset_id];
    auto const num_cells = workset.num_cells;

    if (num_cells == 0)
      continue;

    auto const &ip_coords = workset.int_rules[int_rule_index]->ip_coordinates;

    Kokkos::parallel_for(
        prefix + "create_queries", Kokkos::RangePolicy(space, 0, num_cells),
        KOKKOS_LAMBDA(int cell) {
          auto offset = queries_offset + cell * num_int_points_per_cell;
          Point p;
          for (int int_point = 0; int_point < num_int_points_per_cell;
               ++int_point)
          {
            for (int d = 0; d < DIM; ++d)
              p[d] = ip_coords(cell, int_point, d);
            points(offset++) = p;
          }
        });
    queries_offset += num_cells * num_int_points_per_cell;
  }
  auto queries = ArborX::Experimental::make_nearest(points, 1);

  Kokkos::View<int *, MemorySpace> offset(prefix + "offset", 0);
  Kokkos::View<Scalar *, MemorySpace> distances(prefix + "distances", 0);
  index.query(space, queries,
#ifdef WALL_DISTANCE_USE_REPLICATION
              WallDistanceCallback{},
#else
              ArborX::Experimental::declare_callback_constrained(
                  WallDistanceCallback{}),
#endif
              distances, offset);

  space.fence();

  Kokkos::View<Scalar ***, MemorySpace> workset_distances(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::WallDistance::workset_distances"),
      num_worksets, max_num_cells_per_workset, num_int_points_per_cell);

  for (size_t workset_id = 0, workset_offset = 0; workset_id < num_worksets;
       ++workset_id)
  {
    auto const num_cells = workset_sizes[workset_id];
    if (num_cells == 0)
      continue;

    Kokkos::parallel_for(
        "ArborX::WallDistance::reshape_distances",
        Kokkos::RangePolicy(space, 0, num_cells), KOKKOS_LAMBDA(int cell) {
          auto offset = workset_offset + cell * num_int_points_per_cell;
          for (int int_point = 0; int_point < num_int_points_per_cell;
               ++int_point)
            workset_distances(workset_id, cell, int_point) =
                distances(offset++);
        });
    workset_offset += num_cells * num_int_points_per_cell;
  }
  return workset_distances;
}

} // namespace ArborX::WallDistance
