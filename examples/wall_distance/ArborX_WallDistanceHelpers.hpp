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

#ifndef ARBORX_WALL_DISTANCE_HELPERS_HPP
#define ARBORX_WALL_DISTANCE_HELPERS_HPP

#include <ArborX_Segment.hpp>
#include <ArborX_Triangle.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Predicates.hpp>

#include <Kokkos_DynRankView.hpp>

#include <numeric> // exclusive_scan
#include <vector>

#include <Panzer_STK_Interface.hpp>

namespace ArborX::Details
{
enum class Topology
{
  TRIANGLE,
  QUADRILATERAL,
  TETRAHEDRON,
  HEXAHEDRON
};

template <Topology T, typename Sides>
struct Geometries
{
  Sides _sides;
};

} // namespace ArborX::Details

template <ArborX::Details::Topology T, typename Sides>
struct ArborX::AccessTraits<ArborX::Details::Geometries<T, Sides>>
{
  using Self = ArborX::Details::Geometries<T, Sides>;
  using memory_space = typename Sides::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    using namespace ArborX::Details;

    if constexpr (T == Topology::HEXAHEDRON)
      return 2 * self._sides.extent(0);
    else
      return self._sides.extent(0);
  }

  KOKKOS_FUNCTION static auto get(Self const &self, size_t i)
  {
    using namespace ArborX::Details;

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

namespace ArborX::Details
{
#define WALL_DISTANCE_USE_REPLICATION

#ifdef WALL_DISTANCE_USE_REPLICATION
namespace internal
{
template <typename PointerType>
struct PointerDepth
{
  static constexpr int value = 0;
};

template <typename PointerType>
struct PointerDepth<PointerType *>
{
  static constexpr int value = PointerDepth<PointerType>::value + 1;
};

template <typename PointerType, std::size_t N>
struct PointerDepth<PointerType[N]>
{
  static constexpr int value = PointerDepth<PointerType>::value;
};
} // namespace internal

template <typename View, typename ExecutionSpace, typename MemorySpace>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename ExecutionSpace::memory_space>
create_layout_right_mirror_view_no_init(ExecutionSpace const &execution_space,
                                        MemorySpace const &memory_space,
                                        View const &src)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);

  constexpr bool has_compatible_layout =
      (std::is_same_v<typename View::array_layout, Kokkos::LayoutRight> ||
       (View::rank == 1 &&
        (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>)));
  constexpr bool has_compatible_memory_space =
      std::is_same_v<typename View::memory_space, MemorySpace>;

  if constexpr (has_compatible_layout && has_compatible_memory_space)
  {
    return src;
  }
  else
  {
    constexpr int pointer_depth =
        internal::PointerDepth<typename View::traits::data_type>::value;
    return Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                        MemorySpace>(
        Kokkos::view_alloc(
            execution_space, memory_space, Kokkos::WithoutInitializing,
            std::string(src.label()).append("_layout_right_mirror")),
        src.extent(0), pointer_depth > 1 ? src.extent(1) : KOKKOS_INVALID_INDEX,
        pointer_depth > 2 ? src.extent(2) : KOKKOS_INVALID_INDEX,
        pointer_depth > 3 ? src.extent(3) : KOKKOS_INVALID_INDEX,
        pointer_depth > 4 ? src.extent(4) : KOKKOS_INVALID_INDEX,
        pointer_depth > 5 ? src.extent(5) : KOKKOS_INVALID_INDEX,
        pointer_depth > 6 ? src.extent(6) : KOKKOS_INVALID_INDEX,
        pointer_depth > 7 ? src.extent(7) : KOKKOS_INVALID_INDEX);
  }
}

template <typename View>
inline auto create_layout_right_mirror_view_no_init(View const &src)
{
  typename View::traits::host_mirror_space::execution_space exec;
  auto mirror_view = create_layout_right_mirror_view_no_init(
      exec, typename View::traits::host_mirror_space{}, src);
  exec.fence();
  return mirror_view;
}

template <typename LocalSides>
void getLocalSides(panzer_stk::STK_Interface const &mesh,
                   std::vector<std::string> const &wall_names,
                   LocalSides &local_sides)
{
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

  mesh.getElementVertices(local_side_entities, local_sides);
}

template <typename ExecutionSpace, typename LocalSides, typename GlobalSides>
static void gatherGlobalSides(MPI_Comm comm, ExecutionSpace const &space,
                              LocalSides const &local_sides,
                              GlobalSides &global_sides)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::WallDistance::gatherGlobalSides");

  using MemorySpace = typename GlobalSides::memory_space;

  int num_local = local_sides.extent(0);
  auto const data_size = local_sides.extent(1) * local_sides.extent(2);

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  // Compose gather communication pattern.
  std::vector<int> global_counts(comm_size, 0);
  global_counts[comm_rank] = local_sides.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(global_counts.data()), 1, MPI_INT, comm);

  std::vector<int> offsets(comm_size + 1, 0);
  std::exclusive_scan(global_counts.begin(), global_counts.end(),
                      offsets.begin(), 0);
  offsets[comm_size] = offsets[comm_size - 1] + global_counts.back();
  int num_global_sides = offsets.back();

  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), global_sides,
                 num_global_sides, local_sides.extent(1),
                 local_sides.extent(2));

  // Create host-side mirror for sides
  // Have to be careful with layouts
  auto local_sides_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, local_sides);
  Kokkos::fence();
  auto global_sides_host =
      create_layout_right_mirror_view_no_init(global_sides);

  auto const offset_rank = offsets[comm_rank];
  for (int i = 0; i < num_local; ++i)
    for (int j = 0; j < (int)local_sides.extent(1); ++j)
      for (int k = 0; k < (int)local_sides.extent(2); ++k)
        global_sides_host(offset_rank + i, j, k) = local_sides_host(i, j, k);

  for (int rank = 0; rank < comm_size; ++rank)
  {
    offsets[rank] *= data_size;
    global_counts[rank] *= data_size;
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
    output(distance(getGeometry(query), value));
  }
};

} // namespace ArborX::Details

#endif
