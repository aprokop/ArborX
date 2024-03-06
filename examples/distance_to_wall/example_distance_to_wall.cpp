/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>
#include <ArborX_HyperTriangle.hpp>
#include <ArborX_Version.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>

using ArborX::ExperimentalHyperGeometry::Triangle;

int getDataDimension(std::string const &filename, bool binary)
{
  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Error reading file \"" + filename + "\"");

  int num_points;
  int dim;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  input.close();

  return dim;
}

template <int DIM>
auto loadPointsData(std::string const &filename, bool binary = true)
{
  std::cout << "Reading points in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }

  ARBORX_ASSERT(dim == DIM);

  using Point = ArborX::ExperimentalHyperGeometry::Point<DIM>;

  std::vector<Point> v(num_points);
  if (!binary)
  {
    auto it = std::istream_iterator<float>(input);
    for (int i = 0; i < num_points; ++i)
      for (int d = 0; d < DIM; ++d)
        v[i][d] = *it++;
  }
  else
  {
    // Directly read into a point
    input.read(reinterpret_cast<char *>(v.data()), num_points * sizeof(Point));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  return v;
}

auto loadTrianglesData(std::string const &filename, bool binary = true)
{
  std::cout << "Reading triangles in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_triangles = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_triangles;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_triangles), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  assert(dim == 3);

  std::vector<int> v(3 * num_triangles);
  if (!binary)
  {
    auto it = std::istream_iterator<int>(input);
    for (int i = 0; i < num_triangles; ++i)
      for (int d = 0; d < 3; ++d)
        v[3 * i + d] = *it++ - 1;
  }
  else
  {
    // Directly read into a point
    input.read(reinterpret_cast<char *>(v.data()),
               num_triangles * 3 * sizeof(int));
  }
  input.close();
  std::cout << "done\nRead in " << num_triangles << " " << dim << "D triangles"
            << std::endl;

  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

struct Params
{
  bool binary;
  std::string points_filename;
  std::string triangle_points_filename;
  std::string triangles_filename;
};

template <typename MemorySpace, typename Point>
struct Triangles
{
  using triangle_type = ArborX::ExperimentalHyperGeometry::Triangle<
      ArborX::GeometryTraits::dimension_v<Point>,
      ArborX::GeometryTraits::coordinate_type_t<Point>>;

  Kokkos::View<Point *, MemorySpace> _points;
  Kokkos::View<int *, MemorySpace> _triangle_vertices;
};

template <typename MemorySpace, typename Point>
class ArborX::AccessTraits<Triangles<MemorySpace, Point>, ArborX::PrimitivesTag>
{
  using Self = Triangles<MemorySpace, Point>;

public:
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION auto size(Self const &self)
  {
    return self._triangle_vertices.size() / 3;
  }
  static KOKKOS_FUNCTION auto get(Self const &self, int i)
  {
    using Triangle = typename Self::triangle_type;

    auto const &vertices = self._triangle_vertices;
    return Triangle{self._points(vertices(3 * i + 0)),
                    self._points(vertices(3 * i + 1)),
                    self._points(vertices(3 * i + 2))};
  }
};

struct DistanceCallback
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value,
                                  OutputFunctor const &out) const
  {
    using ArborX::Details::distance;
    out(distance(ArborX::getGeometry(predicate), value));
  }
};

template <typename Points, typename Triangles>
void writeVtk(std::string const &filename, Points const &points,
              Triangles const &triangles)
{
  int const num_vertices = points.size();
  int const num_elements = triangles.size() / 3;

  constexpr int DIM =
      ArborX::GeometryTraits::dimension_v<typename Points::value_type>;

  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);
  auto triangles_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, triangles);

  std::ofstream out(filename);

  out << "# vtk DataFile Version 2.0\n";
  out << "Mesh example\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n\n";
  out << "POINTS " << num_vertices << " float\n";
  for (int i = 0; i < num_vertices; ++i)
  {
    for (int d = 0; d < DIM; ++d)
      out << " " << points(i)[d];
    out << '\n';
  }

  int const num_cell_vertices = 3;
  out << "\nPOLYGONS " << num_elements << " "
      << (num_elements * (1 + num_cell_vertices)) << '\n';
  for (int i = 0; i < num_elements; ++i)
  {
    out << num_cell_vertices;
    for (int j = 0; j < num_cell_vertices; ++j)
      out << " " << triangles(i * num_cell_vertices + j);
    out << '\n';
  }
}

template <int DIM>
void main_(Params const &params)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  ExecutionSpace space;

  auto points = vec2view<MemorySpace>(
      loadPointsData<DIM>(params.points_filename, params.binary),
      "Examples::points");
  auto triangle_points = vec2view<MemorySpace>(
      loadPointsData<DIM>(params.triangle_points_filename, params.binary),
      "Examples::triangle_points");
  auto triangle_vertices = vec2view<MemorySpace>(
      loadTrianglesData(params.triangles_filename, params.binary),
      "Examples::triangles");

  writeVtk("mesh.vtk", triangle_points, triangle_vertices);

  Triangles<MemorySpace, typename decltype(triangle_points)::value_type>
      triangles{triangle_points, triangle_vertices};

  ArborX::BVH<MemorySpace, typename decltype(triangles)::triangle_type> index(
      space, triangles);

  Kokkos::View<int *, MemorySpace> offset("Examples::offsets", 0);
  Kokkos::View<float *, MemorySpace> distances("Examples::distances", 0);
  index.query(space, ArborX::Experimental::make_nearest(points, 1),
              DistanceCallback{}, distances, offset);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard;

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
            << std::endl;

  namespace bpo = boost::program_options;

  Params params;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "binary", bpo::bool_switch(&params.binary), "binary file indicator")
      ( "points-filename", bpo::value<std::string>(&params.points_filename), "filename containing points data" )
      ( "triangle-points-filename", bpo::value<std::string>(&params.triangle_points_filename), "filename containing triangle points data" )
      ( "triangles-filename", bpo::value<std::string>(&params.triangles_filename), "filename containing triangles data" )
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  if (params.points_filename.empty())
  {
    std::cerr << "Please provide a file containing points data\n";
    return 1;
  }
  if (params.triangles_filename.empty())
  {
    std::cerr << "Please provide a file containing triangles data\n";
    return 1;
  }

  int dim = getDataDimension(params.triangles_filename, params.binary);

  switch (dim)
  {
  case 3:
    main_<3>(params);
    break;
  default:
    std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    return 1;
  }

  return 0;
}
