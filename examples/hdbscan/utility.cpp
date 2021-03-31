#include "utility.hpp"

#include <cmath>
#include <limits>

bool compare_edges_less(std::tuple<int, int, double> const &edge1,
                        std::tuple<int, int, double> const &edge2)
{
  double dist1 = std::get<2>(edge1);
  double dist2 = std::get<2>(edge2);
  if (dist1 < dist2)
    return true;
  if (dist1 > dist2)
    return false;

  auto vmin1 = std::min(std::get<0>(edge1), std::get<1>(edge1));
  auto vmin2 = std::min(std::get<0>(edge2), std::get<1>(edge2));
  if (vmin1 < vmin2)
    return true;
  if (vmin1 > vmin2)
    return false;

  auto vmax1 = std::max(std::get<0>(edge1), std::get<1>(edge1));
  auto vmax2 = std::max(std::get<0>(edge2), std::get<1>(edge2));
  if (vmax1 < vmax2)
    return true;
  return false;
}

double distance(Point const &p1, Point const &p2)
{
  // FIXME
  int const dim = p1.size();

  double dist = 0.0;
  for (int d = 0; d < dim; d++)
    dist += (p1[d] - p2[d]) * (p1[d] - p2[d]);

  return std::sqrt(dist);
}

int kNNWithFilter(std::vector<Point> const &points,
                  std::vector<int> const &components, int index)
{
  int const num_points = points.size();

  int min_index = -1;
  auto min_distance = std::numeric_limits<double>::infinity();
  for (int i = 0; i < num_points; ++i)
    if (components[i] != components[index])
    {
      auto d = distance(points[i], points[index]);
      if (compare_edges_less(std::make_tuple(index, i, d),
                             std::make_tuple(index, min_index, min_distance)))
      {
        min_index = i;
        min_distance = d;
      }
    }
  return min_index;
}
