#include "utility.hpp"

double distanceSquared(Point const &p1, Point const &p2)
{
  // FIXME
  int const dim = p1.size();

  double dist = 0.0;
  for (int d = 0; d < dim; d++)
    dist += (p1[d] - p2[d]) * (p1[d] - p2[d]);

  return dist;
}

std::pair<int, double>
findClosestPointWithDifferentLabel(std::vector<Point> const &points,
                                   std::vector<int> const &labels, int index)
{
  int const num_points = points.size();

  int min_index = -1;
  auto min_distance = std::numeric_limits<double>::infinity();
  for (int i = 0; i < num_points; ++i)
    if (labels[i] != labels[index])
    {
      auto d = distanceSquared(points[i], points[index]);
      if (d < min_distance)
      {
        min_index = i;
        min_distance = d;
      }
    }
  return std::make_pair(min_index, min_distance);
}
