#pragma once
#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>

#include "point.hpp"

typedef std::pair<int, int> edge_t;
typedef std::pair<edge_t, double> wtEdge_t;

class parallelBoruvka_t
{
  const std::vector<Point> &_points; // list of points

  std::vector<edge_t> _mst;

public:
  parallelBoruvka_t(const std::vector<std::vector<double>> &points);

  void writeMST(std::ofstream &ofileName);
  std::vector<wtEdge_t> weightedMST();
};
