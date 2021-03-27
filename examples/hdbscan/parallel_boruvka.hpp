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
  int m_numComponents;

  const std::vector<Point> &m_points; // list of points

  std::vector<edge_t> m_MST;

  std::vector<int>
      m_pfxsum; // used for computing next address, length should be n_pts+1

  std::vector<edge_t> m_parent; // parent Edge

public:
  parallelBoruvka_t(const std::vector<std::vector<double>> &points);
  int numComponents() { return m_numComponents; }

  void updateComponents(std::vector<int> &labels, std::vector<int> &xC,
                        std::vector<int> const &next_edge,
                        std::vector<int> const &component_edge_src,
                        std::vector<int> &listC);
  void updateMST(std::vector<int> const &labels, std::vector<int> const &xC,
                 std::vector<int> const &next_edge,
                 std::vector<int> const &component_edge_src,
                 std::vector<int> &listC);

  void writeMST(std::ofstream &ofileName);
  std::vector<wtEdge_t> weightedMST();
};
