#pragma once
#include <algorithm>
#include <vector>

#include "incidence_matrix.hpp"
#include "parallel_boruvka.hpp"

typedef std::pair<edge_t, double> wtEdge_t;

class hdbscan_t
{
  int m_npts, m_dim; //
  int m_kpts;        // used for HDBSCAN core distance computation
  std::vector<std::vector<double>> m_points;
  std::vector<wtEdge_t> wtMST;
  incidenceMatrix_t m_inmatMST;

public:
  hdbscan_t(std::vector<std::vector<double>> &pts, int k_pts);
};

struct compareWtEdges
{
  inline bool operator()(const wtEdge_t &wte1, const wtEdge_t &wte2)
  {
    return (wte1.second < wte2.second);
  }
};

hdbscan_t::hdbscan_t(std::vector<std::vector<double>> &pts, int k_pts)
    : m_points(pts)
    , m_kpts(k_pts)
{
  m_npts = m_points.size();
  m_dim = m_points[0].size();

  parallelBoruvka_t pBv(pts);
  wtMST = pBv.weightedMST();

  // sort the edges
  std::sort(wtMST.begin(), wtMST.end(), compareWtEdges());
};
