#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "parallel_boruvka.hpp"

class incidenceMatrix_t
{
  int m_npts;
  std::vector<wtEdge_t> &m_wtMST;
  std::vector<int> m_vEdgeListIndex;
  std::vector<int> m_edgeIdx;

public:
  incidenceMatrix_t(std::vector<wtEdge_t> &wtMST);
};

incidenceMatrix_t::incidenceMatrix_t(std::vector<wtEdge_t> &wtMST)
    : m_wtMST(wtMST)
{
  m_npts = m_wtMST.size() + 1;
  m_vEdgeListIndex.resize(m_npts + 1);
  m_edgeIdx.resize(2 * m_npts - 2); // double the number of edges

  std::vector<int> edgeCounts(m_npts + 1, 0); // number of edge count

  // get edge count
  for (auto wte : m_wtMST)
  {
    edgeCounts[wte.first.first]++;
    edgeCounts[wte.first.second]++;
  }

  // do a exclusive prefix on edge counts
}
