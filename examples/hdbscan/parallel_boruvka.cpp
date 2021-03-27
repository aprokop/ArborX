#include "parallel_boruvka.hpp"

#include <iostream>
#include <numeric> // iota

#include "utility.hpp"

parallelBoruvka_t::parallelBoruvka_t(const std::vector<Point> &points)
    : m_points(points)
{
  auto const num_points = points.size();

  // initialization
  m_numComponents = num_points; // number of components;
  m_C.resize(num_points);
  m_listC.resize(num_points);
  m_MST.resize(num_points - 1);
  m_pfxsum.resize(num_points + 1);

  m_parent.resize(num_points);

  // initialization
  for (int i = 0; i < num_points; i++)
  {
    m_C[i] = i;     // component of vertex i
    m_listC[i] = i; // list of components
    // m_parent[i] =i;
  }

  std::vector<int> xC(num_points); // next component
  std::vector<int> component_edge_src(num_points);
  std::vector<int> next_edge(num_points);
  std::vector<double> next_edge_len(num_points);

  std::iota(xC.begin(), xC.end(), 0);
  std::iota(component_edge_src.begin(), component_edge_src.end(), 0);
  std::iota(next_edge.begin(), next_edge.end(), 0);
  while (m_numComponents > 1)
  {
    updateCandidateEdges(next_edge, next_edge_len);
    determineComponentEdges(next_edge, next_edge_len, component_edge_src);
    updateComponents(xC, next_edge, component_edge_src);
  }
}

void parallelBoruvka_t::updateMST(std::vector<int> &xC,
                                  std::vector<int> const &next_edge,
                                  std::vector<int> const &component_edge_src)
{
  auto const num_points = m_points.size();

  int numEdgesMST = num_points - m_numComponents;
  for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
  {
    int cc = m_listC[c_idx];
    if (m_C[cc] != xC[cc]) // add its edge
    {
      int cc_SrcVertex = component_edge_src[cc];
      int cc_DstVertex = next_edge[cc_SrcVertex];
      // m_MST[numEdgesMST + m_pfxsum[c_idx]] = std::make_pair(cc_SrcVertex,
      // cc_DstVertex); std::cout << "Adding (" << cc_SrcVertex << " " <<
      // cc_DstVertex << ") at " << numEdgesMST + m_pfxsum[c_idx] << "\n";
      m_MST[numEdgesMST + m_pfxsum[c_idx]] = m_parent[cc];
      std::cout << "Adding (" << m_parent[cc].first << " "
                << m_parent[cc].second << ") at "
                << numEdgesMST + m_pfxsum[c_idx] << "\n";
    }
    else
    {
      m_listC[c_idx - m_pfxsum[c_idx]] = cc;
    }
  }
}

void parallelBoruvka_t::updateComponents(
    std::vector<int> &xC, std::vector<int> const &next_edge,
    std::vector<int> const &component_edge_src)
{
  // parallel label propagation
  int numChanges = 1;
  while (numChanges > 0)
  {
    numChanges = 0;
    for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
    {
      int cc = m_listC[c_idx];
      int cc_SrcVertex = component_edge_src[cc];
      int cc_DstVertex = next_edge[cc_SrcVertex];
      int cc_next = m_C[cc_DstVertex];

      if (xC[cc] > xC[cc_next])
      {
        // int min_cc = std::min(xC[cc], xC[cc_next]);
        xC[cc] = xC[cc_next];
        m_parent[cc] = std::make_pair(cc_SrcVertex, cc_DstVertex);
        // xC[cc_next] = min_cc;
        numChanges++;
      }
      else if (xC[cc] < xC[cc_next])
      {
        xC[cc_next] = xC[cc];
        m_parent[cc_next] = std::make_pair(cc_SrcVertex, cc_DstVertex);
        numChanges++;
      }
    }
  }

  // adding edges
  for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
  {
    int cc = m_listC[c_idx];
    if (m_C[cc] == xC[cc])
      m_pfxsum[c_idx] = 0;
    else
      m_pfxsum[c_idx] = 1;
  }
  // compute inclusive prefix sum
  prefixSumExclusive(m_numComponents, m_pfxsum.data());

  // Update MST
  updateMST(xC, next_edge, component_edge_src);

  // Update component Labels
  auto const num_points = m_points.size();
  for (int pt = 0; pt < num_points; pt++)
  {
    m_C[pt] = xC[m_C[pt]];
  }

  // update number of components
  m_numComponents -= m_pfxsum[m_numComponents];
  std::cout << "Number of components is " << m_numComponents << "\n";
  for (int i = 0; i < m_numComponents; i++)
    std::cout << m_listC[i] << " ";
  std::cout << "\n";
}

void parallelBoruvka_t::updateCandidateEdges(std::vector<int> &next_edge,
                                             std::vector<double> &next_edge_len)
{
  auto const num_points = m_points.size();

  // find and store potential new edges
  for (int pt = 0; pt < num_points; pt++)
    if (m_C[next_edge[pt]] == m_C[pt])
    {
      auto r = findClosestPointWithDifferentLabel(m_points, m_C, pt);
      next_edge[pt] = r.first;
      next_edge_len[pt] = r.second;
    }
}

void parallelBoruvka_t::determineComponentEdges(
    std::vector<int> const &next_edge, std::vector<double> const &next_edge_len,
    std::vector<int> &component_edge_src)
{
  auto const num_points = m_points.size();

  const double infty = std::numeric_limits<double>::infinity();

  // for each component find the edge with minimum edge len
  std::vector<double> component_edge_len(num_points, infty);
  for (int i = 0; i < num_points; i++)
  {
    int const label = m_C[i];
    if (next_edge_len[i] < component_edge_len[label])
    {
      component_edge_len[label] = next_edge_len[i];
      component_edge_src[label] = i;
    }
  }
}

void parallelBoruvka_t::writeMST(std::ofstream &outfile)
{
  for (auto edge : m_MST)
  {
    outfile << edge.first << " " << edge.second << "\n";
  }
}

std::vector<wtEdge_t> parallelBoruvka_t::weightedMST()
{
  std::vector<wtEdge_t> wtmst;
  for (auto edge : m_MST)
  {
    wtEdge_t wte(edge,
                 distanceSquared(m_points[edge.first], m_points[edge.second]));
    wtmst.push_back(wte);
  }
  return wtmst;
}
