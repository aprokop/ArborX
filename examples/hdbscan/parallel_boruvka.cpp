#include "parallel_boruvka.hpp"

#include <iostream>

parallelBoruvka_t::parallelBoruvka_t(
    const std::vector<std::vector<double>> &points)
    : m_points(points)
{
  // initialization
  m_npts = points.size();
  m_numComponents = m_npts; // number of components;
  m_C.resize(m_npts);
  m_xC.resize(m_npts);
  m_listC.resize(m_npts);
  m_MST.resize(m_npts - 1);
  m_nextEdge.resize(m_npts);
  m_nextEdgeLen.resize(m_npts);
  m_pfxsum.resize(m_npts + 1);

  m_componentEdgeLen.resize(
      m_npts); // edgeLength of candidate edge of the component
  m_componentEdgeSrc.resize(
      m_npts); // starting vertex of component's candidate edge

  m_parent.resize(m_npts);

  // initialization
  for (int i = 0; i < m_npts; i++)
  {
    m_C[i] = i;     // component of vertex i
    m_xC[i] = i;    // next component of vertex i
    m_listC[i] = i; // list of components
    m_nextEdge[i] = i;
    m_componentEdgeSrc[i] = i;
    m_componentEdgeLen[i] = INFTY;
    // m_parent[i] =i;
  }

  //
  while (m_numComponents > 1)
  {
    computeCandidateEdges();
    updateComponents();
  }
}

void prefixSumExclusive(int n, int *A)
{
  // int tmp = A[0];
  // A[0] =0;
  for (int i = 1; i < n; i++)
  {
    A[i] = A[i] + A[i - 1];
  }

  // shift right
  for (int i = n; i > 0; i--)
  {
    A[i] = A[i - 1];
  }
  A[0] = 0;
}

void parallelBoruvka_t::updateMST()
{
  int numEdgesMST = m_npts - m_numComponents;
  for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
  {
    int cc = m_listC[c_idx];
    if (m_C[cc] != m_xC[cc]) // add its edge
    {
      int cc_SrcVertex = m_componentEdgeSrc[cc];
      int cc_DstVertex = m_nextEdge[cc_SrcVertex];
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

void parallelBoruvka_t::updateComponents()
{
  // parallel label propagation
  int numChanges = 1;
  while (numChanges > 0)
  {
    numChanges = 0;
    for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
    {
      int cc = m_listC[c_idx];
      int cc_SrcVertex = m_componentEdgeSrc[cc];
      int cc_DstVertex = m_nextEdge[cc_SrcVertex];
      int cc_next = m_C[cc_DstVertex];

      // check next components
      // if(m_xC[cc] != m_xC[cc_next])
      // {
      //     int min_cc = std::min(m_xC[cc], m_xC[cc_next]);
      //     m_xC[cc] = min_cc;
      //     m_xC[cc_next] = min_cc;
      //     numChanges++;
      // }

      if (m_xC[cc] > m_xC[cc_next])
      {
        // int min_cc = std::min(m_xC[cc], m_xC[cc_next]);
        m_xC[cc] = m_xC[cc_next];
        m_parent[cc] = std::make_pair(cc_SrcVertex, cc_DstVertex);
        // m_xC[cc_next] = min_cc;
        numChanges++;
      }
      else if (m_xC[cc] < m_xC[cc_next])
      {
        m_xC[cc_next] = m_xC[cc];
        m_parent[cc_next] = std::make_pair(cc_SrcVertex, cc_DstVertex);
        numChanges++;
      }
    }
  }

  // adding edges
  for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
  {
    int cc = m_listC[c_idx];
    if (m_C[cc] == m_xC[cc])
      m_pfxsum[c_idx] = 0;
    else
      m_pfxsum[c_idx] = 1;
  }
  // compute inclusive prefix sum
  prefixSumExclusive(m_numComponents, m_pfxsum.data());

  // Update MST
  updateMST();

  // Update component Labels
  for (int pt = 0; pt < m_npts; pt++)
  {
    m_C[pt] = m_xC[m_C[pt]];
  }

  // update number of components
  m_numComponents -= m_pfxsum[m_numComponents];
  std::cout << "Number of components is " << m_numComponents << "\n";
  for (int i = 0; i < m_numComponents; i++)
    std::cout << m_listC[i] << " ";
  std::cout << "\n";
}

double parallelBoruvka_t::distSqEuclidean(int v1, int v2)
{
  double out = 0;
  int dim = m_points[0].size();
  for (int i = 0; i < dim; i++)
  {
    double diff = (m_points[v1][i] - m_points[v2][i]);
    out += diff * diff;
  }
  return out;
}

void parallelBoruvka_t::computeNextNeighbour(int pt)
{
  // m_nextEdge[pt] = ?;
  // m_nextEdgeLen[pt] = ?;

  m_nextEdgeLen[pt] = INFTY;
  // linear search
  for (int i = 0; i < m_npts; i++)
  {
    if (m_C[i] != m_C[pt] && distSqEuclidean(pt, i) < m_nextEdgeLen[pt])
    {
      m_nextEdgeLen[pt] = distSqEuclidean(pt, i);
      m_nextEdge[pt] = i;
    }
  }
}

void parallelBoruvka_t::computeCandidateEdges()
{
  // find potential new edges
  for (int pt = 0; pt < m_npts; pt++)
    if (m_C[m_nextEdge[pt]] == m_C[pt])
      computeNextNeighbour(pt);

  for (int cc = 0; cc < m_numComponents; cc++)
  {
    m_componentEdgeLen[m_listC[cc]] = INFTY;
  }

  // for each component find the edge with minimum edge len
  for (int pt = 0; pt < m_npts; pt++)
  {
    if (m_nextEdgeLen[pt] < m_componentEdgeLen[m_C[pt]])
    {
      m_componentEdgeLen[m_C[pt]] = m_nextEdgeLen[pt];
      m_componentEdgeSrc[m_C[pt]] = pt;
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
    wtEdge_t wte(edge, distSqEuclidean(edge.first, edge.second));
    wtmst.push_back(wte);
  }
  return wtmst;
}
