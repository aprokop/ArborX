#include "parallel_boruvka.hpp"

#include <iostream>
#include <numeric> // iota

#include "utility.hpp"

parallelBoruvka_t::parallelBoruvka_t(const std::vector<Point> &points)
    : m_points(points)
{
  // initialization
  m_npts = points.size();
  m_numComponents = m_npts; // number of components;
  m_C.resize(m_npts);
  m_listC.resize(m_npts);
  m_MST.resize(m_npts - 1);
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
    m_listC[i] = i; // list of components
    m_componentEdgeSrc[i] = i;
    m_componentEdgeLen[i] = INFTY;
    // m_parent[i] =i;
  }

  std::vector<int> xC(m_npts); // next component
  std::vector<int> nextEdge(m_npts);
  std::vector<double> nextEdgeLen(m_npts);

  std::iota(xC.begin(), xC.end(), 0);
  std::iota(nextEdge.begin(), nextEdge.end(), 0);
  while (m_numComponents > 1)
  {
    computeCandidateEdges(nextEdge, nextEdgeLen);
    updateComponents(xC, nextEdge, nextEdgeLen);
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

void parallelBoruvka_t::updateMST(std::vector<int> &xC,
                                  std::vector<int> &nextEdge)
{
  int numEdgesMST = m_npts - m_numComponents;
  for (int c_idx = 0; c_idx < m_numComponents; c_idx++)
  {
    int cc = m_listC[c_idx];
    if (m_C[cc] != xC[cc]) // add its edge
    {
      int cc_SrcVertex = m_componentEdgeSrc[cc];
      int cc_DstVertex = nextEdge[cc_SrcVertex];
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

void parallelBoruvka_t::updateComponents(std::vector<int> &xC,
                                         std::vector<int> &nextEdge,
                                         std::vector<double> &nextEdgeLen)
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
      int cc_DstVertex = nextEdge[cc_SrcVertex];
      int cc_next = m_C[cc_DstVertex];

      // check next components
      // if(xC[cc] != xC[cc_next])
      // {
      //     int min_cc = std::min(xC[cc], xC[cc_next]);
      //     xC[cc] = min_cc;
      //     xC[cc_next] = min_cc;
      //     numChanges++;
      // }

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
  updateMST(xC, nextEdge);

  // Update component Labels
  for (int pt = 0; pt < m_npts; pt++)
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

void parallelBoruvka_t::computeCandidateEdges(std::vector<int> &nextEdge,
                                              std::vector<double> &nextEdgeLen)
{
  // find potential new edges
  for (int pt = 0; pt < m_npts; pt++)
    if (m_C[nextEdge[pt]] == m_C[pt])
    {
      auto r = findClosestPointWithDifferentLabel(m_points, m_C, pt);
      nextEdge[pt] = r.first;
      nextEdgeLen[pt] = r.second;
    }

  for (int cc = 0; cc < m_numComponents; cc++)
  {
    m_componentEdgeLen[m_listC[cc]] = INFTY;
  }

  // for each component find the edge with minimum edge len
  for (int pt = 0; pt < m_npts; pt++)
  {
    if (nextEdgeLen[pt] < m_componentEdgeLen[m_C[pt]])
    {
      m_componentEdgeLen[m_C[pt]] = nextEdgeLen[pt];
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
    wtEdge_t wte(edge,
                 distanceSquared(m_points[edge.first], m_points[edge.second]));
    wtmst.push_back(wte);
  }
  return wtmst;
}
