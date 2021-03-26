#include "hdbscan.hpp"

#include <limits>
#include <vector>

double INFTY = std::numeric_limits<double>::infinity();

void parBoruvka_t::updateComponent(std::vector<edge_t> &candidateEdges)
{
  int numChanges = 1;
  for (edge_t edge : candidateEdges)
  {
    m_edgeIdx[m_C[edge.first]] = -1;
    m_edgeIdx[m_C[edge.second]] = -1;
  }

  while (numChanges > 0)
  {
    for (int i = 0; i < m_numComponents; i++)
    {
      edge_t edge = candidateEdges[i];
      int v1 = edge.first;
      int v2 = edge.second;
      int c1 = m_C[v1];
      int c2 = m_C[v2];
      // m_xC contains labels
      if (m_xC[c1] > m_xC[c2])
      {
        m_xC[c1] = m_xC[c2];
        m_edgeIdx[c1] = i;
        numChanges++;
      }
      else if (m_xC[c1] < m_xC[c2])
      {
        m_xC[c2] = m_xC[c1];
        m_edgeIdx[c2] = i;
        numChanges++;
      }
    } /*for all candidate edges*/
  }   /*while(numChanges>0)*/

  // relabel vertices
  for (int ptIdx = 0; ptIdx < m_npts; ptIdx++)
  {
    m_C[ptIdx] = m_xC[m_C[ptIdx]];
  }

  // Add edges to MST and update
  for (int C_idx = 0; C_idx < m_numComponents; C_idx++)
  {
    int c0 = m_ListOfComponents[C_idx];
    if (m_edgeIdx[c0] != -1)
    {
      m_MST[m_npts - m_numComponents] = candidateEdges[m_edgeIdx[c0]];
      m_numComponents--;
    }
  }

  // update list of components
  int Cidx = 0;
  for (int ptIdx = 0; ptIdx < m_npts; ptIdx++)
  {
    if (m_C[ptIdx] == ptIdx)
      m_ListOfComponents[Cidx++] = ptIdx;
  }
}

double hdbscan_t::distSqEuclidean(int v1, int v2)
{
  double out = 0;
  for (int i = 0; i < m_dim; i++)
  {
    double diff = (m_points[v1][i] - m_points[v2][i]);
    out += diff * diff;
  }
  return out;
}

int hdbscan_t::computeNextNeighbour(int v)
{
  nextEdgeWt[v] = INFTY;
  // linear search
  for (int i = 0; i < npts; i++)
  {
    if (i != v && Components[i] != Components[v] &&
        distMreachSq(v, i) < nextEdgeWt[v])
    {
      nextEdgeWt[v] = distMreachSq(v, i);
      nextEdge[v] = i;
    }
  }
  return 0;
}

std::vector<wtEdge_t> hdbscan_t::computeMST()
{
  while (m_component.numComponents() > 1)
  {
    std::vector<edge_t> newEdges = computeCandidateEdges();
    m_component.updateComponent(newEdges);
  }
}

// std::vector<std::vector<int> > hdbscan_t::computeClusters();

// hdbscan_t::hdbscan_t(std::vector<std::vector<double> >& pts, int k_pts):
//          points(pts), kpts(k_pts)
// {
//     npts = points.size();
//     dim = points[0].size();
//     //
//     for(int i=0; i<npts; i++)
//     {
//         nextEdge.push_back(-1);
//         nextEdgeWt.push_back(INFTY);
//     }

// }
