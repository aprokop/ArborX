#pragma once
#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>

typedef std::pair<int, int> edge_t;
typedef std::pair<edge_t, double> wtEdge_t;

using Point = std::vector<double>;

const double INFTY = std::numeric_limits<double>::infinity();

class parallelBoruvka_t
{
  int m_npts;
  int m_numComponents;

  // input
  const std::vector<Point> &m_points; // list of points

  // output
  std::vector<edge_t> m_MST;

  // vertex-wise variables
  std::vector<int> m_C; // my component
  std::vector<int> m_nextEdge;
  std::vector<double> m_nextEdgeLen; // for each vertex

  // component wise variables
  std::vector<int> m_xC;    // next component
  std::vector<int> m_listC; // list of components

  // std::vector<int> m_edgeIdx;          // next component
  std::vector<int>
      m_pfxsum; // used for computing next address, length should be n_pts+1
  std::vector<double>
      m_componentEdgeLen; // edgeLength of candidate edge of the component
  std::vector<int>
      m_componentEdgeSrc; // starting vertex of component's candidate edge

  std::vector<edge_t> m_parent; // parent Edge

public:
  parallelBoruvka_t(const std::vector<std::vector<double>> &points);
  int numComponents() { return m_numComponents; }
  // void updateComponent(std::vector<edge_t> &candidateEdges);
  void computeCandidateEdges();
  void updateComponents();
  void computeNextNeighbour(
      int pt); // finds nearest point to pt, not in this component
  void updateMST();

  void writeMST(std::ofstream &ofileName);
  std::vector<wtEdge_t> weightedMST();
};

