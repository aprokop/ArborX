#include "parallel_boruvka.hpp"

#include <iostream>
#include <numeric> // iota

#include "utility.hpp"

void updateCandidateEdges(std::vector<Point> const &points,
                          std::vector<int> const &labels,
                          std::vector<int> &next_edge,
                          std::vector<double> &next_edge_len)
{
  auto const n = points.size();

  for (int i = 0; i < n; i++)
  {
    bool const update_required = (labels[next_edge[i]] == labels[i]);
    if (update_required)
    {
      auto r = findClosestPointWithDifferentLabel(points, labels, i);
      next_edge[i] = r.first;
      next_edge_len[i] = r.second;
    }
  }
}

void determineComponentEdges(std::vector<int> const &labels,
                             std::vector<int> const &next_edge,
                             std::vector<double> const &next_edge_len,
                             std::vector<int> &component_edge_src)
{
  auto const n = labels.size();

  const double infty = std::numeric_limits<double>::infinity();

  // for each component find the edge with minimum edge len
  std::vector<double> component_edge_len(n, infty);
  for (int i = 0; i < n; i++)
  {
    int const label = labels[i];
    if (next_edge_len[i] < component_edge_len[label])
    {
      component_edge_len[label] = next_edge_len[i];
      component_edge_src[label] = i;
    }
  }
}

void parallelBoruvka_t::updateMST(std::vector<int> const &labels,
                                  std::vector<int> const &xC,
                                  std::vector<int> const &next_edge,
                                  std::vector<int> const &component_edge_src,
                                  std::vector<int> &listC,
                                  std::vector<int> const &offsets)
{
  auto const n = m_points.size();
  auto const num_components = listC.size();

  int numEdgesMST = n - num_components;
  for (int c_idx = 0; c_idx < num_components; c_idx++)
  {
    int cc = listC[c_idx];
    if (labels[cc] != xC[cc]) // add its edge
    {
      int cc_SrcVertex = component_edge_src[cc];
      int cc_DstVertex = next_edge[cc_SrcVertex];
      m_MST[numEdgesMST + offsets[c_idx]] = m_parent[cc];
      std::cout << "Adding (" << m_parent[cc].first << " "
                << m_parent[cc].second << ") at "
                << numEdgesMST + offsets[c_idx] << "\n";
    }
    else
    {
      listC[c_idx - offsets[c_idx]] = cc;
    }
  }
}

void parallelBoruvka_t::updateComponents(
    std::vector<int> &labels, std::vector<int> &xC,
    std::vector<int> const &next_edge,
    std::vector<int> const &component_edge_src, std::vector<int> &listC)
{
  // parallel label propagation
  bool is_updated;
  do
  {
    is_updated = false;
    for (int src_label : listC)
    {
      int src = component_edge_src[src_label];
      int dst = next_edge[src];
      int dst_label = labels[dst];

      if (xC[src_label] > xC[dst_label])
      {
        xC[src_label] = xC[dst_label];
        m_parent[src_label] = std::make_pair(src, dst);
        is_updated = true;
      }
      else if (xC[src_label] < xC[dst_label])
      {
        xC[dst_label] = xC[src_label];
        m_parent[dst_label] = std::make_pair(src, dst);
        is_updated = true;
      }
    }
  } while (is_updated);

  int num_components = listC.size();

  // exclusive prefix sum
  std::vector<int> offsets(num_components + 1);
  for (int k = 0; k < num_components; ++k)
  {
    int label = listC[k];
    offsets[k + 1] = offsets[k] + (labels[label] == xC[label] ? 0 : 1);
  }

  // last element
  int num_removed_components = offsets.back();

  // Update MST
  updateMST(labels, xC, next_edge, component_edge_src, listC, offsets);

  // Update component Labels
  auto const n = m_points.size();
  for (int i = 0; i < n; i++)
    labels[i] = xC[labels[i]];

  // update number of components
  listC.resize(num_components - num_removed_components);
}

parallelBoruvka_t::parallelBoruvka_t(const std::vector<Point> &points)
    : m_points(points)
{
  auto const n = points.size();

  // initialization
  m_MST.resize(n - 1);

  m_parent.resize(n);

  std::vector<int> xC(n); // next component
  std::vector<int> component_edge_src(n);
  std::vector<int> next_edge(n);
  std::vector<double> next_edge_len(n);
  std::vector<int> labels(n);
  std::vector<int> listC(n); // list of components

  std::iota(xC.begin(), xC.end(), 0);
  std::iota(component_edge_src.begin(), component_edge_src.end(), 0);
  std::iota(next_edge.begin(), next_edge.end(), 0);
  std::iota(labels.begin(), labels.end(), 0);
  std::iota(listC.begin(), listC.end(), 0);

  while (listC.size() > 1)
  {
    updateCandidateEdges(m_points, labels, next_edge, next_edge_len);
    determineComponentEdges(labels, next_edge, next_edge_len,
                            component_edge_src);
    updateComponents(labels, xC, next_edge, component_edge_src, listC);

    std::cout << "Number of components is " << listC.size() << "\n";
    for (int label : listC)
      std::cout << label << " ";
    std::cout << "\n";
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
