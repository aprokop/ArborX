#include "parallel_boruvka.hpp"

#include <iostream>
#include <numeric> // iota

#include "utility.hpp"

void updateCandidateEdges(std::vector<Point> const &points,
                          std::vector<int> const &components,
                          std::vector<int> &closest_neighbor,
                          std::vector<double> &closest_neighbor_dist)
{
  auto const n = points.size();

  for (int i = 0; i < n; i++)
  {
    bool const update_required =
        (components[closest_neighbor[i]] == components[i]);
    if (update_required)
    {
      auto r = kNNWithFilter(points, components, i);

      closest_neighbor[i] = r.first;
      closest_neighbor_dist[i] = r.second;
    }
  }
}

void determineComponentEdges(std::vector<int> const &components,
                             std::vector<int> const &closest_neighbor,
                             std::vector<double> const &closest_neighbor_dist,
                             std::vector<int> &component_origin)
{
  auto const n = components.size();

  const double infty = std::numeric_limits<double>::infinity();

  // for each component find the edge with minimum edge len
  std::vector<double> closest_component_dist(n, infty);
  for (int i = 0; i < n; i++)
  {
    int const component = components[i];
    if (closest_neighbor_dist[i] < closest_component_dist[component])
    {
      component_origin[component] = i;
      closest_component_dist[component] = closest_neighbor_dist[i];
    }
  }
}

void parallelBoruvka_t::updateMST(std::vector<int> const &components,
                                  std::vector<int> const &xC,
                                  std::vector<int> const &closest_neighbor,
                                  std::vector<int> const &component_origin,
                                  std::vector<int> &listC,
                                  std::vector<int> const &offsets)
{
  auto const n = m_points.size();
  auto const num_components = listC.size();

  int numEdgesMST = n - num_components;
  for (int c_idx = 0; c_idx < num_components; c_idx++)
  {
    int cc = listC[c_idx];
    if (components[cc] != xC[cc]) // add its edge
    {
      int cc_SrcVertex = component_origin[cc];
      int cc_DstVertex = closest_neighbor[cc_SrcVertex];
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
    std::vector<int> &components, std::vector<int> &xC,
    std::vector<int> const &closest_neighbor,
    std::vector<int> const &component_origin, std::vector<int> &listC)
{
  // parallel component propagation
  bool is_updated;
  do
  {
    is_updated = false;
    for (int src_component : listC)
    {
      // FIXME: ???
      int src = component_origin[src_component];
      int dst = closest_neighbor[src];
      int dst_component = components[dst];

      if (xC[src_component] > xC[dst_component])
      {
        xC[src_component] = xC[dst_component];
        m_parent[src_component] = std::make_pair(src, dst);
        is_updated = true;
      }
      else if (xC[src_component] < xC[dst_component])
      {
        xC[dst_component] = xC[src_component];
        m_parent[dst_component] = std::make_pair(src, dst);
        is_updated = true;
      }
    }
  } while (is_updated);

  int num_components = listC.size();

  // exclusive prefix sum
  std::vector<int> offsets(num_components + 1);
  for (int k = 0; k < num_components; ++k)
  {
    int component = listC[k];
    offsets[k + 1] =
        offsets[k] + (components[component] == xC[component] ? 0 : 1);
  }

  // last element
  int num_removed_components = offsets.back();

  // Update MST
  updateMST(components, xC, closest_neighbor, component_origin, listC, offsets);

  // Update component Labels
  auto const n = m_points.size();
  for (int i = 0; i < n; i++)
    components[i] = xC[components[i]];

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
  std::vector<int> component_origin(n);
  std::vector<int> closest_neighbor(n);
  std::vector<double> closest_neighbor_dist(n);
  std::vector<int> components(n);
  std::vector<int> listC(n); // list of components

  std::iota(xC.begin(), xC.end(), 0);
  std::iota(component_origin.begin(), component_origin.end(), 0);
  std::iota(closest_neighbor.begin(), closest_neighbor.end(), 0);
  std::iota(components.begin(), components.end(), 0);
  std::iota(listC.begin(), listC.end(), 0);

  while (listC.size() > 1)
  {
    updateCandidateEdges(m_points, components, closest_neighbor,
                         closest_neighbor_dist);
    determineComponentEdges(components, closest_neighbor, closest_neighbor_dist,
                            component_origin);
    updateComponents(components, xC, closest_neighbor, component_origin, listC);

    std::cout << "Number of components is " << listC.size() << "\n";
    for (int component : listC)
      std::cout << component << " ";
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
