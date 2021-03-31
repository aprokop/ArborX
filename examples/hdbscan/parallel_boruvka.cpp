#include "parallel_boruvka.hpp"

#include <iostream>
#include <limits>
#include <numeric> // iota

#include "utility.hpp"

void determineComponentEdges(
    std::vector<Point> const &points, std::vector<int> const &labels,
    std::vector<int> &cached_closest_neighbors,
    std::vector<std::pair<int, int>> &component_out_edges)
{
  auto const n = points.size();

  // Update closest neighbors if necessary
  for (int i = 0; i < n; i++)
  {
    bool const update_required =
        (labels[cached_closest_neighbors[i]] == labels[i]);
    if (update_required)
      cached_closest_neighbors[i] = kNNWithFilter(points, labels, i);
  }

  const double infty = std::numeric_limits<double>::infinity();

  // Find outgoing edge for each component
  std::vector<double> closest_component_dist(n, infty);
  for (int i = 0; i < n; i++)
  {
    int const component = labels[i];
    auto &component_out_edge = component_out_edges[component];

    auto dist = distance(points[i], points[cached_closest_neighbors[i]]);
    if (compare_edges_less(
            std::make_tuple(i, cached_closest_neighbors[i], dist),
            std::make_tuple(component_out_edge.first, component_out_edge.second,
                            closest_component_dist[component])))
    {
      component_out_edge = std::make_pair(i, cached_closest_neighbors[i]);
      closest_component_dist[component] = dist;
    }
  }
}

void computeComponentsRemapping(
    std::vector<std::pair<int, int>> const &component_out_edges,
    std::vector<int> &components, std::vector<int> &labels,
    std::vector<int> &components_remapping)
{
  int const n = labels.size();

  auto compute_next = [&component_out_edges, &labels](int component) {
    int next_component = labels[component_out_edges[component].second];
    int next_next_component =
        labels[component_out_edges[next_component].second];

    if (next_next_component != component)
    {
      // The component's edge is unidirectional
      return next_component;
    }
    // The component's edge is bidirectional, resolve uniquely through min
    return std::min(component, next_component);
  };

  for (auto component : components)
  {
    int next_component = compute_next(component);

    if (next_component == component)
    {
      // The edge is bidirectional. To not count it twice, we don't put it into
      // MST in this situation.
      components_remapping[component] = component;
      continue;
    }

    int prev_component;
    do
    {
      prev_component = next_component;
      next_component = compute_next(prev_component);
    } while (next_component != prev_component);

    components_remapping[component] = next_component;
  }
}

void updateMST(int n,
               std::vector<std::pair<int, int>> const &component_out_edges,
               std::vector<int> const &components,
               std::vector<int> const &components_remapping,
               std::vector<edge_t> &mst)
{
  int const num_components = components.size();

  int num_edges = n - num_components;
  for (auto component : components)
    if (component != components_remapping[component])
      mst[num_edges++] = component_out_edges[component];
}

void updateComponentsAndLabels(std::vector<int> const &components_remapping,
                               std::vector<int> &components,
                               std::vector<int> &labels)
{
  // parallel_scan
  int offset = 0;
  for (auto component : components)
    if (component == components_remapping[component])
      components[offset++] = component;
  components.resize(offset);

  // Update component Labels
  int const n = labels.size();
  for (int i = 0; i < n; i++)
    labels[i] = components_remapping[labels[i]];
}

parallelBoruvka_t::parallelBoruvka_t(const std::vector<Point> &points)
    : _points(points)
{
  auto const n = points.size();

  // initialization
  _mst.resize(n - 1);

  std::vector<std::pair<int, int>> component_out_edges(n);
  std::vector<int> cached_closest_neighbors(n);
  std::vector<int> labels(n);
  std::vector<int> components(n);
  std::vector<int> components_remapping(n);

  std::iota(components.begin(), components.end(), 0);
  labels = components;
  cached_closest_neighbors = labels;

  while (components.size() > 1)
  {
    determineComponentEdges(_points, labels, cached_closest_neighbors,
                            component_out_edges);
    for (auto component : components)
    {
      auto edge_out = component_out_edges[component];
      printf("%d -> %d [%d, %d]\n", component, labels[edge_out.second],
             edge_out.first, edge_out.second);
    }
    std::cout << std::endl;

    computeComponentsRemapping(component_out_edges, components, labels,
                               components_remapping);
    updateMST(n, component_out_edges, components, components_remapping, _mst);

    updateComponentsAndLabels(components_remapping, components, labels);

    std::cout << "Components (#" << components.size() << "): [ ";
    for (int component : components)
      std::cout << component << " ";
    std::cout << "]\n";
  }
}

void parallelBoruvka_t::writeMST(std::ofstream &outfile)
{
  for (auto const &edge : _mst)
    outfile << edge.first << " " << edge.second << "\n";
}

std::vector<wtEdge_t> parallelBoruvka_t::weightedMST()
{
  std::vector<wtEdge_t> wtmst;
  for (auto edge : _mst)
  {
    wtEdge_t wte(edge, distance(_points[edge.first], _points[edge.second]));
    wtmst.push_back(wte);
  }
  return wtmst;
}
