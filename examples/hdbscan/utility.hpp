#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "point.hpp"

double distance(Point const &p1, Point const &p2);

bool compare_edges_less(std::tuple<int, int, double> const &edge1,
                        std::tuple<int, int, double> const &edge2);

int kNNWithFilter(std::vector<Point> const &points,
                  std::vector<int> const &components, int index);
