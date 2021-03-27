#pragma once

#include <utility>
#include <vector>

#include "point.hpp"

double distanceSquared(Point const &p1, Point const &p2);

std::pair<int, double> kNNWithFilter(std::vector<Point> const &points,
                                     std::vector<int> const &components,
                                     int index);
