#pragma once

#include <utility>
#include <vector>

#include "point.hpp"

double distanceSquared(Point const &p1, Point const &p2);

std::pair<int, double>
findClosestPointWithDifferentLabel(std::vector<Point> const &points,
                                   std::vector<int> const &labels, int index);

void prefixSumExclusive(int n, int *A);
