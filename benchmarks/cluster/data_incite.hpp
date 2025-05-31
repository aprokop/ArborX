/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_BENCHMARK_DATA_INCITE_HPP
#define ARBORX_BENCHMARK_DATA_INCITE_HPP

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "closest_factors.hpp"
#include <fcntl.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct DataFiles
{
  std::string rawdatadir;
  std::string var;
  std::string delimiter;
  std::string rawdatasnap;
};

// Memory-mapped approach (file must fit in RAM, no true mmap)
template <typename MemorySpace>
inline auto get_data_mmap(MPI_Comm comm, DataFiles const &data_files, int nx,
                          int ny, int nz)
{
  auto filename = data_files.rawdatadir + "/" + data_files.var +
                  data_files.delimiter + data_files.rawdatasnap;

  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Could not open file: \"" + filename + "\".");

  // Get file size
  struct stat sb;
  if (fstat(fd, &sb) == -1)
  {
    close(fd);
    throw std::runtime_error("Could not get file size for: \"" + filename +
                             "\".");
  }
  size_t filesize = sb.st_size;

  // Memory map the file
  void *map = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED)
  {
    close(fd);
    throw std::runtime_error("Could not memory-map file: \"" + filename +
                             "\".");
  }

  float *map_float = reinterpret_cast<float *>(map);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  constexpr int DIM = 3;
  auto factors = closestFactors<3>(comm_size);
  if (ny < nz)
    std::swap(factors[1], factors[2]);

  struct Range
  {
    int start;
    int end;
  };

  std::vector<int> n = {nx, ny, nz};
  std::vector<Range> ranges(3);
  int local_n = 1;
  for (int d = 0, s = comm_rank; d < 3; ++d)
  {
    auto I = s % factors[d];
    s /= factors[d];

    int offset = (d == 0);

    auto points_per_rank = n[d] / factors[d];
    ranges[d].start = (d == 0) + I * points_per_rank;
    if (I < factors[d] - 1)
      ranges[d].end = ranges[d].start + points_per_rank;
    else
      ranges[d].end = (d == 0) + n[d];
    local_n *= (ranges[d].end - ranges[d].start);
  }
  // printf("[%d]: [%d, %d) x [%d, %d) x [%d, %d)\n", comm_rank,
  // ranges[0].start, ranges[0].end, ranges[1].start, ranges[1].end,
  // ranges[2].start, ranges[2].end);

  Kokkos::View<ArborX::Point<1> *, MemorySpace> data(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Benchmark::sst-data"),
      local_n);
  auto data_host = Kokkos::create_mirror_view(data);

  // The data always pads X-dimension with two artifical layers (one on each
  // side). Ignore those.
  auto real_nx = nx + 2;
  int index = 0;
  for (int k = ranges[2].start; k < ranges[2].end; ++k)
    for (int j = ranges[1].start; j < ranges[1].end; ++j)
      for (int i = ranges[0].start; i < ranges[0].end; ++i)
        data_host[index++] = {map_float[k * (real_nx * ny) + j * real_nx + i]};
  Kokkos::deep_copy(data, data_host);

  if (munmap(map, filesize) == -1)
    perror("Error un-mmapping the file");
  close(fd);

  return data;
}

#endif
