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

#ifndef ARBORX_GENERICIO_DATA_HPP
#define ARBORX_GENERICIO_DATA_HPP

#include <ArborX_Config.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "GenericIO.h"

#ifdef ARBORX_ENABLE_MPI
#include <mpi.h>
#else
#define GENERICIO_NO_MPI
#endif

class GenericIO : public gio::GenericIO
{
public:
  GenericIO(std::string const &filename,
            gio::GenericIO::FileIO method = gio::GenericIO::FileIOPOSIX,
            gio::GenericIO::MismatchBehavior mismatch_behavior =
                gio::GenericIO::MismatchRedistribute,
            int eff_rank = -1)
#ifdef GENERICIO_NO_MPI
      : gio::GenericIO(filename, method)
#else
      : gio::GenericIO(MPI_COMM_WORLD, filename, method)
#endif
  {
    openAndReadHeader(mismatch_behavior, eff_rank);
    getVariableInfo(_variables);
  }

  void inspect()
  {
    int rank;
#ifdef GENERICIO_NO_MPI
    rank = 0;
#else
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (rank == 0)
    {
      std::stringstream ss;
      ss << "#elements (local): " << readNumElems() << "\n";
      ss << "#elements (global): " << readTotalNumElems() << "\n";
      ss << "variable name [data type](i=int,f=float,#bits)\n";
      ss << "---------------------------------------------\n";
      for (auto const &var_info : _variables)
      {
        ss << var_info.Name << " [" << (var_info.IsFloat ? "f" : "i")
           << var_info.ElementSize * 8;
        int num_elementsents = var_info.Size / var_info.ElementSize;
        if (num_elementsents > 1)
          ss << "x" << num_elementsents;
        ss << "]\n";
      }
      std::cout << ss.str() << std::endl;
    }
  }

  auto readCoordinates(std::string const &var_name, bool print_stats = true,
                       bool collective_stats = true, int eff_rank = -1)
  {
    int rank;
#ifdef GENERICIO_NO_MPI
    rank = 0;
#else
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    clearVariables();

    auto varp =
        std::find_if(_variables.begin(), _variables.end(),
                     [&var_name](auto const &v) { return v.Name == var_name; });
    if (varp == _variables.end())
      throw std::invalid_argument(
          std::string("requested variable is not defined in GenericIO file: ") +
          var_name);

    auto &var_info = *varp;

    // read number of elements
    int64_t num_elements = readNumElems(eff_rank);

    // extra space
    auto readsize = num_elements + requestedExtraSpace() / var_info.ElementSize;

    KOKKOS_ASSERT(var_info.IsFloat && var_info.ElementSize == 8);

    std::vector<double> v = {readsize, 8};
    addVariable(*varp, v, gio::GenericIO::VarHasExtraSpace);
    readData(eff_rank, print_stats, collective_stats);

    clearVariables();

#ifndef GENERICIO_NO_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return v;
  }

  // returns a list of 3 floats describing the box origin
  auto read_phys_origin()
  {
    std::array<double, 3> origin;
    readPhysOrigin(origin.data());
    return origin;
  }

  // returns a list of 3 floats describing the box size
  auto read_phys_scale()
  {
    std::array<double, 3> scale;
    readPhysScale(scale.data());
    return scale;
  }

  auto get_source_ranks()
  {
    std::vector<int> src_ranks;
    getSourceRanks(src_ranks);
    return src_ranks;
  }

  auto read_dims()
  {
    std::array<int, 3> sd;
    readDims(sd.data());
    return sd;
  }

  auto read_coords(int eff_rank = -1)
  {
    std::array<int, 3> sc;
    readCoords(sc.data(), eff_rank);
    return sc;
  }

private:
  std::vector<gio::GenericIO::VariableInfo> _variables;
};

#endif
