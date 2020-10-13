---
title: 'ArborX: A performance portble geometric search library'
tags:
  - C++
  - geometric search
  - kokkos
  - CUDA
  - milky way
authors:
  - name: Damien Lebrun-Grandi\'e
    orcid: 0000-0003-1952-7219
    affiliation: 1
  - name: Andrey Prokopenko
    orcid: 0000-0003-3616-5504
    affiliation: 1
  - name: Daniel Arndt
    orcid: 0000-0001-8773-490
    affiliation: 1
  - name: Bruno Turcksin
    orcid: 0000-0001-5954-6313
    affiliation: 1
affiliations:
 - name: Oak Ridge National Laboratory
   index: 1
date: 10 November 2020
bibliography: paper.bib

---

# Summary

ArborX is a C++ library for searching close geometric objects in space. ArborX
was designed to be run on multiple hardware architectures using a single
interface definition. with a focus on performance portability for both current
and known future leadership-class supercomputers.

## Parallelism

ArborX uses MPI+X strategy for its parallelism. For on-node parallelism, ArborX
relies on the Kokkos `[@edwards2014]` library for performance portability.
Kokkos is a C++ library providing a uniform programming interface for various
backends, e.g., OpenMP, CUDA, HIP, SYCL. Using Kokkos allows for running the
same code on CPUs or GPUs by simply changing the backend through a template
argument.

## Documentation


# Acknowledgements

The development of ArborX was supported by the Exascale Computing Project
(17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office
of Science and the National Nuclear Security Administration, and sponsored by
the Laboratory Directed Research and Development Program of Oak Ridge National
Laboratory, managed by UT-Battelle, LLC, for the U.S. Department of Energy.

# References
