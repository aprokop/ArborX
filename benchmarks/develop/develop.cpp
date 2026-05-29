#include <vector>

#include <mpi.h>
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int outdegree;
  std::vector<int> destinations;
  std::vector<int> weights;
  // clang-format off
  switch(comm_rank) {
    case 0: outdegree = 2; destinations = {9, 8}; weights = {1, 1}; break;
    case 1: outdegree = 2; destinations = {8, 7}; weights = {1, 1}; break;
    case 2: outdegree = 2; destinations = {7, 6}; weights = {1, 1}; break;
    case 3: outdegree = 2; destinations = {6, 5}; weights = {1, 1}; break;
    case 4: outdegree = 2; destinations = {5, 4}; weights = {1, 1}; break;
    case 5: outdegree = 2; destinations = {4, 3}; weights = {1, 1}; break;
    case 6: outdegree = 2; destinations = {3, 2}; weights = {1, 1}; break;
    case 7: outdegree = 2; destinations = {2, 1}; weights = {1, 1}; break;
    case 8: outdegree = 2; destinations = {1, 0}; weights = {1, 1}; break;
    case 9: outdegree = 1; destinations = {0}; weights = {1}; break;
  }
  // clang-format on
  constexpr int reorder = 0;
  for (int it = 0; it < 10; ++it)
  {
    MPI_Comm graph_comm;
    MPI_Dist_graph_create(comm, 1, &comm_rank, &outdegree, destinations.data(),
                          weights.data(), MPI_INFO_NULL, reorder, &graph_comm);
    MPI_Comm_free(&graph_comm);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
