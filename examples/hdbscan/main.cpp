#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// #include "hdbscan.hpp"
#include "parallel_boruvka.hpp"

std::vector<std::vector<double>> readFile(std::ifstream &inFile)
{
  char c;
  int npts;
  int dim;
  inFile >> c;
  inFile >> npts;
  inFile >> dim;
  std::cout << c << " number of pts is " << npts << " and dimemsion is " << dim
            << "\n";
  std::vector<std::vector<double>> out(npts, std::vector<double>(dim, 0.0));
  for (int i = 0; i < npts; i++)
    for (int j = 0; j < dim; j++)
      inFile >> out[i][j];

  return out;
}

int main(int argc, char *argv[])
{

  // read the points
  if (argc < 2)
  {
    std::cout << " usage: " << argv[0] << " inputFile\n";
    return -1;
  }
  std::cout << "Opening " << argv[1] << "\n";
  std::ifstream infile(argv[1]);

  std::vector<std::vector<double>> points = readFile(infile);
  int kpts = 3;
  // hdbscan_t hdbscan(points,kpts);
  parallelBoruvka_t pB(points);
  // writing MST back to file
  std::string outFileName(argv[1]);
  outFileName = outFileName + ".mst";
  std::cout << "writing to: " << outFileName << "\n";

  std::ofstream outFile(outFileName);
  pB.writeMST(outFile);

  // perform hdbscan

  // return output
  return 0;
}
