#include <iostream>
#include <cstdlib>
#include <random>
using namespace std;

int main(int argc, char** argv) {

	int nPts = atoi(argv[1]);
	int	nDims =atoi(argv[2]) ;
	unsigned seed = 100;
	srand (100);

	FILE *fp;
	fp = fopen( argv[3] , "w" );
	fwrite(&nPts , 1 , sizeof(int) , fp );
	fwrite(&nDims , 1 , sizeof(int) , fp );

	for(int i=0; i<nPts*nDims; i++)
	{
		float ftus = 1.0*rand()/(1.0*RAND_MAX);
		fwrite(&ftus , 1 , sizeof(float) , fp );	
	}

	fclose(fp);

	return 0;
}
