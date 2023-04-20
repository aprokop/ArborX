#include <iostream>
#include <cstdlib>
#include <random>
using namespace std;

int main(int argc, char** argv) {

	int nPts = atoi(argv[1]);
	int	nDims =atoi(argv[2]) ;
	unsigned seed = 100;
	default_random_engine generator (seed);
	normal_distribution<float> distribution (0.0,1.0); 

	FILE *fp;
	fp = fopen( argv[3] , "w" );
	fwrite(&nPts , 1 , sizeof(int) , fp );
	fwrite(&nDims , 1 , sizeof(int) , fp );

	for(int i=0; i<nPts*nDims; i++)
	{
		float ftus = distribution(generator);
		fwrite(&ftus , 1 , sizeof(float) , fp );	
	}

	fclose(fp);

	return 0;
}
