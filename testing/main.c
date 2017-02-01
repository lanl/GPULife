/*  LA-CC-16080
    Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   	Priscilla Kelly <priscilla.noreen@gmail.com>
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

#define ndims 2
#define squareFactor 2 // sqrt of num of proc
#define grid 80

// Function Declarations
int **alloc2D(int row, int col);
void reorderHalo(int s,int len,int *halo);
void dealloc(int **array);
void initialize(int myrank,int row, int col, int **subMat);

// Outside function declarations
#ifdef __NVCC__
	void call_loadTile_CUDA(int flag, int elements, int *Matrix,int **pointer2device);

	void call_cuda_applyRules(int flag,int rows, int cols, int *halo,int *halo_dev, int *update, int *hold);
#endif

#ifdef _OPENACC
	void call_OpenACC_applyRules(int flag,int rows, int cols, int *halo,int *halo_dev, int *update, int *hold);
#endif

#ifdef _OPENMP
	void call_OpenMP4_applyRules(int flag,int rows, int cols, int *halo,int *halo_dev, int *update, int *hold);
#endif

//MAIN
int main(int argc, char *argv[]) {
	printf("Initiating Game of Life...\n");
	clock_t start = clock(), diff;

	printf("%s testing, line %d\n", __FILE__,__LINE__);
	#ifdef HAVE_MPI
	printf("%s testing, line %d\n", __FILE__,__LINE__);
	/***************************************/
	/* Initialize MPI and check dimensions */
	/***************************************/
    int myrank, size;
    int myColRank, myRowRank;
	int dims[ndims], worldCord[ndims];
	int rowCord[ndims], colCord[ndims];
	int wrap_around[ndims], freeCord[ndims];
	int reorder, ierr, errs;
	int nRows, nCols;
	int length; 
	char name[250];
	
	// Initialize communicators
	MPI_Comm commRow;
	MPI_Comm commCol;
	MPI_Comm commWorld; // comm between everyone

    // Set up the MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Get_processor_name(name, &length);

	// Split up the procs based on a 2D grid
	if (size%(int)squareFactor != 0) {
		MPI_Finalize();
		exit(0);
	}
	else {
		int d = size/(int)squareFactor;
		// check if it's square, if not, exit
		if (d*d != size) {
			MPI_Finalize();
			exit(0);
		}
	}
	// Leaving up, initial set up is for square
	nRows = size/(int)squareFactor;
	nCols = size/(int)squareFactor;
	dims[0] = nRows; // rows
	dims[1] = nCols; // cols
	
	MPI_Dims_create(size, ndims, dims);
	
	wrap_around[0] = wrap_around[1] = 0; // periodic shift is true 
	reorder = 1;
	ierr = 0;
	// world comunications
	ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, 
						  wrap_around, reorder, &commWorld);
	MPI_Cart_coords(commWorld, myrank, ndims, worldCord);

	// row communications
	freeCord[0] = 0;
	freeCord[1] = 1;
	MPI_Cart_sub(commWorld, freeCord, &commRow);
    MPI_Comm_rank(commRow, &myRowRank);

	// col communications	
	freeCord[0] = 1;
	freeCord[1] = 0;
	MPI_Cart_sub(commWorld, freeCord, &commCol);
    MPI_Comm_rank(commCol, &myColRank);
 	#endif 	
	/***************************************/
	/*       Set up Game Topology          */
	/***************************************/
	printf("%s testing, line %d\n", __FILE__,__LINE__);
	// move to in line argument sometime
	int myRows = grid+2;
	int myCols = grid+2;
	int matSize = myRows*myCols;

	// Create my subMatrix to send to GPU
	// This will have the extra padding of the halo
	int i, j, c;
	int **subMatrix;
	subMatrix = alloc2D(myRows,myCols);
	initialize(myrank,myRows, myCols,subMatrix);
	int *sub1D = (int *)&(subMatrix[0][0]);

	int *sub1_dev = NULL;
	int *sub2_dev = (int *)malloc(sizeof(int)*matSize);
	int *halo_dev = NULL;
	int *update = NULL;
	int *hold = NULL;

	// load the buffers and inital data
	#ifdef __CUDACC__
		printf("Found CUDA!\nSending data to GPU")
		call_loadTile_CUDA(0,matSize,sub1D,&sub1_dev);
		call_loadTile_CUDA(0,matSize,sub1D,&sub2_dev);
	#endif

	#ifdef _OPENMP
		printf("Found OpenMP!\nSending data to GPU")
		sub1_dev = sub1D;
		memcpy(sub2_dev,sub1D,sizeof(int)*matSize);
		#pragma omp target map(to:sub1_dev[0:matSize],sub2_dev[0:matSize])
	#endif

	// set up haloIn and haloOut so that it can be given 
	// to the package even if the halo isn't updated
	int haloSize = 2*(myRows+myCols);
	int haloIn[haloSize];
	int haloOut[haloSize];

	#ifdef HAVE_MPI
		// Halo
		int n,s,w,e; // starting indices
		n = 0;
		e = myCols;
		s = e+myRows;
		w = s+myCols;

		// set halo to 0 initally	
		for (i=0;i<haloSize;i++){
			haloIn[i] = 0;
			haloOut[i] = 0;
		}
		// get halo location on the gpu
		#ifdef __CUDACC__
			call_loadTile_CUDA(0,haloSize,haloOut,&halo_dev);
		#endif

		#ifdef _OPENMP
			#pragma omp target map(to:halo_dev[0:haloSize])
	#endif
	/***************************************/
	/* Find neighbors and set up trades    */
	/***************************************/
	printf("%s testing, line %d\n", __FILE__,__LINE__);
	// Locate neighbors
	int northP, southP; // along the col or y-axis
	int eastP, westP; // along the row or x-axis
	
	int shift_dim[ndims];
	shift_dim[0] = 0; shift_dim[1] = 1;
	// Cart shift only gives you 1 if there is one below or -2 if not
	MPI_Cart_shift(commCol, shift_dim[1], 1, &northP, &southP);
	MPI_Cart_shift(commRow, shift_dim[0], 1, &westP, &eastP);
    #endif 	
	/***************************************/
	/*Prepare and execute iterations/trades*/
	/***************************************/
	// The code will use either openmp4 or cuda to set the gpu
	// the other options are openacc and eventually opencl
	// therefore, mod = 1 for either openmp4 or cuda and 2 
	// if cuda is there
	int mod = 1;
	#ifdef _OPENACC
		mod++;
		printf("Found OpenACC!\n");
	#endif

	int iterCount = 0;
	int iterEnd = 1; 
	int k = 0; // iteration counter
	while(k<iterEnd) {
		// rotate who is updated and who holds every other iteration
		if (k%2 == 0) {
			update = sub1_dev;
			hold = sub2_dev;
		} else{
			update = sub2_dev;
			hold = sub1_dev;
		}	
		// hold is the current version of the tile		

	printf("%s testing, line %d\n", __FILE__,__LINE__);
		#ifdef HAVE_MPI
		// get the halo values off of the gpu using the package
		// then exchange those values using MPI

		if(k%mod==0) {
			#ifdef __CUDACC__
				call_cuda_applyRules(0,myRows,myCols,haloIn,halo_dev,update,hold);
			#endif
			
			#ifdef _OPENMP
				call_OpenMP4_applyRules(0,myRows,myCols,haloIn,halo_dev,update,hold);
			#endif
		}

		if(k%mod==1) {
			#ifdef _OPENACC
				call_OpenACC_applyRules(0,myRows,myCols,haloIn,halo_dev,update,hold);
			#endif
		}
		// Add in the duplicates at the corners
		haloIn[n] = haloIn[n+1];
		haloIn[e-1] = haloIn[e-2];
		haloIn[e] = haloIn[e+1];
		haloIn[s-1] = haloIn[s-2];		
		haloIn[s] = haloIn[s+1];
		haloIn[w-1] = haloIn[w-2];
		haloIn[w] = haloIn[w+1];
		haloIn[haloSize-1] = haloIn[haloSize-2];
		MPI_Request reqs[4];

		// send cols
		MPI_Isend(&(haloIn[w]),myRows,MPI_INT,westP,0,commRow,&reqs[0]);	
		MPI_Isend(&(haloIn[e]),myRows,MPI_INT,eastP,1,commRow,&reqs[1]);	
		// recv cols
		MPI_Irecv(&(haloOut[e]),myRows,MPI_INT,eastP,0,commRow,&reqs[2]);	
		MPI_Irecv(&(haloOut[w]),myRows,MPI_INT,westP,1,commRow,&reqs[3]);	
		MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
		
		// add north/east corner to north
		haloIn[e-1] = haloOut[e];
		// add south/east corner to south
		haloIn[s+myCols-1] = haloOut[s-1];
		// add north/west corner to north
		haloIn[n] = haloOut[w];;
		// add south/west corner to south
		haloIn[s] = haloOut[haloSize-1];
		
		// send rows after
		MPI_Request reqs2[4];
		// send rows
		MPI_Isend(&(haloIn[n]),myCols,MPI_INT,northP,0,commCol,&reqs2[0]);
		MPI_Isend(&(haloIn[s]),myCols,MPI_INT,southP,1,commCol,&reqs2[1]);
		// receive data
		MPI_Irecv(&(haloOut[n]),myCols,MPI_INT,northP,1,commCol,&reqs2[2]);
		MPI_Irecv(&(haloOut[s]),myCols,MPI_INT,southP,0,commCol,&reqs2[3]);
		MPI_Waitall(4, reqs2, MPI_STATUSES_IGNORE);
		
		// Add north/east corner to east
		haloOut[e] = haloOut[e-1];
		// add south/east corner to east
		haloOut[s-1] = haloOut[s+myCols-1]; 
		// add north/west corner to west
		haloOut[haloSize-1] = haloOut[s]; 
		// add south/west corner to west
		haloOut[w] = haloOut[n];
		#endif
	    	
		// send halo data to GPU, and apply rules	
		if(k%mod==0) {
			#ifdef __CUDACC__
				printf("Applying Rules: CUDA\n");
				call_cuda_applyRules(1,myRows,myCols,haloOut,halo_dev,update,hold);
			#endif
			
			#ifdef _OPENMP
				printf("Applying Rules: OpenMP\n");
				call_OpenMP4_applyRules(1,myRows,myCols,haloOut,halo_dev,update,hold);
			#endif
		}

		// run openacc if it has been found
		if(k%mod==1) {
			#ifdef _OPENACC
				printf("Applying Rules: OpenACC\n");
				call_OpenACC_applyRules(1,myRows,myCols,haloOut,halo_dev,update,hold);
			#endif
		}
		// now update is the current version of the tile	

	printf("%s testing, line %d\n", __FILE__,__LINE__);
		k++; // +1 counter for each iteration
	} // end of iteration loop
	
	/***************************************/
	/* Free and Finalize                   */
	/***************************************/
	dealloc(subMatrix);
	#ifdef HAVE_MPI
		MPI_Comm_free(&commWorld);
		MPI_Comm_free(&commRow);
		MPI_Comm_free(&commCol);
	
    	MPI_Finalize();
	#endif

	printf("%s testing, line %d\n", __FILE__,__LINE__);
	// release the memory for each pointer
	#ifdef __CUDACC__
		call_loadTile_CUDA(1,matSize,sub1D,&sub1_dev);
		call_loadTile_CUDA(1,matSize,sub1D,&sub2_dev);
		#ifdef HAVE_MPI
			call_loadTile_CUDA(1,matSize,sub1D,&halo_dev);
		#endif
	#endif
	// end timing
	diff = clock() - start;
	int msec = diff*1000/CLOCKS_PER_SEC;

	#ifdef HAVE_MPI
	printf("Rank %d -- Time for %d: %d sec, %d millsec\n",myrank,grid*grid,msec/1000,msec%1000);
	#endif
	printf("Game complete!\n");
	exit(0);
} // end of main
/***************************************/
/* Function: alloc 2D                  */
/***************************************/

int **alloc2D(int row, int col) {
	int i, j;
	size_t mem_size;
	int **arr;

	mem_size = col*sizeof(int *);
	arr = (int **)malloc(mem_size);

	mem_size = row*col*sizeof(int);
	arr[0] = (int *)malloc(mem_size);

	for(i = 1; i< col; i++) {
		arr[i] = arr[i-1]+row;
	}
	
	for(i=0;i<row;i++) {
		for(j=0;j<col;j++){
			arr[i][j] = 0;
		}
	}	
	return arr;
}

/***************************************/
/* Function: dealloc 2D array          */
/***************************************/
void dealloc(int **array) {
	free(array[0]);
	free(array);
}

/***************************************/
/* Function: seed the matrix           */
/***************************************/
void initialize(int myrank,int row, int col, int **subMat){
	int i, j;
	srand(1985);
	for (i=0;i<row-2;i++) {
		for(j=0;j<col-2;j++) {
			subMat[i+1][j+1] = rand() & 1;
		}
	}
}
