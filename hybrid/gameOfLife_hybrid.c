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
#include "applyRules.h"
#include <time.h>

#define ndims 2
#define squareFactor 3 // sqrt of num of proc
#define grid 5

// Function Declarations
int **alloc2D(int row, int col);
int *alloc1D(int row);
void dealloc(int **array);
void initialize(int myrank,int row, int col, int **subMatrix);
void collectData(int inter, int subRow, int subCol, int **subMatrix,
				 int nPErows, int nPEcols, int myrank, MPI_Comm comm);
//MAIN
int main(int argc, char *argv[]) {
	clock_t start = clock(), diff;
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

	// Initialize communicators
	MPI_Comm commRow;
	MPI_Comm commCol;
	MPI_Comm commWorld; // comm between everyone

    // Set up the MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	// error handling
	MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

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

	if(myrank == 0) {
		printf("P[%d], World[%d], Dims[%d x %d]\n", myrank, size, dims[0],dims[1]);
	}
	
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
	
	/***************************************/
	/*       Set up Game Topology          */
	/***************************************/
	// move to in line argument sometime
	int myRows = grid;
	int myCols = grid;
	
	// Create my subMatrix
	int i, j;
	int **subMatrix;
	subMatrix = alloc2D(myRows,myCols);
	
	// intialize grid
	initialize(myrank,myRows,myCols,subMatrix);
	// generate halo matrix, dont fill yet
	int **haloGrid;
	haloGrid = alloc2D(myRows+2, myCols+2);
	
	/***************************************/
	/* Find neighbors and set up trades    */
	/***************************************/
	// Locate neighbors
	int northP, southP; // along the col or y-axis
	int eastP, westP; // along the row or x-axis
	
	int shift_dim[ndims];
	shift_dim[0] = 0; shift_dim[1] = 1;
	// Cart shift only gives you 1 if there is one below or -2 if not
	MPI_Cart_shift(commCol, shift_dim[1], 1, &northP, &southP);
	MPI_Cart_shift(commRow, shift_dim[0], 1, &westP, &eastP);
		
	//generate col vectors -> need to do since cols are not contiguous
	MPI_Datatype colSlice;
	MPI_Type_vector(myCols+2,1,myRows+2,MPI_INT,&colSlice);	
	MPI_Type_commit(&colSlice);
	
	/***************************************/
	/* Buffer Set Up                       */
	/***************************************/
	int recvBSize = myRows*myCols; // set reciving buffer size
	int *recvSub;
	recvSub = alloc1D(recvBSize);

	int sendBSize = (myRows+2)*(myCols+2); // set sending buffer size
	int *sendHalo;
	sendHalo = alloc1D(sendBSize);
	
	int c; // cuda counter
	
	/***************************************/
	/*Prepare and execute iterations/trades*/
	/***************************************/
	int **subHold; // holds previous subMatrix
	subHold = alloc2D(myRows,myCols);

	int iterCount = 0;
	int iterEnd = 1; 
	int k = 0; // iteration counter
	while(k<iterEnd) {
		// update temp
		for(i=0;i<myRows;i++){
			for(j=0;j<myCols;j++){
				memcpy(&subHold[i][j], &subMatrix[i][j], sizeof(int));
			}
		}
		
		// fill inner halo
		for(i=0;i<myRows;i++){
			for(j=0;j<myCols;j++){
				haloGrid[i+1][j+1] = subMatrix[i][j];
			}
		}
		// Collect data from all processors to 0		
		//collectData(k, myRows, myCols, subMatrix,nRows,nCols,myrank, commWorld); 

 		//exchange ghost data using MPI 
 		//we want corner pieces so lets split this up
		MPI_Request reqs[4];
		// send cols
		MPI_Isend(&(haloGrid[0][1]),1,colSlice,westP,9,commRow,&reqs[0]);	
		MPI_Isend(&(haloGrid[0][myCols]),1,colSlice,eastP,9,commRow,&reqs[1]);	
		// recv cols
		MPI_Irecv(&(haloGrid[0][myCols+1]),1,colSlice,eastP,9,commRow,&reqs[2]);	
		MPI_Irecv(&(haloGrid[0][0]),1,colSlice,westP,9,commRow,&reqs[3]);	
		MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

		// send rows after
		MPI_Request reqs2[4];
		// send rows
		MPI_Isend(&(haloGrid[1][0]),myRows+2,MPI_INT,northP,9,commCol,&reqs2[0]);	
		MPI_Isend(&(haloGrid[myRows][0]),myRows+2,MPI_INT,southP,9,commCol,&reqs2[1]);	
	
		// receive data
		MPI_Irecv(&(haloGrid[0][0]),myRows+2,MPI_INT,northP,9,commCol,&reqs2[2]);	
		MPI_Irecv(&(haloGrid[myRows+1][0]),myRows+2,MPI_INT,southP,9,commCol,&reqs2[3]);	
		MPI_Waitall(4, reqs2, MPI_STATUSES_IGNORE);

		// store haloGrid in buffer for cuda
		c = 0;
		for(i=0;i<myRows+2;i++){
			for(j=0;j<myCols+2;j++){
				sendHalo[c] =  haloGrid[i][j];	
				c++;
			}
		}
		// swtich between cuda, openCl, and openacc  depending on iteration counter
		switch(k%3) {	
			case 0: 
				// call cuda
				call_cuda_applyRules(myRows,myCols,haloGrid,subMatrix,myrank);
				break;
			case 1:
				// call OpenCl
				call_OpenCL_applyRules(myRows,myCols,haloGrid,subMatrix,myrank);
				break;
			case 2:
				// call OpenAcc
				call_OpenACC_applyRules(myRows,myCols,haloGrid,subMatrix,myrank);
				break;
			default:
				printf("Invalid call in gameOfLife_hybrid.c -- %d\n",k);
				break;
		}
		
		// store updated suMatrix from  buffer
		c = 0;
		for(i=0;i<myRows;i++){
			printf("[");
			for(j=0;j<myCols;j++){
				subMatrix[i][j] = recvSub[c];	
				c++;
			}
		}

		k++; // +1 counter for each iteration
	} // end of iteration loop
	
		if(myrank==0) {
		// store updated suMatrix from  buffer
		//c = 0;
		for(i=0;i<myRows;i++){
			printf("[");
			for(j=0;j<myCols;j++){
				printf(" %2d ",subMatrix[i][j]);
			}
			printf("]\n");
		}

		int *sub1D = (int *)&(subMatrix[0][0]);
		//printf("%p %d %p %d\n",subMatrix,subMatrix[0][0],sub1D, sub1D[0]);

		for(i=0;i<myRows*myCols;i++){
			printf(" %d ",sub1D[i]);
		}
	// So we are now contiguos for sure. Lets look at sending points
	// to the other GPU methods instead of indiviidual buffers
	}
	/***************************************/
	/* Free and Finalize                   */
	/***************************************/
	free(sendHalo);
	free(recvSub);
	dealloc(subMatrix);
	dealloc(haloGrid);
	dealloc(subHold);
	MPI_Type_free(&colSlice);
	MPI_Comm_free(&commWorld);
	MPI_Comm_free(&commRow);
	MPI_Comm_free(&commCol);
	
    MPI_Finalize();
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	//printf("MY RANK: %d, time: %d seconds, %d milliseconds\n", myrank,msec/1000,msec%1000);
    exit(0);
} // end of main

/***************************************/
/* Function: alloc 1D                  */
/***************************************/
// This is necesary because as the buffers get large
// it will take over stack memory. Malloc moves it to heap
int *alloc1D(int row) {
	int i;
	// set up the a row of pointers
	int *arr = (int *)malloc(row*sizeof(int));

	for(i=0;i<row;i++) {
		arr[i] = 0;
	}	
	return arr;
}
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
	/* set up the a row of pointers
	int **arr = (int **)malloc(row*sizeof(int *));
	// set the first element to point to the start of rest
	arr[0] = (int *)malloc(row*col*sizeof(int));

	for(i = 1; i< row; i++) {
		arr[i] = arr[0]+col*i;
	} */
	for(i=0;i<row;i++) {
		for(j=0;j<col;j++){
			arr[i][j] = i*10+j;
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
/* Function: Methuselah                */
/* 			 initialize the grid	   */
/***************************************/
void initialize(int myrank, int row, int col, int **subMatrix) {
	int i, j; // interative variables
	srand(1985); // found this on Oak Ridge tutorials
	/*int loc = (int)grid-1;
	if (myrank==0) {
		subMatrix[loc][loc] = 1;
		subMatrix[loc][loc-1] = 1;
		subMatrix[loc-1][loc] = 1;
		subMatrix[loc-1][loc-1] = 1;
	}	
	if (myrank==4) {
		subMatrix[0][0] = 1;
		subMatrix[0][1] = 1;
		subMatrix[1][0] = 1;
		subMatrix[1][1] = 1;
		subMatrix[loc][loc] = 1;
		subMatrix[loc][loc-1] = 1;
		subMatrix[loc-1][loc] = 1;
		subMatrix[loc-1][loc-1] = 1;
	}	
	if (myrank==8) {
		subMatrix[0][0] = 1;
		subMatrix[0][1] = 1;
		subMatrix[1][0] = 1;
		subMatrix[1][1] = 1;
	}	
	*/
	// fill ranks with random numbers
	if (myrank==0 || myrank==4 || myrank==8) {
		for(i=0;i<row;i++) {
			for(j=0;j<col;j++) {
				subMatrix[i][j] = rand() & 1;
			}
		}
	}
}
/***************************************/
/* Function: Collect data              */
/* 			 All ranks send data to 0  */
/***************************************/
void collectData(int iter, int row, int col, int **subMatrix, 
				 int nPErows, int nPEcols, int myrank, MPI_Comm comm) {
	// row and col only span subMatrix, nPE* goes to entire grid
	FILE *file;
	char name[32]; // filename
	int matSize = row*col;
		
	MPI_Status status;

	// rank 0 will manage all of the matrix print out
	if (myrank == 0) {
		int coords[ndims];
		int nProcs = nPEcols*nPErows;
		int i, j, source; // counter and source proc
		int indexR, indexC; // counter in totalMatrix

		int **hold; // holds matrix from other procs
		hold = alloc2D(row, col);
		
		int **totalMatrix; // holds the entire grid
		int allRows = row*nPErows; // all row elements
		int allCols = col*nPEcols; // all cols
		totalMatrix = alloc2D(allRows, allCols);
	
		// put in 0's data
		for(i=0;i<row;i++) {
			for(j=0;j<col;j++) {
				totalMatrix[i][j] = subMatrix[i][j];
			}
		}
	
		// revc data from others
		for(source=1; source<nProcs; source++) {
			MPI_Recv(&(hold[0][0]), matSize,MPI_INT,source,source,comm,&status);
			MPI_Cart_coords(comm,source,ndims,coords);// locate the source
			
			for(i=0;i<row;i++) {
				indexR = i + coords[0]*row;
				for(j=0;j<col;j++) {
					indexC = j + coords[1]*col;
					totalMatrix[indexR][indexC] = hold[i][j];
				}
			}

		}
		
		// make file
		sprintf(name,"iter_%d.txt",iter);
		file = fopen(name,"w");
		if (file==NULL) exit(-1);

		// print to file
		for(i=0;i<allRows;i++){
			for(j=0;j<allCols;j++){
				fprintf(file," %d ", totalMatrix[i][j]);
			}
			fprintf(file,"\n");
		}
		
		dealloc(totalMatrix);
		dealloc(hold);
	} else { // if I am any other proc
		MPI_Send(&(subMatrix[0][0]), matSize, MPI_INT,0,myrank,comm);

	}
		
}

