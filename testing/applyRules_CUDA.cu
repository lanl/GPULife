/*  LA-CC-16080
    Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   	Priscilla Kelly <priscilla.noreen@gmail.com>
*/
#include "stdio.h"
#include "stdlib.h"
#define maxThread 32
/*******************************************/
/* Cuda kernel to apply the rules of Life  */
/*******************************************/
__global__ void applyRules(int row,int col,int *update, int *hold) {
	int threadMax = blockDim.x;
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int linID;
	if (blockID == 1) {
		linID = threadID;
	} else {
		linID = blockID*threadMax+threadID;
	}
	int elements = (row-2)*(col-2);
	
	int i = linID%(col-2);
	int j = linID/(row-2);
	int loc = col + i*col + j + 1;
	if (linID < elements) {
		int liveCells = 0;   

		int n, s, e, w, nw, ne, sw, se; // location in halo

		n = loc-col;
		nw = n-1;
		ne = n+1;
		w = loc-1;
		e = loc+1;
		s = loc+col;
    	sw = s-1;
    	se = s+1; 

		liveCells = hold[nw] + hold[n] + hold[ne]
				  + hold[w]            + hold[e]
		    	  + hold[sw] + hold[s] + hold[se]; 	
	
		// Apply Rules
		if (hold[loc] == 0) {
			if (liveCells == 3) {
				update[loc] = 1; // reproduction
			} else {
				update[loc] = 0; // remain dead
			}
		} else {  
			if (liveCells < 2){ 
				update[loc] = 0; // under population
			} else {
				if (liveCells < 4) {
					update[loc] = 1; // survivor
				} else {
					 update[loc] = 0; // over population
				}
			}
		}
	}
}
/*******************************************/
/* Cuda kernel to upload N/S halo elements */
/*******************************************/
__global__ void add_NS_Halo(int row,int col,int *haloMat,int *subMat) {
	int b = blockIdx.x;
	int t = threadIdx.x;
	// add North portion	
	if (b == 0) {
		subMat[t] = haloMat[t];
	}
	// add South portion
	if (b == 1) {
		int subLoc = col*(row-1)+t;
		int haloLoc = (row+col)+t;
		subMat[subLoc] = haloMat[haloLoc];
	}
}
/*******************************************/
/* Cuda kernel to upload E/W halo elements */
/*******************************************/
__global__ void add_EW_Halo(int row,int col,int *haloMat,int *subMat) {
	int b = blockIdx.x;
	int t = threadIdx.x;
	// add East portion	
	if (b == 0) {
		int subLoc = col-1+(col)*t;
		int haloLoc = col+t;
		subMat[subLoc] = haloMat[haloLoc];
	}
	// add the West portion
	if (b == 1) {
		int subLoc = (col)*t;
		int haloLoc = 2*row+col+t;
		subMat[subLoc] = haloMat[haloLoc];
	}
}

/*******************************************/
/* Cuda kernel to get N/S halo elements */
/*******************************************/
__global__ void get_NS_Halo(int row,int col,int *haloMat,int *subMat) {
	int b = blockIdx.x;
	int t = threadIdx.x;
	// add North portion	
	if (b == 0) {
		haloMat[t]=subMat[t+col]; 
	}
	// add South portion
	if (b == 1) {
		int subLoc = col*(row-2)+t;
		int haloLoc = (row+col)+t;
		haloMat[haloLoc]=subMat[subLoc];
	}
}
/*******************************************/
/* Cuda kernel to get E/W halo elements */
/*******************************************/
__global__ void get_EW_Halo(int row,int col,int *haloMat,int *subMat) {
	int b = blockIdx.x;
	int t = threadIdx.x;
	// add East portion	
	if (b == 0) {
		int subLoc = (col-2)+col*t;
		int haloLoc = col+t; 
		haloMat[haloLoc]=subMat[subLoc];
	}
	// add the West portion
	if (b == 1) {
		int subLoc = 1+col*t;
		int haloLoc = col+2*row+t;
		haloMat[haloLoc]=subMat[subLoc] ;
	}
}
/***************************************/
/* External c subroutine for CUDA      */
/***************************************/
extern "C" void call_cuda_applyRules(int flag,int rows, int cols,int *halo, int *halo_dev, int *update, int *hold) {

	/**************************************************/
	/* Get the values to exchange over MPI */
	/**************************************************/
	if (flag == 0) {
	int haloSize = sizeof(int)*2*(rows+cols);
	cudaError_t err = cudaSuccess;
		
	// Add the North and South rows to hold
	get_NS_Halo<<<2, cols>>>(rows,cols,halo_dev,hold);
	err = cudaGetLastError();	
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch Kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaDeviceSynchronize();
	
	// Add the East and West columns to hold
	get_EW_Halo<<<2, rows>>>(rows,cols,halo_dev,hold);
	err = cudaGetLastError();	
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch Kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaDeviceSynchronize();
	
	err = cudaMemcpy(halo,halo_dev,haloSize,cudaMemcpyDeviceToHost);	
	return;
	}

	/*****************************************************/
	/* Update hold with halo, then apply rules to update */
	/*****************************************************/
	if (flag == 1) {
	int haloSize = sizeof(int)*2*(rows+cols);
	cudaError_t err = cudaSuccess;
	
	// Copy updated halo to GPU
	err = cudaMemcpy(halo_dev,halo,haloSize,cudaMemcpyHostToDevice);	
	
	// Add the North and South rows to hold
	add_NS_Halo<<<2, cols>>>(rows,cols,halo_dev,hold);
	err = cudaGetLastError();	
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch Kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaDeviceSynchronize();

	// Add the East and West columns to hold
	add_EW_Halo<<<2, rows>>>(rows,cols,halo_dev,hold);
	err = cudaGetLastError();	
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch Kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	// Apply the Rules

	int N = (rows-2)*(cols-2);
	int threadCnt;
	int blockCnt;
	
	//threads per block
	threadCnt = maxThread;

	//blocks
	int check = N/maxThread;
	if (check == 0) {
		blockCnt = 1;
	} else {
		blockCnt = check + 1;
	}
	applyRules<<<blockCnt, threadCnt>>>(rows,cols,update,hold);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy halo to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();

	return;
	}
}
