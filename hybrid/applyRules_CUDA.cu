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

/*****************************************/
/* Cuda for device (Apply the GoL Rules) */
/*****************************************/
__global__ void applyRules(int row,int col,int *haloMat,int *subMat) {
	// each thread gets a single cell in the halo and can identify  
	// itself by it's block and thread d
	int s_i = blockIdx.x*row; // my row in the subMat (starts at 0)
	int s_j = threadIdx.x;	// my col in the subMat

	int haloBS = row+2; // halo's block stride
	// start at subMatrix's ranges
	int h_i = (blockIdx.x+1)*haloBS; // my row in the halo 
	int h_j = threadIdx.x+1;	// my col in the halo
	int liveCells = 0;   

	int hInd = h_i + h_j;
	int sInd = s_i + s_j;

	int n, s, e, w, nw, ne, sw, se; // location in halo

	n = hInd-haloBS;
	nw = n-1;
	ne = n+1;
	w = hInd-1;
	e = hInd+1;
	s = hInd+haloBS;
    sw = s-1;
    se = s+1; 

	liveCells = haloMat[nw] + haloMat[n] + haloMat[ne]
			  + haloMat[w]               + haloMat[e]
		      + haloMat[sw] + haloMat[s] + haloMat[se]; 	
	
	// Apply Rules
	if (haloMat[hInd] == 0) {
		if (liveCells == 3) {
			subMat[sInd] = 1; // reproduction
		} else {
			subMat[sInd] = 0; // remain dead
		}
	} else {  
		if (liveCells < 2){ 
			subMat[sInd] = 0; // under population
		} else {
			if (liveCells < 4) {
				subMat[sInd] = 1; // survivor
			} else {
				 subMat[sInd] = 0; // over population
			}
		}
	}
}

/***************************************/
/* External c subroutine for CUDA      */
/***************************************/
extern "C" void call_cuda_applyRules(int rows, int cols, int *haloMat, int *subMat, int myrank) {
	int i; // iteration counters
	
	size_t haloSize = (rows+2)*(cols+2)*sizeof(int);
	size_t subMatSize = rows*cols*sizeof(int);
	cudaError_t err = cudaSuccess;

	/*******************************************/
	/* Allocate Host and Device copy of subMat */
	/*******************************************/
	// recvs the new solution
	int *host_subMat = (int *)malloc(subMatSize);
	if(host_subMat == NULL) {
		fprintf(stderr, "Failed to alocate host vector!\n");
		exit(EXIT_FAILURE);		
	}

	// sends current solution to device
	int *device_subMat = NULL; 
	err = cudaMalloc(&device_subMat,subMatSize);

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to alocate device vector (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);		
	}
	/***************************************/
	/* Allocate device copy of haloMat     */
	/***************************************/
	// sends current halo to device
	int *device_haloMat = NULL;
	err = cudaMalloc(&device_haloMat, haloSize);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to alocate device vector (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);		
	}
	// move halo to device
	err = cudaMemcpy(device_haloMat,haloMat,haloSize,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy halo from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	/***************************************/
	/* Launch Cuda Kernel                  */
	/***************************************/
	int blockCnt = rows;
	int threadCnt = cols;

	if (myrank==0) {
		printf("__CUDA Portion:__\n");
		printf("Block Size: %d\n",blockCnt);
		printf("Thread Size: %d\n",threadCnt);
	}
		
	applyRules<<<blockCnt, threadCnt>>>(rows,cols,device_haloMat,device_subMat);
	
	err = cudaGetLastError();	
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch Kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaDeviceSynchronize();

	// copy device subMat to host subMat
	err = cudaMemcpy(host_subMat,device_subMat,subMatSize,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy from device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// save the host copy 
	for (i=0; i < (rows*cols); i++) {
		subMat[i] = host_subMat[i];
	}
	
	/***************************************/
	/* Free Device Global Memory           */
	/***************************************/
	err = cudaFree(device_haloMat);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free halo on device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_subMat);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to free subMat on device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// free host memory
	free(host_subMat);
	//reset device
	err = cudaDeviceReset();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return;
}
