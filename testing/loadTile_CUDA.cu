/*  LA-CC-16080
    Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   	Priscilla Kelly <priscilla.noreen@gmail.com>
*/
#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

/***************************************/
/* External c subroutine for CUDA      */
/***************************************/
extern "C" void call_loadTile_CUDA(int flag, int elements, int *Matrix, int **pointer2device) {

	size_t matSize = elements*sizeof(int);
	cudaError_t err = cudaSuccess;

	if (flag == 0) {

		/***************************************/
		/* Allocate Matrix to the GPU          */
		/***************************************/
		int *device;
		err = cudaMalloc(&device, matSize);
		if(err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector (error code %s)!\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);		
		}
		// move matrix to device
		err = cudaMemcpy(device,Matrix,matSize,cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "%s Failed at line %s !\n",__FILE__,__LINE__);
			exit(EXIT_FAILURE);
		}
        
		*pointer2device = device;
		return;
	}

	if (flag == 1) {

		/***************************************/
		/* Free Device Global Memory           */
		/***************************************/
		err = cudaFree(*pointer2device);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to free device!\n");
			exit(EXIT_FAILURE);
		}
	}
	
	if (flag == 3) {
		cudaDeviceSynchronize();
		int *host_subMat = (int *)malloc(matSize);	
		if(host_subMat == NULL) {
			fprintf(stderr, "Failed to alocate host vector!\n");
 			exit(EXIT_FAILURE);     
		}
		err = cudaMemcpy(host_subMat,*pointer2device,matSize,cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr,"Failed to copy the submat from device (error code %s)!\n",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		int i,j;
		int c=0;
		printf("New Rank\n");
		for(i=0;i<6;i++) {
			printf("[");
			for(j=0;j<6;j++) {
				printf(" %d ",host_subMat[c]);
				c++;	
			}	
			printf("]\n");
		}
		printf("\n");
		return;
	}	
}	
