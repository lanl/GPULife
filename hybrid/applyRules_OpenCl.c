/*  LA-CC-16080
    Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   	Priscilla Kelly <priscilla.noreen@gmail.com>
*/
#include <CL/cl.h>
#include "stdio.h"
#include "stdlib.h"

/****************************************/
/* OpenCl Kernel  (Apply the GoL Rules) */
/****************************************/
// elements is the total number of calculations we need to do
// wrkID is the work group id
// locID is the local group id (0-32 or 64)
// i and j is the location of a given subMat in Halo
// 8 halo cells matter -> n,ne,e,se,s,sw,w,nw
// cannot print on a gpu
// note that if statments must be bracketed or it will not register 
const char *kernelSource =											   "\n" \
"__kernel void applyRulesOCL(const int rows, const int cols,            \n" \
"	                   __global int* haloGrid, __global int* subMat)    \n" \
"{																		\n" \
"	int elements = (rows)*(cols);    									\n" \
" 	int n ,ne, e, se, s, sw, w, nw;                                     \n"\
"	int wrkID = get_group_id(0);										\n" \
"	int locID = get_local_id(0);										\n" \
"	int wrkSize = get_local_size(0);									\n" \

"   int subIndex = wrkSize*wrkID + locID;    							\n"\
"   int h_j = subIndex%cols;											\n"\
"   int h_i = subIndex/rows;											\n"\

"   int haloIndex = (1 + h_i)*(cols+2)+ h_j + 1;  						\n"\
"   if (subIndex < elements) {				    						\n"\
"		e = haloIndex + 1;												\n"\
"		w = haloIndex - 1;												\n"\
"		n = haloIndex - (rows+2);										\n"\
"		ne = n + 1;														\n"\
"		nw = n - 1;														\n"\
"		s = haloIndex + (rows+2);										\n"\
"		se = s + 1;														\n"\
"		sw = s - 1;														\n"\
"		int liveCells = 0;												\n"\
"   	liveCells = haloGrid[nw] + haloGrid[n] + haloGrid[ne]			\n"\
"		          + haloGrid[w]                + haloGrid[e]			\n"\
"		 		  + haloGrid[sw] + haloGrid[s] + haloGrid[se];			\n"\
"		if (haloGrid[haloIndex] == 0) {									\n"\
"			if (liveCells == 3) {										\n"\
"				subMat[subIndex] = 1; 									\n"\
"			} else {													\n"\
"				subMat[subIndex] = 0;									\n"\
"			}															\n"\
"		} else {  														\n"\
"			if (liveCells < 2) {										\n"\
"				subMat[subIndex] = 0;				 					\n"\
"			} else {													\n"\
"				if (liveCells < 4) {									\n"\
"					subMat[subIndex] = 1;								\n"\
"				} else {												\n"\
"					subMat[subIndex] = 0; 								\n"\
"				}														\n"\
"			}															\n"\
"		}																\n"\
"   }                                                                   \n"\
"}                                                                      \n"\
																		"\n";

/***************************************/
/* OpenCL SubRoutine                   */
/***************************************/
void call_OpenCL_applyRules(int rows, int cols, int *haloMat, int *subMat, int myrank) {

	int i,j; // iteration counters
	
	/***************************************/
	/* Initialize OpenCL                   */
	/***************************************/
	int elements = rows*cols;
	size_t haloSize = (rows+2)*(cols+2);
	size_t subSize = rows*cols;
	size_t globalSize; // we need threads for every data point
	size_t localSize; // globalSize as a factor of 32

	cl_mem d_subMat; // device submat for openCL
	cl_mem d_haloMat; 

	cl_platform_id cpPlatform; // OpenCL platform
	cl_device_id device_id;	   // get device id
	cl_context context;		   // compute context 
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;

	int err; // will have returned error codes
	
	// bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a platform\n");
		return;
	}	
	// get ID for the device
	err = clGetDeviceIDs(cpPlatform,CL_DEVICE_TYPE_GPU,1,&device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group\n");
		return;		
	}
	// create a context
	context = clCreateContext(0,1,&device_id,NULL,NULL,&err);
	if (!context) {
		printf("Error: Failed to create a computer context\n");
		return;		
	}

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) {
		printf("Error: Failed to create a command queue\n");
		return;		
	}
	// Create the compute program for the kernel
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
	if (!program) {
		printf("Error: Failed to create a compute program!\n");
		return;		
	}
	
	// build executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to build program excecutable! %d\n", err);
		return;		
	}

	// Create the compute kernel 
	kernel = clCreateKernel(program, "applyRulesOCL", &err);
	if (!kernel || err != CL_SUCCESS) { 
		printf("Error: Failed to create kernel!\n");
		return;		
	}
	
	// create input and output arrays
	d_haloMat = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*haloSize, NULL,NULL);
	d_subMat = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*subSize,NULL,NULL);
	if(!d_haloMat || !d_subMat) {
		printf("Error: Failed to allocate device memory!\n");
		return;
	}

	// write data to device memory
	err = clEnqueueWriteBuffer(commands,d_haloMat,CL_TRUE,0,sizeof(int)*haloSize,haloMat,0,NULL,NULL);
	if (err != CL_SUCCESS) { 
		printf("Error: Failed to write source array!\n");
		return;		
	}
	
	// set the kernel arguements
	err = 0;
	err = clSetKernelArg(kernel,0,sizeof(int), &rows);
	err |= clSetKernelArg(kernel,1,sizeof(int), &cols);
	err |= clSetKernelArg(kernel,2,sizeof(cl_mem), &d_haloMat);
	err |= clSetKernelArg(kernel,3,sizeof(cl_mem), &d_subMat);
	if (err != CL_SUCCESS) { 
		printf("Error: Failed to set arguements!\n");
		return;		
	}

	// Create work group	
	// localSize should be a multipule of 32
	int mod = elements%32;
	if (mod == 0) { // check if the total work elements is a factor of 32 
		globalSize = elements;

	} else {
		globalSize = elements + (32-mod);
	}
	// groupSize is now a factor of 32, localSize must be divisable by that
	// NVIDIA recommends 32 threads, AMD recommends 64
	localSize = 32;
	if (myrank==0) {
		printf("__OpenCL Portion:__\n");
		printf("Global Work Size: %d\n",globalSize);
		if (mod == 0) {
			printf("There are no idle threads\n");
		} else {
			printf("There are %d idle threads\n",32-mod);
		}
		printf("Local Work Size: %d\n",localSize);
	}
	// execute kernel using the max amount of elements
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err != CL_SUCCESS) { 
		printf("Error: Failed to execute kernel!\n");
		return;		
	}
	// wait for all to finish
	clFinish(commands);

	// read back updated data
	err = clEnqueueReadBuffer(commands,d_subMat,CL_TRUE,0,sizeof(int)*subSize,subMat,0,NULL,NULL);
	if (err != CL_SUCCESS) { 
		printf("Error: Failed to read output array!\n");
		return;		
	}

	// release and clean
	clReleaseMemObject(d_haloMat);
    clReleaseMemObject(d_subMat);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

	return;
}
