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

/***************************************/
/* OpenACC SubRoutine                  */
/***************************************/
void call_OpenACC_applyRules(int rows, int cols, int *haloMat, int *subMat, int myrank) {

	if (myrank==0) {
		printf("__OpenAcc Portion:__\n");
	}
	int i,j; // iteration counters
	int n, s, e, w, nw, ne, sw, se; // location in halo
	int liveCells = 0;	

	int haloElements = (rows+2)*(cols+2);
	int subElements = rows*cols;

	int hStride = rows+2;

	/***************************************/
	/* Initialize OpenACC                  */
	/***************************************/
	#pragma acc kernels	
	#pragma acc data copy(haloMat), shared(subMat)
	{
		#pragma acc loop
		for(i=0;i<subElements;i++) {
			int sInd = i;
			int h_j = sInd%cols;
			int h_i = sInd/rows;
			int hInd = (1+ h_i)*(cols+2) + h_j + 1;
			
			e = hInd + 1;
			w = hInd - 1;
			n = hInd - (rows+2);
			ne = n + 1;
			nw = n - 1;
			s = hInd + (rows+2);
			se = s + 1;
			sw = s - 1;
			
			liveCells = haloMat[nw] + haloMat[n] + haloMat[ne]
					  + haloMat[w]               + haloMat[e]
			   	  + haloMat[sw] + haloMat[s] +haloMat[se]; 	
			
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
	}
	
	return;
}
