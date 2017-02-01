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
void call_OpenMP4_applyRules(int flag,int rows, int cols, int *halo, int *halo_dev,int *update, int *hold) {
	int i,j; // iteration counters
	int matSize = (rows-2)*(cols-2);
	int haloSize = 2*(rows+cols);
	/***************************************/
	/* Get values from Hold to exchange    */
	/***************************************/
	if (flag == 0) {
		#pragma omp target map(from:hold), map(tofrom:halo[0:haloSize])
		{
			for(i=0;i<rows;i++){
				// adding north to halo
				halo[i] = hold[i+cols];
				// adding south to halo
				halo[rows+cols+i] = hold[cols*(rows-2)+i]; 
			}
			for(j=0;j<cols;j++){
				// adding west to halo
				halo[2*rows+cols+j] = hold[1+j*cols];
				// adding east to halo
				halo[cols+j] = hold[cols - 2 + cols*j]; 
			}
		#pragma omp update to(halo[0:haloSize])
		}
		return;
	} 
	/***************************************/
	/* Apply Rules to Grid                 */
	/***************************************/
	if (flag == 1) {
		#pragma omp target map(tofrom:halo[0:haloSize])
		#pragma target map(from:update), map(from:hold)
		{	
			#ifdef HAVE_MPI	
			// Update Hold with new halo values
			for(i=0;i<rows;i++){
				// adding north to halo
				hold[i] = halo[i];
				// adding south to halo
				hold[(rows-1)*cols+i] = halo[(cols+rows)+i]; 
			}
			for(j=0;j<cols;j++){
				// adding west to halo
				hold[cols*j] = halo[2*rows+cols+j];
				// adding east to halo
				hold[cols-1+cols*j] = halo[cols+j]; 
			}
			#endif
			
			// Apply the Rules of Life
			for(i=0;i<matSize;i++) {
				int h_j = i%(cols-2);
				int h_i = i/(rows-2);
				int loc = (h_i+1)*(cols) + h_j+1;
				int liveCells = 0;
				int e, w, n , ne, nw, s, se, sw;

				e = loc + 1;
				w = loc - 1;
				n = loc - (rows);
				ne = n + 1;
				nw = n - 1;
				s = loc + (rows);
				se = s + 1;
				sw = s - 1;
			
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
		return;
	}
}
