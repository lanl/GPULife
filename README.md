This is repository is a demonstration of GPU interoperability. 

So far, only CUDA and OpenACC with MPI are functioning, but 
OpenMP4 is being added in Testing. 

CUDA+OpenACC: contains the interoperable GPU languages + MPI
Hybrid: contains all the GPU languages, but they are not interoperable

testing: OpenACC+OpenMP4 with cmake

  In testing directory
  -- with PGI 16.5, OpenMPI 1.10.2, Cuda 8.0
  cmake .
  make
  mpirun -n 2 ./gameOfLife
  -- with GCC 6.1.0, OpenMPI 1.10.3, Cuda 8.0
  cmake .
  make
  mpirun -n 2 ./gameOfLife
  -- with Intel 16.0.3, OpenMPI 1.10.3, Cuda 8.0
  cmake .
     Fails to detect OpenACC and doesn't set compile definition
     Detect OpenMP and sets flags, but it doesn't have OpenMP 4.0 with
       the target (accelerator) directives
  make
  mpirun -n 2 ./gameOfLife

// OpenAcc Code for Game Of Life 
// Priscilla Kelly June 28, 2016                                 

/*  LA-CC-16080    
    Copyright © 2016 Priscilla Kelly and Los Alamos National Laboratory. All Rights Reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. The name of the author may not be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Priscilla Kelly and Los Alamos National Laboratory "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Priscilla Kelly <priscilla.noreen@gmail.com>
        GPULife applyRules_OpenACC
*/

