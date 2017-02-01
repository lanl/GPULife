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
