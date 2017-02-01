#!/bin/csh
#module load openmpi
#module load gcc
echo compiling game of life
rm iter*
make clean
make
# mpi run new file
mpirun -np 9 ./gameOfLife 
echo done 

