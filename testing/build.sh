#!/bin/csh
#module load pgi
#module load cuda
#module load openmp/1.10.2-pgi
#module unload gcc

echo compiling game of life
rm -rf CMakeFiles
rm CMakeCache.txt
rm *.o
make 
# mpi run new file
#mpirun --report-bindings -np 4 ./gameOfLife
#mpiexec -np 4 -H cn85,cn84 --bind-to core --npernode 2 ./gameOfLife
#mpirun -np 4 -H cn63,cn67,cn146,cn15 --bind-to core --npernode 1 ./gameOfLife
#mpirun -np 1 -H cn81 ./gameOfLife
./gameOfLife
echo done

