#!/bin/csh

echo compiling game of life
rm iter*
make clean
make run
# mpi run new file
#mpirun --report-bindings -np 4 ./gameOfLife
#mpiexec -np 4 -H cn85,cn84 --bind-to core --npernode 2 ./gameOfLife
mpirun -np 4 -H cn15,cn21,cn22,cn23 --bind-to core --npernode 1 ./gameOfLife
echo done

