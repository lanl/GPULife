#!/bin/csh
#module load openmpi
#module load gcc
echo cleaning...
rm cMakeCache.txt
rm -rf CMakeFiles
echo building new cmake...
cmake .
echo make...
make 
echo building...
./OpenCL_GLtest 
echo done 

