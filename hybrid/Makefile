## target makes
.PHONY: all build clean

#MPI_INCLUDE = ${MPI_ROOT}/include
#MPI_LIB = ${MPI_ROOT}/lib

#CUDA_INCLUDE = ${CUDA_INCLUDES}

OpenCL_LIBS = -L$(CUDA_LIBS)/

COM_LIBS = -L${MPI_LIB} -lmpi -L${CUDA_LIB} -lcuda -L${OpenCL_LIBS} -lOpenCL

all: build

build: gameOfLife

# compile cu as object
applyRules_CUDA.o:
	nvcc -c -I${CUDA_INCLUDE} applyRules_CUDA.cu

# compile cu as object
applyRules_OpenCl.o:
	gcc -c -I${MPI_INCLUDE} -I${CUDA_INCLUDE}/ -lOpenCl applyRules_OpenCl.c

# compile cu as object
applyRules_OpenAcc.o:
	gcc -c -lOpenAcc applyRules_OpenAcc.c

# compile c as object
gameOfLife_hybrid.o:
	gcc -c -I${MPI_INCLUDE} gameOfLife_hybrid.c

## compile all objects together
gameOfLife: gameOfLife_hybrid.o applyRules_CUDA.o applyRules_OpenCl.o applyRules_OpenAcc.o
	nvcc -g $^ ${COM_LIBS} -o gameOfLife

clean:
	rm -f *.o *.txt gameOfLife

