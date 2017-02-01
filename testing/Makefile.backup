## target makes
.PHONY: all build clean
# module load pgi and cuda
MPI_CFLAGS = ${MPI_ROOT}/include
MPI_LDFLAGS = ${MPI_ROOT}/lib
#MPI_CFLAGS = $(shell mpicc --showme:compile)
#MPI_LDFLAGS = $(shell mpicc --showme:link)
CC = gcc -g -fopenmp
#pgcc -g -acc -mp -cudalibs 

COM_LIBS = -L${MPI_LIB} -lmpi 
#-L${CUDA_LIB} -lcuda

all: build

build: run

loadTile_CUDA.o:
	nvcc -g -c -I${MPI_CFLAGS} loadTile_CUDA.cu

applyRules_CUDA.o:
	nvcc -g -c -I${MPI_CFLAGS} applyRules_CUDA.cu

applyRules_OpenAcc.o:
	${CC} -c applyRules_OpenAcc.c

applyRules_OpenMP4.o:
	${CC} -c applyRules_OpenMP4.c

main.o:
	${CC} -c -I${MPI_CFLAGS} main.c

run: main.o  applyRules_OpenMP4.o applyRules_OpenAcc.o
#loadTile_CUDA.o applyRules_CUDA.o applyRules_OpenAcc.o
	${CC} -o gameOfLife $^ ${COM_LIBS} 

clean:
	rm -f *.o *.txt gameOfLife

