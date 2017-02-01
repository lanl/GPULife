## target makes
.PHONY: all build clean
# module load pgi and cuda
MPI_CFLAGS = ${MPI_ROOT}/include
MPI_LDFLAGS = ${MPI_ROOT}/lib
#MPI_CFLAGS = $(shell mpicc --showme:compile)
#MPI_LDFLAGS = $(shell mpicc --showme:link)
CC = pgcc -g -acc -mp -cudalibs 

COM_LIBS = -L${MPI_LIB} -lmpi -L${CUDA_LIB} -lcuda

all: build

build: run

loadTile_CUDA.o:
	nvcc -g -c -I${MPI_INCLUDE} loadTile_CUDA.cu

applyRules_CUDA.o:
	nvcc -g -c -I${MPI_INCLUDE} applyRules_CUDA.cu

applyRules_OpenAcc.o:
	${CC} -c applyRules_OpenAcc.c

main.o:
	${CC} -c -I${MPI_INCLUDE} main.c

run: main.o loadTile_CUDA.o applyRules_CUDA.o applyRules_OpenAcc.o
	${CC} -o gameOfLife $^ ${COM_LIBS} 

clean:
	rm -f *.o *.txt gameOfLife

