## ALL MPI PORTIONS HAVE BEEN REMOVED
.PHONY: all build clean

#MPI_INCLUDE = ${MPI_ROOT}/include
#MPI_LIB = ${MPI_ROOT}/lib

OpenCL_LIBS = -L$(CUDA_LIBS)/
CC = pgcc -g 
COM_LIBS = -cudalibs -L${OpenCL_LIBS} -lOpenCL -lglut  -lGL

all: build

build: run 

main.o:
	${CC} -c -I. -I${CUDA_INCLUDE}/ -lOpenCl main.c


run: main.o 
	${CC} $^ ${COM_LIBS} -o gameOfLife 

clean:
	rm -f *.o *.txt gameOfLife

