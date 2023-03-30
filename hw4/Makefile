all: reduce-mpi.c reduce-cuda.cu
<tab> mpixlc -O3 reduce-mpi.c -c -o reduce-mpi.o
<tab> nvcc -O3 reduce-cuda.cu -c -o reduce-cuda.o
<tab> mpixlc -O3 reduce-mpi.o reduce-cuda.o -o reduce-exe \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++