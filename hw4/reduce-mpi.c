#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"
#include "reduce-cuda.cu"

#define ARRAY_SIZE 1610612736
#define clock_frequency 512000000

int main(int argc, char** argv){

    //Initialize MPI
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Compute local array sum
    int elements = ARRAY_SIZE/size;
    double sum, finalSum = 0;
    double *array = (double*)malloc(elements*sizeof(double));
    for(int i = 0; i < elements; i++){
        array[i] = rank*elements+i;
    }

    for(int i = 0; i < elements; i++){
        sum += array[i];
    }

    //Display results for root process
    if(rank == 0){

    start_cycles=clock_now();
    MPI_Reduce(&sum,&finalSum,sizeof(MPI_DOUBLE),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    end_cycles=clock_now();

    double reduceTime = (end_cycles - start_cycles)/clock_frequency;

    printf("Reduce time: %f\n", reduceTime);
    }

    free(array);

    MPI_Finalize();
    return 0;
}