#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "clockcycle.h"

#define ARRAY_SIZE 1000000000
#define clock_frequency 512000000

int MPI_P2P_Reduce(
    long long* send_data,
    long long* recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Comm comm){
        MPI_Status status;
        int rank;
        MPI_Comm_rank(comm,&rank);

        if(rank == 0){
            MPI_Request request;
            int size;
            long long sum = 0;
            MPI_Comm_size(comm,&size);
            for(int i = 0; i < size; i++){
                if(i != 0){
                    MPI_Irecv(&recv_data,count,datatype,0,1,comm,&request);
                    sum += *recv_data;
                }
            }
            printf("Total sum: %llu", sum);
        }
        else{
            MPI_Request request;
            MPI_Irecv(&recv_data,count,datatype,0,1,comm,&request);
            *send_data += *recv_data;
            MPI_Isend(&send_data,count,datatype,0,1,comm,&request);
        }
        return MPI_SUCCESS;
    }

int main(int argc, char** argv){
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements = ARRAY_SIZE/size;
    long long sum, finalSum = 0;
    long long *array = (long long*)malloc(elements*sizeof(long long));
    for(int i = 0; i < elements; i++){
        array[i] = rank*elements+i;
    }

    for(int i = 0; i < elements; i++){
        sum += array[i];
    }

    double start_cycles=clock_now();
    MPI_P2P_Reduce(&sum,&finalSum,sizeof(MPI_LONG_LONG),MPI_LONG_LONG,MPI_COMM_WORLD);
    double end_cycles=clock_now();

    double p2pTime = (end_cycles - start_cycles)/clock_frequency;

    printf("P2P time: %d", p2pTime);

    start_cycles=clock_now();
    MPI_Reduce(&sum,&finalSum,sizeof(MPI_LONG_LONG),MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    end_cycles=clock_now();

    double reduceTime = (end_cycles - start_cycles)/clock_frequency;

    printf("Reduce time: %d", reduceTime);

    free(array);

    MPI_Finalize();
    return 0;
}