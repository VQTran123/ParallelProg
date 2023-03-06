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
        MPI_Request request;
        int rank, size;
        int stride = 2;
        MPI_Comm_rank(comm,&rank);
        MPI_Comm_size(comm,&size);

        while(stride <= size){
            if(rank%stride == stride/2){
                MPI_Isend(&sendbuf,count,datatype,(rank - stride)/2,1,comm,&request);
                MPI_Wait(&request, &status);
            }
            else if(rank%stride == 0) {
                MPI_Irecv(&recvbuf,count,datatype,(rank + stride)/2,1,comm,&request);
                MPI_Wait(&request, &status);
                *sendbuf += *recvbuf;
            }
            step *= 2;
        }
        
        if(rank == 0){
            printf("%llu\n",*recvbuf);
        }

        /*if(rank == 0){
            int size;
            long long sum = 0;
            MPI_Comm_size(comm,&size);
            for(int i = 0; i < size; i++){
                if(i != 0){
                    printf("%llu\n",sum);
                    MPI_Irecv(&recv_data,count,datatype,i,1,comm,&request);
                    MPI_Wait(&request, &status);
                    sum += *recv_data;
                }
            }
            printf("Total sum: %llu\n", sum);
        }
        else{
            *send_data += *recv_data;
            MPI_Isend(&send_data,count,datatype,0,1,comm,&request);
            MPI_Wait(&request, &status);
        }*/
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

    printf("%llu\n",sum);

    double start_cycles=clock_now();
    MPI_P2P_Reduce(&sum,&finalSum,sizeof(MPI_LONG_LONG),MPI_LONG_LONG,MPI_COMM_WORLD);
    double end_cycles=clock_now();

    if(rank == 0){

    double p2pTime = (end_cycles - start_cycles)/clock_frequency;

    printf("P2P time: %f\n", p2pTime);

    start_cycles=clock_now();
    MPI_Reduce(&sum,&finalSum,sizeof(MPI_LONG_LONG),MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    end_cycles=clock_now();

    double reduceTime = (end_cycles - start_cycles)/clock_frequency;

    printf("Reduce time: %f\n", reduceTime);
    }

    free(array);

    MPI_Finalize();
    return 0;
}