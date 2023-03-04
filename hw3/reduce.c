#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clockcycle.h"
#include <mpi.h>

int MPI_P2P_Reduce(long long *sendbuf, long long *recvbuf, int count, MPI_Datatype datatype, MPI_Comm comm) {
    MPI_Request request;
    MPI_Status status;
    int rank;
    MPI_Comm_rank(comm,&rank);
    int size;
    MPI_Comm_size(comm,&size);
    int step = 2;

    while(step <= size) {
        if(rank%step == step/2) {
            MPI_Isend(&sendbuf,count,datatype,rank - step/2,1,comm,&request);
            MPI_WAIT(&request, &status);
        }
        else if(rank%step == 0) {
            MPI_Irecv(&recvbuf,count,datatype,rank + step/2,1,comm,&request);
            MPI_WAIT(&request, &status);
            *sendbuf += *recvbuf;
        }
        step *= 2;
    }

    if(rank == 0) {
        *recvbuf = *sendbuf;
    }

    MPI_Request_free(&request);

    return MPI_SUCCESS;
}

int main(int argc, char** argv) {
    int rank;
    int size;
    long long p2pSum = 0;
    long long p2ptotalSum = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int array_size = 1073741824/size;

    for(int i = 0; i < array_size; i++) {
        p2pSum += (long long) (rank*array_size+i);
    }

    long long regularSum = p2pSum;
    long long regulartotalSum = 0;

    auto double start_p2p = clock_now();
    MPI_P2P_Reduce(&p2pSum,&p2ptotalSum,sizeof(long long),MPI_LONG_LONG,MPI_COMM_WORLD);
    auto double end_p2p = clock_now();

    if(rank == 0) {
        auto double start_reduce = clock_now();
        MPI_Reduce(&regularSum, &regulartotalSum, sizeof(long long), MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        auto double end_reduce = clock_now();

        double time_p2p = (double)((end_p2p - start_p2p) / 512000000);
        double time_regular = (double)((end_reduce - start_reduce) / 512000000);

        printf("Total p2p seconds: %f", time_p2p);
        printf("Total regular reduce seconds: %f", time_regular);

        printf("Total p2p sum: %llu", p2ptotalSum);
        printf("Total regular reduce sum: %llu", regulartotalSum);
    }
    
    MPI_Finalize();
    return 0;
}