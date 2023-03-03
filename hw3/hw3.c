#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 1000000000

int MPI_P2P_Reduce(
    void* send_data,
    void* recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Comm comm){
        MPI_Status status;
        int rank;
        MPI_Comm_rank(comm,&rank);

        if(rank == 0){
            int size;
            long long sum = 0;
            MPI_Comm_size(comm,&size);
            for(int i = 0; i < size; i++){
                if(i != 0){
                    MPI_Irecv(&recv_data,count,datatype,i,0,1,comm,&status);
                    sum += recv_data;
                }
            }
            printf("Total sum: %llu", sum);
        }
        else{
            MPI_Isend(&send_data,count,datatype,0,1,comm);
        }
        return MPI_SUCCESS;
    }

int main(int argc, char** argv){
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements = ARRAY_SIZE/size;
    int sum, finalSum = 0;
    long long *array = (long long*)malloc(elements*sizeof(long long));
    for(int i = 0; i < elements; i++){
        array[i] = rank*elements+i;
    }

    for(int i = 0; i < elements; i++){
        sum += array[i];
    }
    MPI_P2P_Reduce(&sum,&finalSum,1,MPI_INT,MPI_COMM_WORLD);

    free(array);

    MPI_Finalize();
    return 0;
}