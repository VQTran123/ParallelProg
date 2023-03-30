#!/bin/sh
module load spectrum-mpi cuda/11.2
############################################################################################
# Launch N tasks per compute node allocated. Per below this launches 6 MPI rank per compute# taskset insures that hyperthreaded cores are skipped.
############################################################################################
taskset -c 0-159:4 mpirun -N 6 /gpfs/u/home/SPNR/SPNRcaro/scratch/MPI-CUDA-Reduction/reduce