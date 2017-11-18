/**
* CS3210 - Custom communication in MPI.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define NPROCS 8

int main(int argc, char *argv[])
{
    int rank, new_rank, sendbuf, recvbuf, numtasks, ranks1[4]={0,1,2,3}, ranks2[5]={0, 4,5,6,7};
    MPI_Group  orig_group, new_group, other_group;
    MPI_Comm   new_comm, other_comm, field_comm;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks != NPROCS) {
        printf("Must specify MP_PROCS= %d. Terminating.\n",NPROCS);
        MPI_Finalize();
        exit(0);
    }

    sendbuf = rank;

    /* Extract the original group handle */
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (rank < NPROCS/2) {
        MPI_Group_incl(orig_group, NPROCS/2, ranks1, &new_group);
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

        MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &field_comm);
    }
    if (rank == 0 || rank >= NPROCS/2) {
        MPI_Group_incl(orig_group, NPROCS/2 + 1, ranks2, &other_group);
        MPI_Comm_create(MPI_COMM_WORLD, new_group, &other_comm);
    }  

    /* Create new new communicator and then perform collective communications */
    MPI_Finalize();
    
    return 0;
}