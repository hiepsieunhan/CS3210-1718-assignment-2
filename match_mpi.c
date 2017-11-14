/**
 ** CS3210 - AY201718 - Assignment 2: Football match parallel using MPI
 **/

 #include "mpi.h"
 #include <stdlib.h>
 #include <assert.h>
 #include <stdio.h>
 #include <stdbool.h>
 #include <time.h>
 
 #define min(a, b) (a < b ? a : b)
 
 const int NUM_PROC = 34;
 const int TEAM_PLAYER = 11;
 const int FIELD_WIDTH = 128;
 const int FIELD_HEIGHT = 96;
 const int MAX_RUN = 10;
 const int TOTAL_ROUND = 2700;
 const int GOAL_Y_POSITION[2] = {43, 51};
 
 int main(int argc, char *argv[])
 {
     int numtasks, rank, tag = 0;

     MPI_Init(&argc,&argv);
     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
     if (numtasks != NUM_PROC) {
         printf("Must specify %d processors. Terminating.\n", NUM_PROC);
         MPI_Finalize();
         return 0;
     }
 
     srand(time(NULL) * rank);
 
     MPI_Finalize();
     
     return 0;
 }
 
 