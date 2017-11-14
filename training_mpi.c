/**
 *  * CS3210 - Non-blocking communication in MPI.
 *   */

#include "mpi.h"
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#define NUM_PROC 12
#define NUM_PLAYER 11
#define FIELD_WIDTH 128
#define FIELD_HEIGHT 64
#define FIELD_RANK 11

struct Player {
	int x, y;
	int reached_ball_cnt;
	int kicked_ball_cnt;
};

void get_random_position(int* x) {
	x[0] = (rand() % (FIELD_WIDTH + 1));
	x[1] = (rand() % (FIELD_HEIGHT + 1));
}

int** malloc_players_position(int num_player) {
	int** players_position = (int **)malloc(sizeof(int*) * num_player); 
	int i;
	for (i = 0; i < num_player; i++) {
		players_position[i] = (int*)malloc(sizeof(int) * 2);
	}
	return players_position;
}

int** init_players_position(int num_player) {
	int** players_position = malloc_players_position(num_player);
	int i;
	for (i = 0; i < num_player; i++) {
		get_random_position(players_position[i]);
	}
	return players_position;
}

int main(int argc, char *argv[])
{
    int numtasks, rank, tag = 0;
	MPI_Request* reqs = NULL;
	int** players_position = NULL;
	struct Player player;

	int position[2] = {0, 0};
	int ball_position[2] = {0, 0};
	MPI_Request req;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (numtasks != NUM_PROC) {
		printf("Must specify %d processors. Terminating.\n", NUM_PROC);
    	MPI_Finalize();
		return 0;
	}
	
	srand(time(NULL));

	// Init all players and ball position
	if (rank == FIELD_RANK) {
		reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * NUM_PLAYER);
		// stats = (MPI_Status*)(sizeof(MPI_Status) * NUM_PLAYER);
		players_position = init_players_position(NUM_PLAYER);
		get_random_position(ball_position);
		int i;
		for (i = 0; i < NUM_PLAYER; i++) {
			MPI_Isend(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
		}

		MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
		printf("Process rank %d finish init player position:\n", rank);
		for (i = 0; i < NUM_PLAYER; i++) {
			printf("Player %d: %d %d\n", i, players_position[i][0], players_position[i][1]);
		}
	} else {	
		MPI_Recv(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		player.x = position[0];
		player.y = position[1];
		player.reached_ball_cnt = 0;
		player.kicked_ball_cnt = 0;
		printf("Process rank %d receive position of the ball: %d %d\n", rank, player.x, player.y);
	}	

	MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    
    return 0;
}

