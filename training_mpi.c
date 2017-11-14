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
#define MAX_RUN 10
#define swap(a, b) {typeof(x) _tmp = x; x = y; y = _tmp;}

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

void send_position_to_player(MPI_Request* reqs, int* position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Isend(position, 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
} 

void receive_players_position(MPI_Request* reqs, int** players_position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Isend(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
}

void receive_position(int* position, int tag) {
	MPI_Recv(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void send_position(int* position, int tag) {
	MPI_Send(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD);
}

void get_player_new_position(int* position, int* ball_position) {
	int dis_x = ball_position[0] - cur_position[0];
	int dis_y = ball_position[1] - cur_position[1];
	if (abs(dis_x) + abs(dis_y) <= MAX_RUN) {
		position[0] = ball_position[0];
		position[1] = ball_position[1];
		return;
	}
	// always run horizontally first
	int remaining_run = MAX_RUN;
	if (dis_x != 0) {
		int step = min(remaining_run, abs(dis_x));
		remaining_run -= step;
		position[0] += step * dis_x / abs(dis_x);
	}
	if (dis_y != 0) {
		int step = min(remaining_run, abs(dis_y));
		remaining_run -= step;
		position[1] += step * dis_y / abs(dis_y);
	}
}

// get ball winner, if more than one player get the ball, chooes the one with highest weight (random value)
int get_ball_winner(int** players_position, int* ball_position) {
	int max_weight = -1;
	int winner = -1;
	int i;
	for (i = 0; i < NUM_PLAYER; i++) if (
		players_position[i][0] == ball_position[0] 
		&& players_position[i][1] == ball_position[1]
	) {
		int random_weight = rand();
		if (winner < 0 || max_weight < random_weight) {
			max_weight = random_weight;
			winner = i;
		}
	}
	return winner;
}

int main(int argc, char *argv[])
{
    int numtasks, rank, tag = 0;
	MPI_Request* reqs = NULL;
	int** players_position = NULL;
	int** pre_players_position = NULL;

	int buf[2] = {0, 0};
	int ball_position[2] = {0, 0};
	int player_position[2] = {0, 0};
	int kicked_ball_cnt = 0;
	int reached_ball_cnt = 0;
	// MPI_Request req;
    
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
		pre_players_position = malloc_player_position(NUM_PlAYER);
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
		receive_position(player_position, tag);
		reached_ball_cnt = 0;
		kicked_ball_cnt = 0;
		printf("Process rank %d receive position of the ball: %d %d\n", rank, player.x, player.y);
	}	

	MPI_Barrier(MPI_COMM_WORLD);

	int round_cnt = 10;
	if (rank == FIELD_RANK) {
		printf("Start traning...");
	}

	while (round_cnt > 0) {
		round_cnt--;
		if (rank == FIELD_RANK) {
			send_position_to_players(reqs, ball_position, tag);
			swap(pre_players_position, players_position);
			receive_players_position(reqs, players_position, tag);
		} else {
			receive_position(ball_position, tag);
			get_player_new_position(ball_position, player_position);
			send_position(player_position);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}


    MPI_Finalize();
    
    return 0;
}

