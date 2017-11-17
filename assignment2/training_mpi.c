/**
 ** CS3210 - AY201718 - Assignment 2: Traning football parallel using MPI
 **/

#include "mpi.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define min(a, b) (a < b ? a : b)

const int NUM_PROC = 12;
const int NUM_PLAYER = 11;
const int FIELD_WIDTH = 128;
const int FIELD_HEIGHT = 64;
const int FIELD_RANK = 11;
const int MAX_RUN = 10;
const int TOTAL_ROUND = 900;

typedef struct {
	int kicked_ball_cnt;
	int reached_ball_cnt;
	int total_meters_run;
} PlayerInfo;

// This method produce random position in the field
void get_random_position(int* x) {
	x[0] = (rand() % (FIELD_WIDTH));
	x[1] = (rand() % (FIELD_HEIGHT));
}

// This method compare 2 position and return true if they are the same
bool is_same(int* l_post, int* r_post) {
	return l_post[0] == r_post[0] && l_post[1] == r_post[1];
}

// This assign the first position to the second one
void assign_position(int* l_pos, int* r_pos) {
	l_pos[0] = r_pos[0];
	l_pos[1] = r_pos[1];
}

// This method create 2d array of dimension length x size
int** malloc_2d_array(int length, int size) {
	int** array = (int **)malloc(sizeof(int*) * length); 
	int i;
	for (i = 0; i < length; i++) {
		array[i] = (int*)malloc(sizeof(int) * size);
	}
	return array;
}

// This method initialize all players position randomly
int** init_players_position(int num_player) {
	int** players_position = malloc_2d_array(num_player, 2);
	int i;
	for (i = 0; i < num_player; i++) {
		get_random_position(players_position[i]);
	}
	return players_position;
}

// Broadcast a position from field to all players
void send_position_to_players(MPI_Request* reqs, int* position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Isend(position, 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
} 

// Send position of players to the corresponding process
void send_player_position_to_players(MPI_Request* reqs, int** players_position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Isend(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
}

// send the ball winner to all the players who reached the ball
void send_ball_winner_to_players(MPI_Request* reqs, int** players_position, int* ball_position, int winner, int tag) {
	if (winner < 0 || winner >= NUM_PLAYER) {
		return;
	}
	int i;
	int req_cnt = 0;
	for (i = 0; i < NUM_PLAYER; i++) if (is_same(players_position[i], ball_position)) {
		MPI_Isend(&winner, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[req_cnt]);
		req_cnt++;
	}
	MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);
} 

// Get the ball winner player from the field process
void receive_ball_winner_from_field(int* winner, int tag) {
	MPI_Recv(winner, 1, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Get all players position from all player processes
void receive_position_from_players(MPI_Request* reqs, int** players_position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Irecv(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
}

// Get the new ball position from the winner, after he kicked the ball
void receive_ball_position_from_player(int winner, int* ball_position, int tag) {
	MPI_Recv(ball_position, 2, MPI_INT, winner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Get the position sent by field process
void receive_position_from_field(int* position, int tag) {
	MPI_Recv(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Send some position from player process to field process
void send_position_to_field(int* position, int tag) {
	MPI_Send(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD);
}

// Send all necessary player info from player to the field process, after each round
void send_player_info_to_field(int* pre_player_position, PlayerInfo player_info, bool is_won_ball, bool is_reached_ball, int tag) {
	int* buf = (int*)malloc(sizeof(int) * 7);
	buf[0] = pre_player_position[0];
	buf[1] = pre_player_position[1];
	buf[2] = player_info.total_meters_run;
	buf[3] = player_info.reached_ball_cnt;
	buf[4] = player_info.kicked_ball_cnt;
	buf[5] = (int)is_won_ball;
	buf[6] = (int)is_reached_ball;
	MPI_Send(buf, 7, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD);
	free(buf);
}

// Collect all players information after each round
void receive_info_from_players(MPI_Request* reqs, int** players_info, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Irecv(players_info[i], 7, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
}

// Make player run forward the ball direction randomly 
void get_player_new_position(int* position, int* ball_position) {
	int dis_x = ball_position[0] - position[0];
	int dis_y = ball_position[1] - position[1];
	if (abs(dis_x) + abs(dis_y) <= MAX_RUN) {
		position[0] = ball_position[0];
		position[1] = ball_position[1];
		return;
	}
	int max_step_x = min(MAX_RUN, abs(dis_x));
	int max_step_y = min(MAX_RUN, abs(dis_y));

	int step_x = rand() % (max_step_x + 1);
	int step_y = MAX_RUN - step_x;
	while (step_y > max_step_y) {
		step_y--;
		step_x++;
	}
	if (dis_x != 0) {
		position[0] += step_x * dis_x / abs(dis_x);
	}
	if (dis_y != 0) {
		position[1] += step_y * dis_y / abs(dis_y);
	}
}

// get ball winner, if more than one player get the ball, chooes the one with highest weight (random value)
int get_ball_winner(int** players_position, int* ball_position) {
	int max_weight = -1;
	int winner = -1;
	int i;
	for (i = 0; i < NUM_PLAYER; i++) if (is_same(players_position[i], ball_position)) {
		int random_weight = rand();
		if (winner < 0 || max_weight < random_weight) {
			max_weight = random_weight;
			winner = i;
		}
	}
	return winner;
}

int get_distance(int* a, int* b) {
	return abs(a[0] - b[0]) + abs(a[1] - b[1]);
}

void print_players_info(int round, int** players_info, int** players_position, int* pre_ball_position) {
	printf("%d\n", round);
	printf("%d %d\n", pre_ball_position[0], pre_ball_position[1]);
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		printf("%d %d %d %d %d %d %d %d %d %d\n", i, players_info[i][0], players_info[i][1],
												players_position[i][0], players_position[i][1],
												players_info[i][6], players_info[i][5],
												players_info[i][2],
												players_info[i][3], players_info[i][4]);
	}
	printf("-----------------\n");
}

int main(int argc, char *argv[]) {
    int numtasks, rank, tag = 0;
	MPI_Request* reqs = NULL;
	int** players_position = NULL;
	// This is NUM_PLAYER x 7 2D array to collect players info after each round
	int** players_info = NULL;

	int ball_position[2] = {0, 0};
	int pre_ball_position[2] = {0, 0};
	int pre_player_position[2] = {0, 0};
	int player_position[2] = {0, 0};
	bool is_won_ball = false;
	bool is_reached_ball = false;
	PlayerInfo player_info;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (numtasks != NUM_PROC) {
		printf("Must specify %d processors. Terminating.\n", NUM_PROC);
    	MPI_Finalize();
		return 0;
	}

	srand(time(NULL) * rank);

	// Init all players and ball position
	if (rank == FIELD_RANK) {
		reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * NUM_PLAYER);
		players_position = init_players_position(NUM_PLAYER);
		players_info = malloc_2d_array(NUM_PLAYER, 7);
		get_random_position(ball_position);
		send_player_position_to_players(reqs, players_position, tag);
	} else {	
		receive_position_from_field(player_position, tag);
		player_info.reached_ball_cnt = 0;
		player_info.kicked_ball_cnt = 0;
		player_info.total_meters_run = 0;
	}	

	MPI_Barrier(MPI_COMM_WORLD);

	
	// Start training
	int round_cnt = 0;
	
	// TODO: move these to 2 separated methods
	while (round_cnt < TOTAL_ROUND) {
		round_cnt++;
		if (rank == FIELD_RANK) {
			send_position_to_players(reqs, ball_position, tag);
			receive_position_from_players(reqs, players_position, tag);
			assign_position(pre_ball_position, ball_position);
			int winner = get_ball_winner(players_position, ball_position);
			if (winner >= 0) {
				send_ball_winner_to_players(reqs, players_position, ball_position, winner, tag);
				receive_ball_position_from_player(winner, ball_position, tag);
			}
			receive_info_from_players(reqs, players_info, tag);
			print_players_info(round_cnt, players_info, players_position, pre_ball_position);
		} else {
			is_won_ball = false;
			is_reached_ball = false;
			receive_position_from_field(ball_position, tag);
			assign_position(pre_player_position, player_position);
			get_player_new_position(player_position, ball_position);
			send_position_to_field(player_position, tag);
			if (is_same(ball_position, player_position)) {
				is_reached_ball = true;
				int winner;
				receive_ball_winner_from_field(&winner, tag);
				if (winner == rank) {
					is_won_ball = true;
					get_random_position(ball_position);
					send_position_to_field(ball_position, tag);
				}
			}
			player_info.total_meters_run += get_distance(pre_player_position, player_position);
			if (is_won_ball) {
				player_info.kicked_ball_cnt++;
			}
			if (is_reached_ball) {
				player_info.reached_ball_cnt++;
			}
			send_player_info_to_field(pre_player_position, player_info, is_won_ball, is_reached_ball, tag);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

    MPI_Finalize();
    
    return 0;
}

