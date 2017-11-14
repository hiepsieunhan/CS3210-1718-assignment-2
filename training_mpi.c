/**
 *  * CS3210 - Non-blocking communication in MPI.
 *   */

#include "mpi.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

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

void get_random_position(int* x) {
	x[0] = (rand() % (FIELD_WIDTH + 1));
	x[1] = (rand() % (FIELD_HEIGHT + 1));
}

bool is_same(int* l_post, int* r_post) {
	return l_post[0] == r_post[0] && l_post[1] == r_post[1];
}

int** malloc_players_position(int num_player) {
	int** players_position = (int **)malloc(sizeof(int*) * num_player); 
	int i;
	for (i = 0; i < num_player; i++) {
		players_position[i] = (int*)malloc(sizeof(int) * 2);
	}
	return players_position;
}

PlayerInfo* malloc_players_info(int num_player) {
	PlayerInfo* players_info = (PlayerInfo*)malloc(sizeof(PlayerInfo) * num_player);
	int i;
	for (i = 0; i < num_player; i++) {
		players_info[i].kicked_ball_cnt = 0;
		players_info[i].reached_ball_cnt = 0;
		players_info[i].total_meters_run = 0;
	}
	return players_info;
}

int** init_players_position(int num_player) {
	int** players_position = malloc_players_position(num_player);
	int i;
	for (i = 0; i < num_player; i++) {
		get_random_position(players_position[i]);
	}
	return players_position;
}

void send_position_to_players(MPI_Request* reqs, int* position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Isend(position, 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
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

void receive_ball_winner_from_field(int* winner, int tag) {
	MPI_Recv(winner, 1, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void receive_position_from_players(MPI_Request* reqs, int** players_position, int tag) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		MPI_Irecv(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
	}
	MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
}

void receive_ball_position_from_player(int winner, int* ball_position, int tag) {
	MPI_Recv(ball_position, 2, MPI_INT, winner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void receive_position_from_field(int* position, int tag) {
	MPI_Recv(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void send_position_to_field(int* position, int tag) {
	MPI_Send(position, 2, MPI_INT, FIELD_RANK, tag, MPI_COMM_WORLD);
}

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

void update_players_info(PlayerInfo* players_info, int** pre_players_position, int** players_position,
						int* ball_position, int winner) {
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		players_info[i].total_meters_run +=  get_distance(pre_players_position[i], players_position[i]);
		if (is_same(players_position[i], ball_position)) {
			players_info[i].reached_ball_cnt++;
		}
		if (i == winner) {
			players_info[i].kicked_ball_cnt++;
		}
	}
}

void print_players_info(int round, PlayerInfo* players_info,
						int** pre_players_position, int** players_position,
						int* ball_position, int winner) {
	printf("%d\n", round);
	printf("%d %d\n", ball_position[0], ball_position[1]);
	int i;
	for (i = 0; i < NUM_PLAYER; i++) {
		bool is_reached_ball = is_same(players_position[i], ball_position);
		bool is_won_ball = (winner == i);
		printf("%d %d %d %d %d %d %d %d %d %d\n", i, pre_players_position[i][0], pre_players_position[i][1],
												players_position[i][0], players_position[i][1],
												(int)is_reached_ball, (int)is_won_ball,
												players_info[i].total_meters_run,
												players_info[i].reached_ball_cnt, players_info[i].kicked_ball_cnt);
	}
	printf("-----------------\n");
}

void print_final_player_info(int rank, int* player_position, int *pre_player_position, PlayerInfo player_info) {
	printf("---Player %d ----Position: (%d, %d) -> (%d, %d) --- meter_run :%d ---- reached ball: %d --- kicked ball %d\n",
		rank, pre_player_position[0], pre_player_position[1], player_position[0], player_position[1],
		player_info.total_meters_run, player_info.reached_ball_cnt, player_info.kicked_ball_cnt);
}

int main(int argc, char *argv[])
{
    int numtasks, rank, tag = 0;
	MPI_Request* reqs = NULL;
	int** players_position = NULL;
	int** pre_players_position = NULL;
	PlayerInfo* players_info = NULL;

	int ball_position[2] = {0, 0};
	int pre_player_position[2] = {0, 0};
	int player_position[2] = {0, 0};
	PlayerInfo player_info;
	// MPI_Request req;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (numtasks != NUM_PROC) {
		printf("Must specify %d processors. Terminating.\n", NUM_PROC);
    	MPI_Finalize();
		return 0;
	}

	srand(rank);

	// Init all players and ball position
	if (rank == FIELD_RANK) {
		reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * NUM_PLAYER);
		// stats = (MPI_Status*)(sizeof(MPI_Status) * NUM_PLAYER);
		players_position = init_players_position(NUM_PLAYER);
		pre_players_position = malloc_players_position(NUM_PLAYER);
		players_info = malloc_players_info(NUM_PLAYER);
		get_random_position(ball_position);
		int i;
		for (i = 0; i < NUM_PLAYER; i++) {
			MPI_Isend(players_position[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
		}

		MPI_Waitall(NUM_PLAYER, reqs, MPI_STATUSES_IGNORE);
		// printf("Process rank %d finish init player position:\n", rank);
		// for (i = 0; i < NUM_PLAYER; i++) {
		// 	printf("Player %d: %d %d\n", i, players_position[i][0], players_position[i][1]);
		// }
		// printf("Initial ball position: %d %d\n", ball_position[0], ball_position[1]);
	} else {	
		receive_position_from_field(player_position, tag);
		player_info.reached_ball_cnt = 0;
		player_info.kicked_ball_cnt = 0;
		player_info.total_meters_run = 0;
		// printf("Process rank %d receive position of the ball: %d %d\n", rank, player_position[0], player_position[1]);
	}	

	MPI_Barrier(MPI_COMM_WORLD);

	int round_cnt = 0;
	if (rank == FIELD_RANK) {
		// printf("Start traning...\n");
	}

	while (round_cnt < TOTAL_ROUND) {
		round_cnt++;
		if (rank == FIELD_RANK) {
			send_position_to_players(reqs, ball_position, tag);
			// swap
			{
				int** tmp = players_position;
				players_position = pre_players_position;
				pre_players_position = tmp;
			}
			receive_position_from_players(reqs, players_position, tag);
			int winner = get_ball_winner(players_position, ball_position);
			update_players_info(players_info, pre_players_position, players_position, ball_position, winner);
			if (winner >= 0) {
				send_ball_winner_to_players(reqs, players_position, ball_position, winner, tag);
				receive_ball_position_from_player(winner, ball_position, tag);
			}
			print_players_info(round_cnt, players_info, pre_players_position, players_position, ball_position, winner);
		} else {
			receive_position_from_field(ball_position, tag);
			pre_player_position[0] = player_position[0];
			pre_player_position[1] = player_position[1];
			get_player_new_position(player_position, ball_position);
			player_info.total_meters_run += get_distance(pre_player_position, player_position);
			send_position_to_field(player_position, tag);
			if (is_same(ball_position, player_position)) {
				player_info.reached_ball_cnt++;
				int winner;
				receive_ball_winner_from_field(&winner, tag);
				if (winner == rank) {
					player_info.kicked_ball_cnt++;
					get_random_position(ball_position);
					send_position_to_field(ball_position, tag);
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// if (rank != FIELD_RANK) {
	// 	print_final_player_info(rank, player_position, pre_player_position, player_info);
	// }

    MPI_Finalize();
    
    return 0;
}

