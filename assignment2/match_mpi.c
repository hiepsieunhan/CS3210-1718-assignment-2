/**
 ** CS3210 - AY201718 - Assignment 2: Football match parallel using MPI
**/

/**
* ------------NOTE------------
* Communicator ids:
* 0 -> 11: Communicator to each sub field
* 12 : Communicator among all sub fields
* 13 : Communicator for field 0 gather all infor of players each round
* 
* For processes:
* 0 -> 11: Field processes.
* 12 -> 22: Team A players
* 23 -> 33: Team B players
**/


 #include "mpi.h"
 #include <stdlib.h>
 #include <assert.h>
 #include <stdio.h>
 #include <stdbool.h>
 #include <time.h>
 
#define min(a, b) (a < b ? a : b)

// Config constants so that we can easily change the number of players or the field size,...
const int TEAM_PLAYER = 11;
const int FIELD_WIDTH = 128;
const int FIELD_HEIGHT = 96;
const int SUB_FIELD_SIZE = 32;
const int SUB_FIELD_COUNT_X = FIELD_WIDTH / SUB_FIELD_SIZE;
const int SUB_FIELD_COUNT_Y = FIELD_HEIGHT / SUB_FIELD_SIZE;
const int NUM_FIELD = SUB_FIELD_COUNT_Y * SUB_FIELD_COUNT_X;
const int NUM_PROC = NUM_FIELD + TEAM_PLAYER * 2;
const int MAX_RUN = 10;
const int TOTAL_ROUND = 2700;
const int GOAL_Y_POSITION[2] = {43, 51};


/**
* Each player is allowed to access his own attributes from here.
* Attributes are speed, dribbling and kick power in order
*/
const int ATTRS_TEAM_A[11][3] = {
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5}
};
const int ATTRS_TEAM_B[11][3] = {
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5}
};

// As we all use the same tag, just make it as global
int tag = 0;

// Global value to indicate which current half is, either 1 or 2
int current_half = 1;

/**
*   Get random position in the field of size (field_width, field_height)
*/
void get_random_position(int field_width, int field_height, int* x) {
	x[0] = (rand() % (field_width ));
	x[1] = (rand() % (field_height));
}

void reverse_position_in_field(int* x) {
    x[0] = FIELD_WIDTH - x[0];
    x[1] = FIELD_HEIGHT - x[1];
}

/**
* Malloc helper method. Note that for 1D and 2D arrays, memory will be continous
*/
int** malloc_2d_array(int length, int size) {
    int** array = (int **)malloc(sizeof(int*) * length);
    int* elements = (int *)malloc(sizeof(int) * length * size);
	int i;
	for (i = 0; i < length; i++) {
		array[i] = &elements[i * size];
	}
	return array;
}

int* malloc_array(int size) {
    int* array = (int*)malloc(sizeof(int) * size);
    return array;
}

/**
* Reset all value to -1
*/
void clear_players_position(int** players_position) {
    int i, j;
    for (i = 0; i < TEAM_PLAYER * 2; i++)
        for (j = 0; j < 2; j++)
            players_position[i][j] = -1;
}

/**
* Compare 2 position in fields and return whether they are the same position or not
*/
bool is_same(int* l_post, int* r_post) {
	return l_post[0] == r_post[0] && l_post[1] == r_post[1];
}

/**
* Check if the rank rerepresent the field (0 -> 11)
*/
bool is_field_rank(int rank) {
    return 0 <= rank && rank < NUM_FIELD;
}

/**
* Check if the rank represent a player
*/
bool is_player_rank(int rank) {
    return NUM_FIELD <= rank && rank < NUM_FIELD + 2 * TEAM_PLAYER;
}

/**
* Check if the rank represent the player in team A (from 12 -> 22)
*/
bool is_team_A_player_rank(int rank) {
    return NUM_FIELD <= rank && rank < NUM_FIELD + TEAM_PLAYER;
}

/**
* Check if the rank represent the player in team B (from 23 -> 33)
*/
bool is_team_B_player_rank(int rank) {
    return NUM_FIELD + TEAM_PLAYER <= rank && rank < NUM_FIELD + 2 * TEAM_PLAYER;
}

/**
* Get team index of the player respect to team A (0 -> 10)
* return -1 if this player is not team A player
*/
int get_team_A_player_index(int rank) {
    if (!is_team_A_player_rank(rank)) {
        return -1;
    }
    return rank - NUM_FIELD;
}

/**
* Get team index of the player respect to team B (0 -> 10)
* return -1 if this player is not team B player
*/
int get_team_B_player_index(int rank) {
    if (!is_team_B_player_rank(rank)) {
        return -1;
    }
    return rank - NUM_FIELD - TEAM_PLAYER;
}

/**
* Assign the l_pos values with r_pos values
*/
void assign_position(int* l_pos, int* r_pos) {
	l_pos[0] = r_pos[0];
	l_pos[1] = r_pos[1];
}

void assign_attributes(int* l_attrs, const int* r_attrs) {
    int i;
    for (i = 0; i < 3; i++) l_attrs[i] = r_attrs[i];
}

/**
* Get mahatan distance between 2 positions
*/
int get_distance(int* l_pos, int* r_pos) {
    return abs(l_pos[0] - r_pos[0]) + abs(l_pos[1] - r_pos[1]);
}

/**
* Get the initial ball position
*/
void reset_ball_position(int* x) {
    x[0] = FIELD_WIDTH / 2;
    x[1] = FIELD_HEIGHT / 2;
}

/**
* Get the subfield index that the position belongs to
*/
int get_sub_field_index(int* position) {
    int row_index = position[0] / SUB_FIELD_SIZE;
    int col_index = position[1] / SUB_FIELD_SIZE;
    return row_index * SUB_FIELD_COUNT_X + col_index;
}

/**
* Check whether the player with the particular kick power
* can score from a particular position or not.
* target_goal is true if the target goal is on the left, otherwise it is the right one.
*/
bool can_player_score_from_position(int kick_power, int* position, bool target_goal) {
    int target_goal_x = target_goal ? 0 : FIELD_WIDTH - 1;
    int goal[2] = {0, 0};
    int i;
    for (int i = GOAL_Y_POSITION[0]; i <= GOAL_Y_POSITION[1]; i++) {
        goal[0] = target_goal_x;
        goal[1] = i;
        if (get_distance(position, goal) < kick_power * 2)
            return true;
    }
    return false;
}


/**
* Set attributes to this player process - taken from global connts.
* Note that each player is allowed to take his own attributes only.
* Accessing others players attributes must be done via communication
*/
void set_attributes(int rank, int* attributes) {
    if (is_team_A_player_rank(rank)) {
        int player_id = get_team_A_player_index(rank);
        assign_attributes(attributes, ATTRS_TEAM_A[player_id]);
    } else if (is_team_B_player_rank(rank)) {
        int player_id = get_team_B_player_index(rank);
        assign_attributes(attributes, ATTRS_TEAM_B[player_id]);
    }
}

/**
* TODO: change this to use strategy
*/
void reset_positions(int rank, int* ball_position, int* player_position) {
    reset_ball_position(ball_position);
    if (is_player_rank(rank)) {
        get_random_position(FIELD_WIDTH / 2, FIELD_HEIGHT, player_position);
        if (is_team_B_player_rank(rank)) {
            reverse_position_in_field(player_position);
        }
        if (current_half == 2) {
            reverse_position_in_field(player_position);
        }
    }
}

/**
* Collect players position for all fields
*/
void collect_players_position(
    MPI_Comm* field_comm, int rank,
    int* player_position, int* player_position_buf,
    int** players_position, int** players_position_buf
) {
    int field_id = get_sub_field_index(player_position);
    if (is_player_rank(rank)) {
        MPI_Comm_split(MPI_COMM_WORLD, field_id, rank, field_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (is_field_rank(rank)) {
        MPI_Gather(player_position_buf, 3, MPI_INT, &players_position_buf[0][0], 3, MPI_INT, 0, *field_comm);    
    } else {
        assign_position(player_position_buf, player_position);
        player_position[2] = rank;
        MPI_Gather(player_position_buf, 3, MPI_INT, NULL, 3, MPI_INT, 0, *field_comm);
    }

    if (is_field_rank(rank)) {
        int i, size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        clear_players_position(players_position);
        for (i = 0; i < size - 1; i++) {
            int player_id = players_position_buf[i][2];
            assign_position(players_position[player_id], players_position_buf[i]);
        }
    }

    MPI_Barrier(*field_comm);

    if (is_player_rank(rank)) {
        MPI_Comm_free(field_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int numtasks, rank, tag = 0;

    // TODO: create comm of all players with sub field 0 to report each round....
    MPI_Comm all_field_comm; // Communicator for all sub fields
    MPI_Comm field_comm; // for other players to communicate with this sub field
    int** players_position = NULL;
    int** players_position_buf = NULL;
    int** all_gathered_players_position = NULL; // For process 0 only

    int ball_position[2] = {0, 0};
    int player_position[2] = {0, 0};
    int player_position_buf[3] = {0, 0, 0};
    int attributes[3] = {0, 0, 0};

    int score[2] = {0, 0}; // Score of the match, initialy is 0 - 0
    int is_just_scored = 0;  // To check after each round is there any just scored.

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numtasks != NUM_PROC) {
        printf("Must specify %d processors. Terminating.\n", NUM_PROC);
        MPI_Finalize();
        return 0;
    }

    srand(time(NULL) * rank);

    // Initialization

    if (is_field_rank(rank)) {
        // Setup comm for sub field
        MPI_Comm_split(MPI_COMM_WORLD, 12, rank, &all_field_comm);
        MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &field_comm);
        players_position = malloc_2d_array(TEAM_PLAYER * 2, 2);
        players_position_buf = malloc_2d_array(TEAM_PLAYER * 2 + 1, 2);
        if (rank == 0) {
            all_gathered_players_position = malloc_2d_array(NUM_FIELD, TEAM_PLAYER * 4);
        }
    } else if (is_player_rank(rank)) {
        // Set attributes to this players
        set_attributes(rank, attributes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Start the match

    int round_cnt = 0;
    while (round_cnt < TOTAL_ROUND * 2) {
        // Move from 1st half to 2nd half.
        if (round_cnt == TOTAL_ROUND) {
            current_half = 2;
        }
        // Begin of a half of some team just scored, we reset all positions
        if (round_cnt == 0 || round_cnt == TOTAL_ROUND || is_just_scored != 0) {
            reset_positions(rank, ball_position, player_position);
            collect_players_position(
                &field_comm, rank, player_position, player_position_buf,
                players_position, players_position_buf
            );
        }
        round_cnt++;
    }

    MPI_Finalize();
    
    return 0;
}
 
 