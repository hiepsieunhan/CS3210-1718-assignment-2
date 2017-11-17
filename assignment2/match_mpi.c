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
const int SUB_FIELD_SIZE = 32;
const int SUB_FIELD_WIDTH = FIELD_WIDTH / SUB_FIELD_SIZE;
const int SUB_FIELD_HEIGHT = FIELD_HEIGHT / SUB_FIELD_SIZE;
const int NUM_FIELD = SUB_FIELD_HEIGHT * SUB_FIELD_WIDTH;
const int MAX_RUN = 10;
const int TOTAL_ROUND = 2700;
const int GOAL_Y_POSITION[2] = {43, 51};

// Each player is allowed to access his own attributes from here.
// Attributes are speed, dribbling and kick power in order
const int PLAYER_ATTRIBUTES[22][3] = {
    // For team 1
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
    {5, 5, 5},
    // For team 2
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
}

// As we all use the same tag, just make it as global
int tag = 0;

/**
*   Get random position in the field of size (field_width, field_height)
*/
void get_random_position(int field_width, int field_height, int* x) {
	x[0] = (rand() % (field_width ));
	x[1] = (rand() % (field_height));
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
bool is_rank_field(int rank) {
    return 0 <= rank && rank < NUM_FIELD;
}

/**
* Check if the rank represent the player in team A (from 12 -> 22)
*/
bool is_team_a_player(int rank) {
    return NUM_FIELD <= rank && rank < NUM_FIELD + TEAM_PLAYER;
}

/**
* Check if the rank represent the player in team B (from 23 -> 33)
*/
bool is_team_b_player(int rank) {
    return NUM_FIELD + TEAM_PLAYER <= rank && rank < NUM_FIELD + 2 * TEAM_PLAYER;
}

/**
* Assign the l_pos values with r_pos values
*/
void assign_position(int* l_pos, int* r_pos) {
	l_pos[0] = r_pos[0];
	l_pos[1] = r_pos[1];
}

/**
* Get mahatan distance between 2 positions
*/
int get_distance(int* l_pos, int* r_pos) {
    return abs(l_post[0] - r_pos[0]) + abs(l_pos[1] - r_pos[1]);
}

/**
* Get the initial ball position
*/
void get_initial_ball_position(int* x) {
    x[0] = FIELD_WIDTH / 2;
    x[1] = FIELD_HEIGHT / 2;
}

/**
* Get the subfield index that the player belongs to
*/
int get_player_sub_field(int* position) {
    int row_index = position[0] / SUB_FIELD_SIZE;
    int col_index = position[1] / SUB_FIELD_SIZE;
    return row_index * SUB_FIELD_WIDTH + col_index;
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

// This method create 2d array of dimension length x size
int** malloc_2d_array(int length, int size) {
	int** array = (int **)malloc(sizeof(int*) * length); 
	int i;
	for (i = 0; i < length; i++) {
		array[i] = (int*)malloc(sizeof(int) * size);
	}
	return array;
}

int* malloc_array(int size) {
    int* array = (int*)malloc(sizeof(int) * size);
    return array;
}


int main(int argc, char *argv[])
{
    int numtasks, rank, tag = 0;

    MPI_Comm field;

    int* ball_location = {0, 0};
    int* player_location = {0, 0};
    int* attributes = {0, 0, 0};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numtasks != NUM_PROC) {
        printf("Must specify %d processors. Terminating.\n", NUM_PROC);
        MPI_Finalize();
        return 0;
    }

    srand(time(NULL) * rank);

    if ()

    MPI_Finalize();
    
    return 0;
}
 
 