/**
* CS3210 - AY201718 - Assignment 2: Football match parallel using MPI
*
* ================================ NOTE ================================
* Communicator ids:
* 0 -> 11: Communicator to each sub field
* 12 : Communicator among all sub fields
* 13 : Communicator for field 0 gather all infor of players each round
* 
* For processes:
* 0 -> 11: Field processes.
* 12 -> 22: Team A players
* 23 -> 33: Team B players
* ======================================================================
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
const int SUB_FIELD_COUNT_X = 4;
const int SUB_FIELD_COUNT_Y = 3;
const int NUM_FIELD = 12;
const int NUM_PROC = 34;
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
	x[0] = (rand() % (field_width));
	x[1] = (rand() % (field_height));
}

void reverse_position_in_field(int* x) {
    x[0] = FIELD_WIDTH - 1 - x[0];
    x[1] = FIELD_HEIGHT - 1 - x[1];
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
* Init field communcication ranks
*/
int* get_all_field_comm_ranks() {
    int i;
    int* ranks = malloc_array(NUM_FIELD);
    for (i = 0; i < NUM_FIELD; i++) {
        ranks[i] = i;
    }
    return ranks;
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
* Get target goal of a player, depend on the team and the half
* true is left goal, false is right goal
*/
bool get_target_goal(int rank) {
    return (
        (is_team_A_player_rank(rank) && current_half ==  2)
        || (is_team_B_player_rank(rank) && current_half == 1)
    );
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
    int x_index = position[0] / SUB_FIELD_SIZE;
    int y_index = position[1] / SUB_FIELD_SIZE;
    return y_index * SUB_FIELD_COUNT_X + x_index;
}

/**
* Check whether the player with the particular kick power
* can score from a particular position or not.
* target_goal is true if the target goal is on the left, otherwise it is the right one.
*/
bool can_player_score_from_position(int* position, bool target_goal, int kick_power) {
    int target_goal_x = target_goal ? 0 : FIELD_WIDTH - 1;
    int goal[2] = {0, 0};
    int i;
    for (i = GOAL_Y_POSITION[0]; i <= GOAL_Y_POSITION[1]; i++) {
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
* Forward a player/ball from a position to target_position
* if used to run player, max_run is player speed
* if used to kick the ball, max_run is 2 * player kick power
*/
void forward_position(int* cur_position, int* target_position, int max_run) {
    if (get_distance(cur_position, target_position) <= max_run) {
        assign_position(cur_position, target_position);
        return;
    }
    int dis_x = target_position[0] - cur_position[0];
    int dis_y = target_position[1] - cur_position[1];
    int max_step_x = min(max_run, abs(dis_x));
	int max_step_y = min(max_run, abs(dis_y));

	int step_x = rand() % (max_step_x + 1);
	int step_y = max_run - step_x;
	while (step_y > max_step_y) {
		step_y--;
		step_x++;
	}
	if (dis_x != 0) {
		cur_position[0] += step_x * dis_x / abs(dis_x);
	}
	if (dis_y != 0) {
		cur_position[1] += step_y * dis_y / abs(dis_y);
	}
}

/**
* Make player run forward the ball direction randomly 
*/
void player_follow_ball(int* position, int* ball_position, int* attrs) {
    int max_run = min(MAX_RUN, attrs[0]);
	forward_position(position, ball_position, max_run);
}

/**
* Kick the ball toward the goal - To simplify, just target the ball into 
*   center of the goal
*/
void kick_ball_toward_goal(int* position, bool target_goal, int kick_power) {
    int* goal_center = malloc_array(2);

    goal_center[0] = target_goal ? 0 : FIELD_WIDTH - 1;
    goal_center[1] = (GOAL_Y_POSITION[0] + GOAL_Y_POSITION[1]) / 2;
    forward_position(position, goal_center, 2 * kick_power);

    free(goal_center);
}

/**
* Get player new position, depend on stretagy
* TODO: implement this method
*/
void get_player_new_position(int rank, int* position, int* ball_position, int* attrs) {
    player_follow_ball(position, ball_position, attrs);
}

/**
* Player decide to kick the ball
*/
void player_kick_the_ball(int rank, int* ball_position, int* attrs, int* is_just_scored) {
    if (!is_player_rank(rank)) {
        return;
    }
    *is_just_scored = 0;
    int kick_power = attrs[2];
    bool target_goal = get_target_goal(rank);
    if (can_player_score_from_position(ball_position, target_goal, kick_power)) {
        // If this player can score, just score, and we dont need to care about new ball position
        *is_just_scored = 1;
        return;
    }
    // TODO: decide position using strategy, For now just kick the ball forward the goal
    kick_ball_toward_goal(ball_position, target_goal, kick_power);
}

/**
* Collect players position for all fields
*/
void collect_players_position(
    MPI_Comm* field_comm, int rank,
    int* player_position, int* player_position_buf,
    int** players_position, int** players_position_buf
) {
    bool is_player = is_player_rank(rank);
    int field_id = get_sub_field_index(player_position);
    if (is_player) {
        MPI_Comm_split(MPI_COMM_WORLD, field_id, rank, field_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, rank, rank, field_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_player) {
        MPI_Gather(player_position_buf, 3, MPI_INT, &players_position_buf[0][0], 3, MPI_INT, 0, *field_comm);
    } else {
        assign_position(player_position_buf, player_position);
        player_position_buf[2] = rank;
        MPI_Gather(player_position_buf, 3, MPI_INT, NULL, 3, MPI_INT, 0, *field_comm);
    }

    if (!is_player) {
        int i, size;
        MPI_Comm_size(*field_comm, &size);
        clear_players_position(players_position);
        for (i = 1; i < size; i++) {
            int player_id = players_position_buf[i][2] - NUM_FIELD;
            assign_position(players_position[player_id], players_position_buf[i]);
        }
    }

    MPI_Barrier(*field_comm);

    MPI_Comm_free(field_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
* Gather the ball challenge. Note that this code handle both players and field
*/
void gather_ball_challenge(
    MPI_Comm* field_comm, int rank, int* attrs,
    int* player_position, int* ball_position,
    int* ball_challenge, int** ball_challenges, int* winner
) {
    bool is_player = is_player_rank(rank);
    int field_id = get_sub_field_index(ball_position);
    bool is_join = false;
    if (is_player) {
        is_join = is_same(ball_position, player_position);
        int color = is_join ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, field_comm);
    } else {
        is_join = rank == field_id;
        int color = is_join ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, field_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (is_join) {
        if (is_player) {
            ball_challenge[0] = rank;
            ball_challenge[1] = attrs[1] * (rand() % 10 + 1);
            MPI_Gather(ball_challenge, 2, MPI_INT, NULL, 2, MPI_INT, 0, *field_comm); 
        } else {
            MPI_Gather(ball_challenge, 2, MPI_INT, &ball_challenges[0][0], 2, MPI_INT, 0, *field_comm); 
        }
    }

    *winner = -1;

    if (rank == field_id) {
        int i, size;
        int max_challenge = -1;
        int max_weight = -1; // weight is for tie break using random weight
        MPI_Comm_size(*field_comm, &size);
        for (i = 1; i < size ; i++) {
            int player_id = ball_challenges[i][0];
            int challenge = ball_challenges[i][1];
            int weight = rand();
            if (challenge > max_challenge || (challenge == max_challenge && weight > max_weight)) {
                *winner = player_id;
                max_weight = weight;
                max_challenge = challenge;
            }
        }
    }

    MPI_Barrier(*field_comm);
    
    MPI_Comm_free(field_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
* Broadcast the ball winner from the field contains the ball to all processes
*/
void broadcast_ball_winner(int* ball_position, int* winner) {
    int bcast_rank = get_sub_field_index(ball_position); // rank of the field has ball
    MPI_Bcast(winner, 1, MPI_INT, bcast_rank, MPI_COMM_WORLD);
}

/**
* The winner kick the ball and bcast new ball position to all other processes
*/
void kick_ball_and_broadcast_ball_position(int rank, int winner, int* attrs, int* ball_position, int* is_just_scored) {
    if (!is_player_rank(winner)) {
        return;
    }
    if (winner == rank) {
        player_kick_the_ball(rank, ball_position, attrs, is_just_scored);
    }
    MPI_Bcast(ball_position, 2, MPI_INT, winner, MPI_COMM_WORLD);
}

/**
* Broadcast whether the ball winner just scored in this round of not
*/
void broadcast_score_check(int rank, int winner, int* is_just_scored, int* score) {
    if (winner < 0)
        return;
    MPI_Bcast(is_just_scored, 1, MPI_INT, winner, MPI_COMM_WORLD);
    // Update score at process 0
    if (rank == 0 && is_just_scored) {
        if (is_team_A_player_rank(winner))
            score[0]++;
        else
            score[1]++;
    }
}

/**
* Gather player info from process field 0
*/
void gather_players_info(
    int round_cnt, MPI_Comm* field_comm, int rank, int winner, int ball_challenge,
    int* pre_ball_position, int* player_position, int* pre_player_position,
    int* player_info, int** players_info, int* score
) {
    bool is_join = false;
    if (is_player_rank(rank)) {
        is_join = true;
        player_info[0] = pre_player_position[0];
        player_info[1] = pre_player_position[1];
        player_info[2] = player_position[0];
        player_info[3] = player_position[1];
        player_info[4] = (int)is_same(player_position, pre_ball_position);
        player_info[5] = (int)(winner == rank);
        player_info[6] = ball_challenge;
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, field_comm);
    } else {
        is_join = rank == 0;
        int color = is_join ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, field_comm);
    }
    
    if (is_join) {
        MPI_Gather(player_info, 7, MPI_INT, &players_info[0][0], 7, MPI_INT, 0, *field_comm);
    }

    if (rank == 0) {
        int i;
        printf("Round %d\n", round_cnt);
        printf("Ball position: %d %d\n", pre_ball_position[0], pre_ball_position[1]);
        for (i = 1; i <= 2 * TEAM_PLAYER; i++) {
            if (i == 1)
                printf("Team A player info:\n");
            if (i == TEAM_PLAYER + 1)
                printf("Team B player info:\n");
            printf(
                "%d %d %d %d %d %d %d\n",
                players_info[i][0], players_info[i][1],
                players_info[i][2], players_info[i][3],
                players_info[i][4], players_info[i][5],
                players_info[i][7]
            );
        }
        printf("Score: %d %d\n", score[0], score[1]);
        printf("--------------------------");
    }

    MPI_Barrier(*field_comm);

    MPI_Comm_free(field_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int numtasks, rank;

    MPI_Group orig_group, all_field_group;
    MPI_Comm all_field_comm; // Communicator for all sub fields
    MPI_Comm field_comm; // for other players to communicate with this sub field
    int** players_position = malloc_2d_array(TEAM_PLAYER * 2, 2);
    int** players_position_buf = NULL;
    int** ball_challenges = NULL;
    // int** all_gathered_players_position = NULL; // For process 0 only
    int** players_info = NULL;

    int ball_position[2] = {0, 0};
    int player_position[2] = {0, 0};
    int player_position_buf[3] = {0, 0, 0};
    int attributes[3] = {0, 0, 0};
    int* player_info = malloc_array(7);

    int pre_ball_position[2] = {0, 0};
    int pre_player_position[2] = {0, 0};
    int ball_challenge[2] = {0, 0}; // First value is the process index, second value is the ball challenge

    int score[2] = {0, 0}; // Score of the match, initialy is 0 - 0
    int is_just_scored = 0;  // To check after each round is there any just scored.
    int ball_winner = -1; // Ball winner player process id in each round

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

    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    if (is_field_rank(rank)) {
        // Setup comm for sub field
        // MPI_Group_incl(orig_group, NUM_FIELD, get_all_field_comm_ranks(), &all_field_group);
        // MPI_Comm_create(MPI_COMM_WORLD, all_field_group, &all_field_comm);

        players_position_buf = malloc_2d_array(TEAM_PLAYER * 2 + 1, 3);
        ball_challenges = malloc_2d_array(TEAM_PLAYER * 2 + 1, 2);
        if (rank == 0) {
            players_info = malloc_2d_array(NUM_PROC, 7);
            // all_gathered_players_position = malloc_2d_array(NUM_FIELD, TEAM_PLAYER * 4);
        }
    } else if (is_player_rank(rank)) {
        // Set attributes to this players
        set_attributes(rank, attributes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Start the match

    int round_cnt = 0;
    bool is_player = is_player_rank(rank);
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
        is_just_scored = false;

        // TODO: get all players position to field 0 and bcast to all players.

        // Broadcast ball position from field 0 to all other processes
        MPI_Bcast(ball_position, 2, MPI_INT, 0, MPI_COMM_WORLD);
        assign_position(pre_ball_position, ball_position);

        // Player run to new posiiton
        if (is_player) {
            assign_position(pre_player_position, player_position);
            get_player_new_position(rank, player_position, ball_position, attributes);
        }

        collect_players_position(
            &field_comm, rank, player_position, player_position_buf,
            players_position, players_position_buf
        );

        ball_challenge[1] = -1;

        gather_ball_challenge(
            &field_comm, rank, attributes, player_position, ball_position,
            ball_challenge, ball_challenges, &ball_winner
        );

        broadcast_ball_winner(ball_position, &ball_winner);

        kick_ball_and_broadcast_ball_position(rank, ball_winner, attributes, ball_position, &is_just_scored);

        broadcast_score_check(rank, ball_winner, &is_just_scored, score);

        gather_players_info(
            round_cnt, &field_comm, rank, ball_winner, ball_challenge[1], pre_ball_position,
            player_position, pre_player_position, player_info, players_info, score
        );

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    return 0;
}
 
 