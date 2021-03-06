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
* 
* For strategy: Team A will attack when team B will defend
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
const int NUM_PLAYER = 22;
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
    {4, 4, 7},
    {3, 5, 7},
    {2, 6, 7},
    {5, 4, 6},
    {8, 5, 2},
    {6, 2, 7},
    {7, 2, 6},
    {2, 7, 6},
    {4, 8, 3},
    {4, 5, 6},
    {6, 6, 3}
};
const int ATTRS_TEAM_B[11][3] = {
    {2, 6, 7},
    {8, 5, 2},
    {6, 5, 4},
    {2, 6, 7},
    {3, 8, 4},
    {7, 4, 4},
    {3, 5, 7},
    {2, 8, 5},
    {3, 6, 6},
    {5, 3, 7},
    {2, 7, 6}
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

int* malloc_array(int size) {
    int* array = (int*)malloc(sizeof(int) * size);
    return array;
}

int** malloc_2d_array(int length, int size) {
    int** array = (int **)malloc(sizeof(int*) * length);
    int* elements = (int *)malloc(sizeof(int) * length * size);
	int i;
	for (i = 0; i < length; i++) {
		array[i] = &elements[i * size];
	}
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
    for (i = 0; i < NUM_PLAYER; i++)
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
    return NUM_FIELD <= rank && rank < NUM_FIELD + NUM_PLAYER;
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
    return NUM_FIELD + TEAM_PLAYER <= rank && rank < NUM_FIELD + NUM_PLAYER;
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
* Check if player rank is defending or attacking
* - Defending when the ball is in their half field 
* - attacking when the ball is in the half field that contains the target goal
*/
bool is_attacking(int rank, int* ball_position) {
    int is_ball_in_left_half = ball_position[0] < FIELD_WIDTH / 2;
    bool target_goal = get_target_goal(rank);
    return is_ball_in_left_half == target_goal;
}

void set_target_goal(int rank, int* goal) {
    goal[0] = get_target_goal(rank) ? 0 : FIELD_WIDTH - 1;
    goal[1] = (GOAL_Y_POSITION[0] + GOAL_Y_POSITION[1]) / 2 + (rand() % 9 - 4);
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
* Reset all players position randomly and put the ball in the center of the field
* this method is used after a goal or in the beging of a half.
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

void player_run_toward_goal(int rank, int* position, int* attrs) {
    int max_run = attrs[0];
    int* target = malloc_array(2);
    set_target_goal(rank, target);
    forward_position(position, target, max_run);
    
    free(target);    
}

void player_go_back_to_defend(int rank, int* position, int* attrs) {
    int max_run = attrs[0];
    int* target = malloc_array(2);
    get_random_position(SUB_FIELD_SIZE, FIELD_HEIGHT, target);
    if (get_target_goal(rank)) {
        reverse_position_in_field(target);
    }
    forward_position(position, target, max_run);

    free(target);
}

void get_target_nearest_position(int rank, int** players_position, int* position) {
    int* target = malloc_array(2);
    int start_index = is_team_A_player_rank(rank) ? 0 : TEAM_PLAYER;
    int i;
    int min_distance = 10000000;
    set_target_goal(rank, target);

    for (i = 0; i < TEAM_PLAYER; i++) {
        int* cur_position = players_position[start_index + i];
        int distance = get_distance(cur_position, target);
        if (distance < min_distance) {
            assign_position(position, cur_position);
        }
    }
    free(target);
}

/**
* Kick the ball toward the goal - To simplify, just target the ball into 
*   center of the goal
*/
void kick_ball_toward_goal(int rank, int* position, int kick_power) {
    int* target = malloc_array(2);

    set_target_goal(rank, target);
    forward_position(position, target, 2 * kick_power);

    free(target);
}


/**
* Pass the ball toward the nearest players who nearer to the target goal
*/
void kick_ball_toward_teammate(int rank, int* position, int** players_position, int kick_power) {
    int* target = malloc_array(2);
    get_target_nearest_position(rank, players_position, target);
    // if the goal-nearest teammate is himself, just kick the ball toward the goal
    if (is_same(target, position)) {
        set_target_goal(rank, target);
    }
    forward_position(position, target, 2 * kick_power);

    free(target);
}

// Count how many other players in the same team nearer to the ball, including itself
int get_target_nearest_rank(int rank, int* position, int* target, int** players_position) {
    int start_index = is_team_A_player_rank(rank) ? 0 : TEAM_PLAYER;
    int i;
    int player_index = is_team_A_player_rank(rank) ? get_team_A_player_index(rank) : get_team_B_player_index(rank);
    int player_distance = get_distance(position, target);
    int cnt = 1;
    for (i = 0; i < TEAM_PLAYER; i++) if (i != player_index) {
        int distance = get_distance(target, players_position[i + start_index]);
        if (distance < player_distance || (distance == player_distance && i < player_index))
            cnt++;
    }
    return cnt;
}

/**
* Get player new position, depend on stretagy
*/
void get_player_new_position(int rank, int* position, int* ball_position, int* attrs, int** players_position) {
    bool attacking = is_attacking(rank, ball_position);
    bool is_team_A = is_team_A_player_rank(rank);

    if (!attacking) {
        // if defending, all players run forward the ball
        player_follow_ball(position, ball_position, attrs);
    } else {
        int cnt = get_target_nearest_rank(rank, position, ball_position, players_position);
        if (is_team_A) {
            if (cnt <= 8) {
                player_follow_ball(position, ball_position, attrs);
            } else {
                player_run_toward_goal(rank, position, attrs);
            }
        } else {
            if (cnt <= 3) {
                player_follow_ball(position, ball_position, attrs);
            } else if (cnt <= 5) {
                player_run_toward_goal(rank, position, attrs);
            } else {
                player_go_back_to_defend(rank, position, attrs);
            }
        }
    }
}

/**
* Player decide to kick the ball
*/
void player_kick_the_ball(int rank, int* ball_position, int* attrs, int** players_position, int* is_just_scored) {
    if (!is_player_rank(rank)) {
        return;
    }
    *is_just_scored = 0;
    int kick_power = attrs[2];
    bool target_goal = get_target_goal(rank);
    bool is_team_A = is_team_A_player_rank(rank);
    bool attacking = is_attacking(rank, ball_position);
    if (can_player_score_from_position(ball_position, target_goal, kick_power)) {
        // If this player can score, just score, and we dont need to care about new ball position
        *is_just_scored = 1;
        return;
    }
    // Team A player will always kick the ball toward the target goal
    if (is_team_A) {
        kick_ball_toward_goal(rank, ball_position, kick_power);
    } else {
        int r = rand() % 3 == 0;
        if (!attacking || r == 0) {
            kick_ball_toward_goal(rank, ball_position, kick_power);
        } else {
            kick_ball_toward_teammate(rank, ball_position, players_position, kick_power);
        }
    }
}


/**
* Field 0 gather all players position from all player
*/
void gather_all_field_players(
    MPI_Comm* field_comm, int rank, int** all_gathered_players_position,
    int** players_position, int* player_position
) {
    bool is_player = is_player_rank(rank);
    if (is_player || rank == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, field_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, 1, rank, field_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 || is_player) {
        if (is_player) {
            MPI_Gather(player_position, 2, MPI_INT, NULL, 2, MPI_INT, 0, *field_comm);
        } else {
            MPI_Gather(player_position, 2, MPI_INT, &all_gathered_players_position[0][0], 2, MPI_INT, 0, *field_comm);
            int i, j;
            for (i = 0; i < NUM_PLAYER; i++)
                for (j = 0; j < 2; j++)
                    players_position[i][j] = all_gathered_players_position[i + 1][j];
        }
    }

    MPI_Comm_free(field_comm);
    MPI_Barrier(MPI_COMM_WORLD);
}


/**
* Collect all player positions into field 0 and broadcast to all players
*/
void collect_players_position_and_broadcast(
    MPI_Comm* field_comm, int rank, int** all_gathered_players_position,
    int** players_position, int* player_position
) {
    gather_all_field_players(
        field_comm, rank, all_gathered_players_position, players_position, player_position
    );
    MPI_Bcast(&players_position[0][0], NUM_PLAYER * 2, MPI_INT, 0, MPI_COMM_WORLD);
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
    int size = 0;
    if (is_player) {
        MPI_Comm_split(MPI_COMM_WORLD, field_id, rank, field_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, rank, rank, field_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_size(*field_comm, &size);

    if (!is_player) {
        MPI_Gather(player_position_buf, 3, MPI_INT, &players_position_buf[0][0], 3, MPI_INT, 0, *field_comm);
    } else {
        assign_position(player_position_buf, player_position);
        player_position_buf[2] = rank;
        MPI_Gather(player_position_buf, 3, MPI_INT, NULL, 3, MPI_INT, 0, *field_comm);
    }

    if (!is_player) {
        int i;
        clear_players_position(players_position);
        for (i = 1; i < size; i++) {
            int player_id = players_position_buf[i][2] - NUM_FIELD;
            assign_position(players_position[player_id], players_position_buf[i]);
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);

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
    int size = 0;
    bool is_join = false;
    if (is_player) {
        is_join = is_same(ball_position, player_position);
        int color = is_join ? 0 : 1;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, field_comm);
    } else {
        is_join = rank == field_id;
        int color = is_join ? 0 : 1;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, field_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_size(*field_comm, &size);

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
        int i;
        int max_challenge = -1;
        int max_weight = -1; // weight is for tie break using random weight
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

    // MPI_Barrier(*field_comm);
    
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
void kick_ball_and_broadcast_ball_position(int rank, int winner, int* attrs, int* ball_position, int** players_position, int* is_just_scored) {
    if (!is_player_rank(winner)) {
        return;
    }
    if (winner == rank) {
        player_kick_the_ball(rank, ball_position, attrs, players_position, is_just_scored);
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
    if (rank == 0 && *is_just_scored) {
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
    bool is_player = is_player_rank(rank);
    if (is_player_rank(rank)) {
        is_join = true;
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, field_comm);
    } else {
        is_join = rank == 0;
        int color = is_join ? 0 : 1;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, field_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (is_join) {
        if (is_player) {
            player_info[0] = pre_player_position[0];
            player_info[1] = pre_player_position[1];
            player_info[2] = player_position[0];
            player_info[3] = player_position[1];
            player_info[4] = (int)is_same(player_position, pre_ball_position);
            player_info[5] = (int)(winner == rank);
            player_info[6] = ball_challenge;
            MPI_Gather(player_info, 7, MPI_INT, NULL, 7, MPI_INT, 0, *field_comm);
        } else {
            MPI_Gather(player_info, 7, MPI_INT, &players_info[0][0], 7, MPI_INT, 0, *field_comm);
        }
    }
    

    if (rank == 0) {
        int i;
        printf("Round %d\n", round_cnt);
        printf("Ball position: %d %d\n", pre_ball_position[0], pre_ball_position[1]);
        for (i = 1; i <= NUM_PLAYER; i++) {
            int player_id = (i - 1) % TEAM_PLAYER;
            if (i == 1)
                printf("Team A player info:\n");
            if (i == TEAM_PLAYER + 1)
                printf("Team B player info:\n");
            printf(
                "%d %d %d %d %d %d %d %d\n",
                player_id,
                players_info[i][0], players_info[i][1],
                players_info[i][2], players_info[i][3],
                players_info[i][4], players_info[i][5],
                players_info[i][6]
            );
        }
        printf("Score: %d %d\n", score[0], score[1]);
        printf("--------------------------\n");
    }

    // MPI_Barrier(*field_comm);

    MPI_Comm_free(field_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    int numtasks, rank;

    MPI_Group orig_group, all_field_group;
    MPI_Comm all_field_comm; // Communicator for all sub fields
    MPI_Comm field_comm; // for other players to communicate with this sub field
    int** players_position = malloc_2d_array(NUM_PLAYER, 2);
    int** players_position_buf = NULL;
    int** ball_challenges = NULL;
    int** all_gathered_players_position = NULL; // For process 0 only
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
        players_position_buf = malloc_2d_array(NUM_PLAYER + 1, 3);
        ball_challenges = malloc_2d_array(NUM_PLAYER + 1, 2);
        players_info = malloc_2d_array(NUM_PROC, 7);
        if (rank == 0) {
            all_gathered_players_position = malloc_2d_array(NUM_PLAYER + 1, 2);
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

        collect_players_position_and_broadcast(
            &field_comm, rank, all_gathered_players_position, players_position, player_position
        );

        // Broadcast ball position from field 0 to all other processes
        MPI_Bcast(ball_position, 2, MPI_INT, 0, MPI_COMM_WORLD);
        assign_position(pre_ball_position, ball_position);

        // Player run to new position
        if (is_player) {
            assign_position(pre_player_position, player_position);
            get_player_new_position(rank, player_position, ball_position, attributes, players_position);
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

        kick_ball_and_broadcast_ball_position(rank, ball_winner, attributes, ball_position, players_position, &is_just_scored);

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
 
 
