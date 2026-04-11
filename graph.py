from game import Game
from player import Player
from optimization import (
    create_prob_matrix,
    build_W_table,
    build_c_from_distribution,
    build_transition_matrix_from_strategy,
    stationary_distribution,
    optimize_leader,
)
import numpy as np
import matplotlib.pyplot as plt
from data import q_11, q_12, q_21, q_22, d_1, d_2



def graph():
    # Convert q matrices to probability matrices
    pi_11 = create_prob_matrix(q_11)
    pi_12 = create_prob_matrix(q_12)
    pi_21 = create_prob_matrix(q_21)
    pi_22 = create_prob_matrix(q_22)

    pi_player_1 = np.array([pi_11, pi_12])
    pi_player_2 = np.array([pi_21, pi_22])

    state_values = np.array([3, 5, 2, 8, 6, 4], dtype=float)

    W_leader = build_W_table(
        player=1,
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        state_values=state_values,
    )

    P_2 = build_transition_matrix_from_strategy(d_2, pi_player_2)

    state_dist_2 = stationary_distribution(P_2)

    c_2 = build_c_from_distribution(state_dist_2, d_2)

    best_d_1, best_cost = optimize_leader(
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        c_2=c_2,
        W_leader=W_leader,
    )

    step_tests = [10,50,100,500,1000,5000,10000]
    best_cost_list = []
    leader_step_list = []
    leader_step_sd_list = []

    for test in step_tests:
        results = optimize_leader(
            pi_player_1=pi_player_1,
            pi_player_2=pi_player_2,
            c_2=c_2,
            W_leader=W_leader,
            n_iter=test,
            )

        best_cost_list.append(results[1])

        leader_optimized = Player(best_d_1, np.array([q_11, q_12]), 2)
        follower_fixed = Player(d_2, np.array([q_21, q_22]), 4)

        game_optimized = Game(leader_optimized, follower_fixed)
        leader_steps_optimized, _,leader_steps_sd, _, _ = game_optimized.repeat_games(num_games=1000, step_cap=100)
        leader_step_list.append(leader_steps_optimized)
        leader_step_sd_list.append(leader_steps_sd)

    # Print mean and standard deviation for amount of steps the leader takes to capture the follower
    print(step_tests)
    print(leader_step_list)
    print(leader_step_sd_list)

    # Create the plot for leader cost
    fig, ax = plt.subplots()

    ax.plot(step_tests, best_cost_list, linewidth=2.0, label="cost")
    ax.set_xscale('log')
    ax.set_xlabel('log')

    plt.legend()
    plt.show()    


if __name__ == "__main__":
    graph()