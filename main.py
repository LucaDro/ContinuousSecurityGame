from game import Game
from player import Player
from optimization import (
    create_prob_matrix,
    expected_cost_W,
    build_A_matrix,
    recover_strategy_from_c,
    build_W_table,
    check_admissibility,
    build_c_from_distribution,
    build_transition_matrix_from_strategy,
    stationary_distribution,
    compute_expected_cost,
    optimize_leader,
)
import numpy as np
from data import q_11, q_12, q_21, q_22, d_1, d_2


def main():
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

    W_follower = build_W_table(
        player=2,
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        state_values=state_values,
    )

    # Test expected one-step costs
    leader_W = expected_cost_W(
        player=1,
        state_player_1=2,
        action_player_1=1,
        state_player_2=4,
        action_player_2=0,
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        state_values=state_values,
    )

    follower_W = expected_cost_W(
        player=2,
        state_player_1=2,
        action_player_1=1,
        state_player_2=4,
        action_player_2=0,
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        state_values=state_values,
    )

    print("Leader W test:", leader_W)
    print("Follower W test:", follower_W)

    # Build admissibility matrices
    A1, b1 = build_A_matrix(pi_player_1)
    A2, b2 = build_A_matrix(pi_player_2)

    print("A1 shape:", A1.shape, "| b1 shape:", b1.shape)
    print("A2 shape:", A2.shape, "| b2 shape:", b2.shape)

    P_1 = build_transition_matrix_from_strategy(d_1, pi_player_1)
    P_2 = build_transition_matrix_from_strategy(d_2, pi_player_2)

    state_dist_1 = stationary_distribution(P_1)
    state_dist_2 = stationary_distribution(P_2)

    c_1 = build_c_from_distribution(state_dist_1, d_1)
    c_2 = build_c_from_distribution(state_dist_2, d_2)

    best_d_1, best_cost = optimize_leader(
        pi_player_1=pi_player_1,
        pi_player_2=pi_player_2,
        c_2=c_2,
        W_leader=W_leader,
    )

    print("Best leader cost found:", best_cost)
    print("Best leader strategy:\n", best_d_1)

    leader_cost = compute_expected_cost(c_1, c_2, W_leader)
    follower_cost = compute_expected_cost(c_1, c_2, W_follower)

    print("Leader expected cost:", leader_cost)
    print("Follower expected cost:", follower_cost)

    admissible_1, residual_1 = check_admissibility(c_1, A1, b1)
    admissible_2, residual_2 = check_admissibility(c_2, A2, b2)

    print("c_1 admissible:", admissible_1)
    print("Residual c_1:", residual_1)

    print("c_2 admissible:", admissible_2)
    print("Residual c_2:", residual_2)

    recovered_d_1 = recover_strategy_from_c(c_1, n_states=6, n_actions=2)
    recovered_d_2 = recover_strategy_from_c(c_2, n_states=6, n_actions=2)

    print("Recovered d_1:\n", recovered_d_1)
    print("Recovered d_2:\n", recovered_d_2)

    print("\n--- Original paper strategy ---")
    leader_original = Player(d_1, np.array([q_11, q_12]), 2)
    follower_original = Player(d_2, np.array([q_21, q_22]), 4)

    game_original = Game(leader_original, follower_original)
    leader_steps_original, follower_steps_original, unfinished_original = game_original.repeat_games(num_games=100, step_cap=100)

    print(f"Original game: Leader had {leader_steps_original} steps on average, follower had {follower_steps_original}, step_cap was exceeded in {unfinished_original} games")

    print("\n--- Optimized leader strategy ---")
    leader_optimized = Player(best_d_1, np.array([q_11, q_12]), 2)
    follower_fixed = Player(d_2, np.array([q_21, q_22]), 4)

    game_optimized = Game(leader_optimized, follower_fixed)
    leader_steps_optimized, follower_steps_optimized, leader_steps_sd, follower_steps_sd, unfinished_optimized = game_optimized.repeat_games(num_games=100, step_cap=100)

    print(f"Optimized game: Leader had {leader_steps_optimized} steps on average, follower had {follower_steps_optimized}, step_cap was exceeded in {unfinished_optimized} games")

    print("\n--- Summary ---")
    print("Original leader cost:", leader_cost)
    print("Optimized leader cost:", best_cost)
    print("Original leader strategy:\n", d_1)
    print("Optimized leader strategy:\n", best_d_1)


if __name__ == "__main__":
    main()