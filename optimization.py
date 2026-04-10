import numpy as np


def create_prob_matrix(q_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a CTMC rate matrix q into a transition probability matrix pi
    by normalizing the positive off-diagonal rates row-wise
    """
    q_matrix = np.asarray(q_matrix, dtype=float)
    n_states = q_matrix.shape[0]
    pi = np.zeros_like(q_matrix, dtype=float)

    for i in range(n_states):
        positive_rates = np.maximum(q_matrix[i], 0.0)
        total_rate = positive_rates.sum()

        if total_rate > 0:
            pi[i] = positive_rates / total_rate

    return pi


def cost_function_V(
    player: int,
    new_state_player_1: int,
    new_state_player_2: int,
    state_values: np.ndarray,
    capture_reward: float = 10.0,
) -> float:
    """
    Assumption:
    - player=1 is the leader/defender, who wants lower cost for itself
    - player=2 is the follower/attacker, who wants lower cost for itself 
    - If both land in the same state, that is treated as capture

    Modeling choice:
    - Simplified one-step cost function used as a proxy for long-term utility
    - We have to specify V explicitly in simpler way since the paper is a bit abstract

    Defender:
    - capture is very good => large negative cost
    - attacker reaching valuable states is bad => positive cost

    Attacker:
    - capture is very bad => large positive cost
    - reaching valuable states safely is good => negative cost
    """
    same_state = (new_state_player_1 == new_state_player_2)
    target_value = state_values[new_state_player_2]

    if player == 1: 
        if same_state:
            return -capture_reward - target_value
        return target_value

    if player == 2: 
        if same_state:
            return capture_reward + target_value
        return -target_value

    raise ValueError("player must be 1 or 2")


def expected_cost_W(
    player: int,
    state_player_1: int,
    action_player_1: int,
    state_player_2: int,
    action_player_2: int,
    pi_player_1: np.ndarray,
    pi_player_2: np.ndarray,
    state_values: np.ndarray,
    capture_reward: float = 10.0,
) -> float:
    """
    Expected one-step cost W from equation 28 logic:
    sum over all possible next states of
    V(...) * P(player 1 goes to j1) * P(player 2 goes to j2)
    """
    n_states = pi_player_1.shape[1]
    W = 0.0

    for j1 in range(n_states):
        for j2 in range(n_states):
            cost = cost_function_V(
                player=player,
                new_state_player_1=j1,
                new_state_player_2=j2,
                state_values=state_values,
                capture_reward=capture_reward,
            )

            prob = (
                pi_player_1[action_player_1, state_player_1, j1]
                * pi_player_2[action_player_2, state_player_2, j2]
            )

            W += cost * prob

    return W


def build_A_matrix(pi_player: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds the equality constraint matrix A and vector b for one player.

    c is assumed ordered as:
    [c(0,0), c(0,1), ..., c(1,0), c(1,1), ..., c(n-1,m-1)]

    Constraints:
    1) sum_{i,k} c(i,k) = 1
    2) for each state j:
       sum_{i,k} pi(j|i,k) c(i,k) - sum_k c(j,k) = 0
    """
    n_actions, n_states, _ = pi_player.shape
    n_vars = n_states * n_actions

    rows = []
    b = []

    rows.append(np.ones(n_vars, dtype=float))
    b.append(1.0)

    for j in range(n_states):
        row = np.zeros(n_vars, dtype=float)

        for i in range(n_states):
            for k in range(n_actions):
                idx = i * n_actions + k
                value = pi_player[k, i, j]

                if i == j:
                    value -= 1.0

                row[idx] = value

        rows.append(row)
        b.append(0.0)

    A = np.vstack(rows)
    b = np.array(b, dtype=float)
    return A, b


def recover_strategy_from_c(c_vector: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
    """
    Recovers d(k|i) from c(i,k) via:
    d(k|i) = c(i,k) / sum_k c(i,k)

    If a row sums to 0, use uniform probabilities
    """
    c_vector = np.asarray(c_vector, dtype=float)
    c_matrix = c_vector.reshape(n_states, n_actions)

    d = np.zeros_like(c_matrix)

    for i in range(n_states):
        row_sum = c_matrix[i].sum()
        if row_sum > 0:
            d[i] = c_matrix[i] / row_sum
        else:
            d[i] = np.ones(n_actions) / n_actions

    return d


def build_W_table(
    player: int,
    pi_player_1: np.ndarray,
    pi_player_2: np.ndarray,
    state_values: np.ndarray,
    capture_reward: float = 10.0,
) -> np.ndarray:
    """
    For each current state-action pair of both players, it stores the
    expected one-step cost computed by expected_cost_W().
    """
    n_actions_1, n_states_1, _ = pi_player_1.shape
    n_actions_2, n_states_2, _ = pi_player_2.shape

    W_table = np.zeros((n_states_1, n_actions_1, n_states_2, n_actions_2), dtype=float)

    for state_player_1 in range(n_states_1):
        for action_player_1 in range(n_actions_1):
            for state_player_2 in range(n_states_2):
                for action_player_2 in range(n_actions_2):
                    W_table[state_player_1, action_player_1, state_player_2, action_player_2] = expected_cost_W(
                        player=player,
                        state_player_1=state_player_1,
                        action_player_1=action_player_1,
                        state_player_2=state_player_2,
                        action_player_2=action_player_2,
                        pi_player_1=pi_player_1,
                        pi_player_2=pi_player_2,
                        state_values=state_values,
                        capture_reward=capture_reward,
                    )

    return W_table


def check_admissibility(c_vector: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> tuple[bool, np.ndarray]:
    """
    Checks whether a candidate c vector satisfies A @ c = b approximately
    """
    c_vector = np.asarray(c_vector, dtype=float)
    residual = A @ c_vector - b
    is_admissible = np.all(np.abs(residual) < tol)
    return is_admissible, residual


def build_c_from_distribution(state_distribution: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """
    Builds a c vector from a state distribution and strategy matrix d
    c(i,k) = P(state=i) * d(k|i)
    """
    state_distribution = np.asarray(state_distribution, dtype=float)
    d_matrix = np.asarray(d_matrix, dtype=float)

    c_matrix = state_distribution[:, None] * d_matrix
    return c_matrix.reshape(-1)


def build_transition_matrix_from_strategy(d_matrix: np.ndarray, pi_player: np.ndarray) -> np.ndarray:
    """
    Builds the effective state-to-state transition matrix under a mixed strategy d
    P(j | i) = sum_k d(k|i) * pi(j|i,k)
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    pi_player = np.asarray(pi_player, dtype=float)

    n_actions, n_states, _ = pi_player.shape
    P = np.zeros((n_states, n_states), dtype=float)

    for i in range(n_states):
        for j in range(n_states):
            for k in range(n_actions):
                P[i, j] += d_matrix[i, k] * pi_player[k, i, j]

    return P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Computes a stationary distribution x such that xP = x and sum(x)=1.
    """
    P = np.asarray(P, dtype=float)
    n_states = P.shape[0]

    A = P.T - np.eye(n_states)
    A[-1] = np.ones(n_states)
    b = np.zeros(n_states)
    b[-1] = 1.0

    x = np.linalg.solve(A, b)
    return x


def compute_expected_cost(c_1: np.ndarray, c_2: np.ndarray, W_table: np.ndarray) -> float:
    """
    Computes the total expected cost for one player

    c_1: leader 
    c_2: follower 

    W_table[i, k, j, l] = cost when:
        leader is in (i,k) and follower is in (j,l)
    """
    c_1 = np.asarray(c_1)
    c_2 = np.asarray(c_2)

    n_states = W_table.shape[0]
    n_actions = W_table.shape[1]

    cost = 0.0

    for i in range(n_states):
        for k in range(n_actions):
            idx_1 = i * n_actions + k

            for j in range(n_states):
                for l in range(n_actions):
                    idx_2 = j * n_actions + l

                    cost += (
                        c_1[idx_1]
                        * c_2[idx_2]
                        * W_table[i, k, j, l]
                    )

    return cost

def random_strategy(n_states: int, n_actions: int) -> np.ndarray:
    """
    Generates a random valid strategy 
    """
    d = np.random.rand(n_states, n_actions)
    d /= d.sum(axis=1, keepdims=True)
    return d


def build_c_from_strategy(d_matrix: np.ndarray, pi_player: np.ndarray) -> np.ndarray:
    """
    d -> P -> stationary distribution -> c
    """
    P = build_transition_matrix_from_strategy(d_matrix, pi_player)
    state_dist = stationary_distribution(P)
    c = build_c_from_distribution(state_dist, d_matrix)
    return c


def optimize_leader(
    pi_player_1: np.ndarray,
    pi_player_2: np.ndarray,
    c_2: np.ndarray,
    W_leader: np.ndarray,
    n_iter: int = 500,
) -> tuple[np.ndarray, float]:
    """
    Finds a better leader strategy by random search
    """
    n_states = pi_player_1.shape[1]
    n_actions = pi_player_1.shape[0]

    best_d = None
    best_cost = float("inf")

    for _ in range(n_iter):
        d_candidate = random_strategy(n_states, n_actions)

        c_candidate = build_c_from_strategy(d_candidate, pi_player_1)

        cost = compute_expected_cost(c_candidate, c_2, W_leader)

        if cost < best_cost:
            best_cost = cost
            best_d = d_candidate

    return best_d, best_cost


