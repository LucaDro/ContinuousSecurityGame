import numpy as np
rand = np.random.default_rng(32)

class Player:
    def __init__(self, d_matrix, q_matrices, initial_state):
        self.strategy = d_matrix
        self.transition_rates = q_matrices
        self.state = initial_state

    def probability_array(self, array:np.ndarray):
        sum = array.sum()
        prob_array = np.zeros(len(array))
        for idx in range(len(array)):
            prob_array[idx] = array[idx]/sum
        return prob_array

    def choose_action(self):
        action = np.random.choice([0, 1], p=self.strategy[self.state])
        return action    
    
    def choose_new_state(self, action):
        action_array = np.delete(self.transition_rates[action][self.state], self.state)
        prob_action_array = self.probability_array(action_array)
        states = np.delete([0,1,2,3,4,5], self.state)
        new_state = np.random.choice(states, p=prob_action_array)
        return new_state
    
    def choose_time(self, action, new_state):
        denominator = self.transition_rates[action][self.state][new_state]
        random_number = rand.random()
        while random_number == 0:
            random_number = rand.random()
        numerator = -np.log10(random_number)
        time = numerator/denominator
        return time
