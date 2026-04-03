import numpy as np
rand = np.random.default_rng(32)

class Player:
    def __init__(self, d_matrix: np.ndarray, q_matrices: np.ndarray, initial_state: np.ndarray) -> None:
        self.strategy = d_matrix
        self.transition_rates = q_matrices
        self.state = initial_state

    def probability_array(self, array: np.ndarray) -> np.ndarray:
        '''
        Takes an array and returns the array as probability
        '''
        sum = array.sum()
        prob_array = np.zeros(len(array))
        for idx in range(len(array)):
            prob_array[idx] = array[idx]/sum
        return prob_array

    def choose_action(self) -> int:
        '''
        Chooses an action based on which state the player is currently in
        '''
        action = np.random.choice([0, 1], p=self.strategy[self.state])
        return action    
    
    def choose_new_state(self, action) -> int:
        '''
        Chooses new state using the transition rates based on the current state and action that was chosen
        '''
        action_array = np.delete(self.transition_rates[action][self.state], self.state)
        prob_action_array = self.probability_array(action_array)
        states = np.delete([0,1,2,3,4,5], self.state)
        new_state = np.random.choice(states, p=prob_action_array)
        return new_state
    
    def choose_time(self, action: int, new_state: int) -> float:
        '''
        Chooses time that the step takes using the formula in the paper
        '''
        denominator = self.transition_rates[action][self.state][new_state]
        random_number = rand.random()
        while random_number == 0:
            random_number = rand.random()
        numerator = -np.log10(random_number)
        time = numerator/denominator
        return time
    
    def take_step(self) -> tuple[int, float]:
        '''
        Calls the functions to choose actions, find new state and get time for the step. Updates the state
        Returns the new state and the time taken for the step
        '''
        action = self.choose_action()
        new_state = self.choose_new_state(action)
        time = self.choose_time(action, new_state)
        self.state = new_state
        return new_state, time
