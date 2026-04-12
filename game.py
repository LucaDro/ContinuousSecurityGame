from matplotlib.pyplot import step
import numpy as np
from math import sqrt
from player import Player
rand = np.random.default_rng(32)

class Game:
    '''
    For the moment, it is only a two-player game, one leader and one follower
    '''
    def __init__(self, leader: Player, follower: Player) -> None:
        self.leader = leader
        self.follower = follower
        self.step = 0

    def check_capture(self) -> bool:
        '''
        Checks if the capture condition is fulfilled. 
        The leader and the follower need to be in the same state at the same time.
        returns true or false
        '''
        #Horrible if statement but it works
        if self.leader.state == self.follower.state:
            if (self.leader.time <= self.follower.time and self.leader.time >= self.follower.previous_time) or (self.follower.time <= self.leader.time and self.follower.time >= self.leader.previous_time):
                return True
        return False
    
    
    def play_game(self, step_cap = 100, debug = False) -> tuple[int, int] | None:
        '''
        Plays the game step by step, checking the capture condition at every step. 
        Returns step at which the leader captured the follower
        '''
        if debug == True:
            print(f"The leader starts at state {self.leader.state} and the follower starts at state {self.follower.state}")
        capture = False
        leader_step = 0
        follower_step = 0
        while not capture and leader_step < step_cap and follower_step < step_cap:
            if self.follower.time < self.leader.time:
                follower_step += 1
                self.follower.take_step()
                if debug == True:
                    print(f"the follower end up at state {self.follower.state} at time {self.follower.time} and step {follower_step}")
            else:
                leader_step += 1
                self.leader.take_step()
                if debug == True:
                    print(f"the leader end up at state {self.leader.state} at time {self.leader.time} and step {leader_step}")
            capture = self.check_capture()
            if capture:
                if debug == True:
                    print(f"The game ends at step {leader_step} for the leader and step {follower_step} for the follower")
            if debug == True:
                print("\n")
        if capture:
            return leader_step, follower_step
        return None 
    
    def repeat_games(self, num_games: int, step_cap = 100) -> tuple[int, int]:
        """
        Plays multiple games and returns the average step count of all players excluding the games where they exceeded the step_cap.
        """
        start_leader = self.leader.copy()
        start_follower = self.follower.copy()
        leader_steps = []
        follower_steps = []
        steps = self.play_game(step_cap, True)
        if steps is not None:
                leader_step = steps[0]
                follower_step = steps[1]
        leader_steps.append(leader_step)
        follower_steps.append(follower_step)
        for i in range(num_games):
            self.leader = start_leader.copy()
            self.follower = start_follower.copy()
            steps = self.play_game(step_cap)
            if steps is not None:
                leader_step = steps[0]
                follower_step = steps[1]
            leader_steps.append(leader_step)
            follower_steps.append(follower_step)
        # Calculate the mean steps
        num_games_finished = 0
        unfinished_games = 0
        leader_average = 0
        follower_average = 0
        for idx in range(len(leader_steps)):
            if leader_steps[idx] is not None:
                num_games_finished += 1
                leader_average += leader_steps[idx]
                follower_average += follower_steps[idx]
            else:
                unfinished_games += 1
        leader_average /= num_games_finished
        follower_average /= num_games_finished
        # Calculate the standard deviation of the steps
        leader_variance = 0
        follower_variance = 0
        for idx in range(len(leader_steps)):
            if leader_steps[idx] is not None:
                leader_variance = (leader_steps[idx]-leader_average)**2
                follower_variance = (follower_steps[idx]-follower_average)**2
        leader_sd = sqrt(leader_variance)
        follower_sd = sqrt(follower_variance)
        return leader_average, follower_average, leader_sd, follower_sd, unfinished_games