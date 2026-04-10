from matplotlib.pyplot import step
import numpy as np
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


    def play_game(self, step_cap = 100, debug = False) -> int | None:
        '''
        Plays the game step by step, checking the capture condition at every step. 
        Returns step at which the leader captured the follower
        '''
        if debug == True:
            print(f"The leader starts at state {self.leader.state} and the follower starts at state {self.follower.state}")
        capture = False
        step = 0
        while not capture and step < step_cap: #step_cap wasn't used before
            step += 1
            self.leader.take_step()
            if debug == True:
                print(f"the leader end up at state {self.leader.state} at time {self.leader.time}")
            self.follower.take_step()
            if debug == True:
                print(f"the follower end up at state {self.follower.state} at time {self.follower.time}")
            capture = self.check_capture()
            if capture:
                if debug == True:
                    print(f"The game ends at step {step}")
            if debug == True:
                print("\n")
        if capture:
            return step
        return None