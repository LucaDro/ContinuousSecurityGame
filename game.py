import numpy as np
from player import Player
rand = np.random.default_rng(32)

class game:
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
        The leader and the follower need to be in the same state at the same state and the time needs to be within a second of each other.
        returns true or false
        '''
        #Horrible if statement but it works
        if self.leader.state == self.follower.state and self.leader.time - 1 <= self.follower.time and self.leader.time + 1 >= self.follower.time:
            return True
        else:
            return False

    def play_game(self) -> int:
        '''
        Plays the game step by step, checking the capture condition at every step. 
        Returns step at which the leader captured the follower
        '''
        capture = False
        step = 0
        while not capture:
            step += 1
            self.leader.take_step()
            self.follower.take_step()
            self.check_capture()
        return step

