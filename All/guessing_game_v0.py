import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import os
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22")
from number_tracker import number_tracking



class GuessingGame(gym.Env):
    """Number guessing game

    The object of the game is to guess within 1% of the randomly chosen number
    within 200 time steps

    After each step the agent is provided with one of four possible observations
    which indicate where the guess is in relation to the randomly chosen number

    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target

    The rewards are:
    0 if the agent's guess is outside of 1% of the target
    1 if the agent's guess is inside 1% of the target

    The episode terminates after the agent guesses within 1% of the target or
    200 steps have been taken

    The agent will need to use a memory of previously submitted actions and observations
    in order to efficiently explore the available actions

    The purpose is to have agents optimise their exploration parameters (e.g. how far to
    explore from previous actions) based on previous experience. Because the goal changes
    each episode a state-value or action-value function isn't able to provide any additional
    benefit apart from being able to tell whether to increase or decrease the next guess.

    The perfect agent would likely learn the bounds of the action space (without referring
    to them explicitly) and then follow binary tree style exploration towards to goal number
    """
    
    
    def __init__(self):
        self.range = 3  # Randomly selected number is within +/- this value
        self.bounds = 30

        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]),
                                       dtype=np.float32)
        self.observation_space = spaces.Discrete(4)


        self.observation_if_outside_target = spaces.Discrete(4)
        self.observation_view_of_alternative_variables = spaces.Box(other_info_with_number, 
                                     dtype=np.float32)
        self.observation_space = Tuple([self.observation_if_outside_target, 
                                        self.observation_view_of_alternative_variables])
        
        numeric_high = np.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.max_vel_ang_y,
                            self.max_vel_ang_z])

        self.number = 0
        self.guess_count = 0
        self.guess_max = 8
        self.observation = 0

        self.seed()
        self.reset()
        #print("break-this is in source code its the initial reset being called so the number comes alive at this point")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2

        elif action > self.number:
            self.observation = 3

        reward = 0
        done = False

        if (self.number - self.range * 0.0001) < action < (self.number + self.range * 0.0001):
            reward = 1
            done = True

        self.guess_count += 1
        if self.guess_count >= self.guess_max:
            done = True

        return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        
        obj = number_tracking()
        
        self.number = obj.new_df() #theses are numbers for the GIC#self.np_random.uniform(-self.range, self.range)
        #print(self.number)
        self.guess_count = 0
        self.observation = 0
        return self.observation

