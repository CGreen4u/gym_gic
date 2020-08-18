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
        self.bounds = 10
        self.action_space = spaces.Box(low=np.array([-4.  , -3.99, -3.98, -3.97, -3.96, -3.95, -3.94, -3.93, -3.92,
       -3.91, -3.9 , -3.89, -3.88, -3.87, -3.86, -3.85, -3.84, -3.83,
       -3.82, -3.81, -3.8 , -3.79, -3.78, -3.77, -3.76, -3.75, -3.74,
       -3.73, -3.72, -3.71, -3.7 , -3.69, -3.68, -3.67, -3.66, -3.65,
       -3.64, -3.63, -3.62, -3.61, -3.6 , -3.59, -3.58, -3.57, -3.56,
       -3.55, -3.54, -3.53, -3.52, -3.51, -3.5 , -3.49, -3.48, -3.47,
       -3.46, -3.45, -3.44, -3.43, -3.42, -3.41, -3.4 , -3.39, -3.38,
       -3.37, -3.36, -3.35, -3.34, -3.33, -3.32, -3.31, -3.3 , -3.29,
       -3.28, -3.27, -3.26, -3.25, -3.24, -3.23, -3.22, -3.21, -3.2 ,
       -3.19, -3.18, -3.17, -3.16, -3.15, -3.14, -3.13, -3.12, -3.11,
       -3.1 , -3.09, -3.08, -3.07, -3.06, -3.05, -3.04, -3.03, -3.02,
       -3.01, -3.  , -2.99, -2.98, -2.97, -2.96, -2.95, -2.94, -2.93,
       -2.92, -2.91, -2.9 , -2.89, -2.88, -2.87, -2.86, -2.85, -2.84,
       -2.83, -2.82, -2.81, -2.8 , -2.79, -2.78, -2.77, -2.76, -2.75,
       -2.74, -2.73, -2.72, -2.71, -2.7 , -2.69, -2.68, -2.67, -2.66,
       -2.65, -2.64, -2.63, -2.62, -2.61, -2.6 , -2.59, -2.58, -2.57,
       -2.56, -2.55, -2.54, -2.53, -2.52, -2.51, -2.5 , -2.49, -2.48,
       -2.47, -2.46, -2.45, -2.44, -2.43, -2.42, -2.41, -2.4 , -2.39,
       -2.38, -2.37, -2.36, -2.35, -2.34, -2.33, -2.32, -2.31, -2.3 ,
       -2.29, -2.28, -2.27, -2.26, -2.25, -2.24, -2.23, -2.22, -2.21,
       -2.2 , -2.19, -2.18, -2.17, -2.16, -2.15, -2.14, -2.13, -2.12,
       -2.11, -2.1 , -2.09, -2.08, -2.07, -2.06, -2.05, -2.04, -2.03,
       -2.02, -2.01, -2.  , -1.99, -1.98, -1.97, -1.96, -1.95, -1.94,
       -1.93, -1.92, -1.91, -1.9 , -1.89, -1.88, -1.87, -1.86, -1.85,
       -1.84, -1.83, -1.82, -1.81, -1.8 , -1.79, -1.78, -1.77, -1.76,
       -1.75, -1.74, -1.73, -1.72, -1.71, -1.7 , -1.69, -1.68, -1.67,
       -1.66, -1.65, -1.64, -1.63, -1.62, -1.61, -1.6 , -1.59, -1.58,
       -1.57, -1.56, -1.55, -1.54, -1.53, -1.52, -1.51, -1.5 , -1.49,
       -1.48, -1.47, -1.46, -1.45, -1.44, -1.43, -1.42, -1.41, -1.4 ,
       -1.39, -1.38, -1.37, -1.36, -1.35, -1.34, -1.33, -1.32, -1.31,
       -1.3 , -1.29, -1.28, -1.27, -1.26, -1.25, -1.24, -1.23, -1.22,
       -1.21, -1.2 , -1.19, -1.18, -1.17, -1.16, -1.15, -1.14, -1.13,
       -1.12, -1.11, -1.1 , -1.09, -1.08, -1.07, -1.06, -1.05, -1.04,
       -1.03, -1.02, -1.01, -1.  , -0.99, -0.98, -0.97, -0.96, -0.95,
       -0.94, -0.93, -0.92, -0.91, -0.9 , -0.89, -0.88, -0.87, -0.86,
       -0.85, -0.84, -0.83, -0.82, -0.81, -0.8 , -0.79, -0.78, -0.77,
       -0.76, -0.75, -0.74, -0.73, -0.72, -0.71, -0.7 , -0.69, -0.68,
       -0.67, -0.66, -0.65, -0.64, -0.63, -0.62, -0.61, -0.6 , -0.59,
       -0.58, -0.57, -0.56, -0.55, -0.54, -0.53, -0.52, -0.51, -0.5 ,
       -0.49, -0.48, -0.47, -0.46, -0.45, -0.44, -0.43, -0.42, -0.41,
       -0.4 , -0.39, -0.38, -0.37, -0.36, -0.35, -0.34, -0.33, -0.32,
       -0.31, -0.3 , -0.29, -0.28, -0.27, -0.26, -0.25, -0.24, -0.23,
       -0.22, -0.21, -0.2 , -0.19, -0.18, -0.17, -0.16, -0.15, -0.14,
       -0.13, -0.12, -0.11, -0.1 , -0.09, -0.08, -0.07, -0.06, -0.05,
       -0.04, -0.03, -0.02, -0.01]), high=np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
       0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
       0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
       0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
       0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
       1.1 , 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2 ,
       1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3 , 1.31,
       1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4 , 1.41, 1.42,
       1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5 , 1.51, 1.52, 1.53,
       1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6 , 1.61, 1.62, 1.63, 1.64,
       1.65, 1.66, 1.67, 1.68, 1.69, 1.7 , 1.71, 1.72, 1.73, 1.74, 1.75,
       1.76, 1.77, 1.78, 1.79, 1.8 , 1.81, 1.82, 1.83, 1.84, 1.85, 1.86,
       1.87, 1.88, 1.89, 1.9 , 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97,
       1.98, 1.99, 2.  , 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08,
       2.09, 2.1 , 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18, 2.19,
       2.2 , 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.3 ,
       2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39, 2.4 , 2.41,
       2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5 , 2.51, 2.52,
       2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59, 2.6 , 2.61, 2.62, 2.63,
       2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7 , 2.71, 2.72, 2.73, 2.74,
       2.75, 2.76, 2.77, 2.78, 2.79, 2.8 , 2.81, 2.82, 2.83, 2.84, 2.85,
       2.86, 2.87, 2.88, 2.89, 2.9 , 2.91, 2.92, 2.93, 2.94, 2.95, 2.96,
       2.97, 2.98, 2.99, 3.  , 3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07,
       3.08, 3.09, 3.1 , 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18,
       3.19, 3.2 , 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 3.28, 3.29,
       3.3 , 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39, 3.4 ,
       3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49, 3.5 , 3.51,
       3.52, 3.53, 3.54, 3.55, 3.56, 3.57, 3.58, 3.59, 3.6 , 3.61, 3.62,
       3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69, 3.7 , 3.71, 3.72, 3.73,
       3.74, 3.75, 3.76, 3.77, 3.78, 3.79, 3.8 , 3.81, 3.82, 3.83, 3.84,
       3.85, 3.86, 3.87, 3.88, 3.89, 3.9 , 3.91, 3.92, 3.93, 3.94, 3.95,
       3.96, 3.97, 3.98, 3.99]),
                                       dtype=np.float64)
        #self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]),
        #                               dtype=np.float32)
        self.observation_space = spaces.Discrete(4)

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
        #assert self.action_space.contains(action)

        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2

        elif action > self.number:
            self.observation = 3

        reward = 0
        done = False

        if (self.number - self.range * 0.01) < action < (self.number + self.range * 0.01):
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
