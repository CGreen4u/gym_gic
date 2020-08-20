import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import os
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22")
from number_tracker import number_tracking
#https://docs.ray.io/en/latest/rllib-env.html # advanced multi agent
#https://stable-baselines.readthedocs.io/en/master/modules/a2c.html #alternative to gym


class GuessingGame(gym.Env):
    """Number guessing game

    The object of the game is to guess within 1% of the randomly chosen number
    within 200 time steps

    After each step the agent is provided with one of four possible observations
    which indicate where the guess is in relation to the randomly chosen number

    We will update the observation to be the 4 varaiables as well.
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target

    #The rewards are:
    #0 if the agent's guess is outside of 1% of the target
    #1 if the agent's guess is inside 1% of the target

    The rewards is calculated as:
    (min(action, self.number) + self.range) / (max(action, self.number) + self.range)
    Ideally an agent will be able to recognise the 'scent' of a higher reward and
    increase the rate in which is guesses in that direction until the reward reaches
    its maximum

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
        self.range = 4  # Randomly selected number is within +/- this value
        self.bounds = 4
        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]),
                                       dtype=np.float64)
        self.observation_space = spaces.Discrete(4)

        self.number = 0
        self.guess_count = 0
        self.guess_max = 20
        self.observation = 0 #always starts at zero because it hasn't seen anything

        self.seed()
        self.reset()
        #print("break-this is in source code its the initial reset being called so the number comes alive at this point")
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print(action[0])
        #assert self.action_space.contains(action[0]), "%r (%s) invalid" % (action[0], type(action[0])) #simply verifying that the information being introduced is actually in our action space -4 through 4 
        #we need to add the observation of the random variables here. just so it can view it on observation
        #we want it to view the last 10 states in observations instead of 0 after first 
        #last_position, last_change_in_velocity = self.state
        reward = 0
        
        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2
            
        elif action > self.number:
            self.observation = 3

        done = False

        if (self.number - self.range * 0.005) < action < (self.number + self.range * 0.005):
            reward += ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2
            done = True
            if reward == 1:
                reward += 10

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
