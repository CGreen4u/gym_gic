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

    We will update the observation with 3 other variables as well. given a time step
    self.max_Bz
    self.min_AE
    self.min_SymH 

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
        #defining the lower and upper bounds of the variables
        self.min_Bz = -11.22680
        self.max_Bz = 11.09640
        self.min_AE = 53.833333
        self.max_AE = 58.000000
        self.min_SymH = -52.68000
        self.max_SymH = 34.64000

        self.low = np.array([self.min_Bz,self.min_AE,self.min_SymH], dtype=np.float32)
        self.high = np.array([self.max_Bz,self.max_AE,self.max_SymH], dtype=np.float32)
        self.observation_spaces_2 = spaces.Discrete(4)

        self.observation = tuple((self.observation_spaces_2, self.low, self.high))
        
        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]),
                                       dtype=np.float64)
        self.observation_spaces_2 = spaces.Discrete(4)
        #self.observation_space = spaces.Discrete(4)
##        self.observation_view_of_alternative_variables = np.array([0,0,0])
##        self.observation_view_of_alternative_variables = spaces.Box(self.observation_view_of_alternative_variables[0],
##                                                                    self.observation_view_of_alternative_variables[1],
##                                                                    self.observation_view_of_alternative_variables[2],
##                                                                    dtype=np.float32)
##        self.observation_space = Tuple((self.observation_spaces_2,self.observation_view_of_alternative_variables))
##        
        #self.observation_view_of_alternative_variables = spaces.Box(other_info_with_number, 
        #                             dtype=np.float32)
##        observation_all = np.array([self.observation_spaces_2, self.observation_view_of_alternative_variables]) 
##        self.observation_space = Tuple((observation_all[0],
##                                        observation_all[1][0],
##                                        observation_all[1][1],
##                                        observation_all[1][2]))

##        other_info_with_number = np.array([self.work_space_x_min,
##                        self.work_space_y_min,
##                        self.work_space_z_min,
##                        -1*self.max_qw,])
##        #define the enviroment
##        self.low = np.array([self.min_Bz,self.min_AE,self.min_SymH], dtype=np.float32)
##        self.high = np.array([self.max_Bz,self.max_AE,self.max_SymH], dtype=np.float32)
        self.variables = np.array([0,0,0])
        self.number = 0
        self.guess_count = 0
        self.guess_max = 20
        self.observation = np.array([0,0,0,0]) #always starts at zero because it hasn't seen anything

        self.seed()
        self.reset()
        #print("break-this is in source code its the initial reset being called so the number comes alive at this point")
##    def get_observation_info(self):
##        '''We need to view what the current observation/state is as they will help to predict what the value needs to be for gic'''
##        observation_view_of_alternative_variables = obj.other_variables_in_list() #this pulls the information AL, Symh and the BZ
##        #the above comes in as a np array this order gic,Bz,AE,SymH, but you only get the last three in the observation
##        #observation_view_of_alternative_variables = [ -1.68443333          nan -10.60999999] is an example
##        return observation_view_of_alternative_variables
        

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
        #other_variables = self.variables
        if action < self.number:
            self.observation = np.array([1, self.variables])
            self.observation = np.array([self.observation[0],
                                         self.observation[1][0],
                                         self.observation[1][1],
                                         self.observation[1][2]])

        elif action == self.number:
            print("you got it")
            self.observation = np.array([2, self.variables])
            self.observation = np.array([self.observation[0],
                                         self.observation[1][0],
                                         self.observation[1][1],
                                         self.observation[1][2]])
            
        elif action > self.number:
            self.observation = np.array([3, self.variables])
            self.observation = np.array([self.observation[0],
                                         self.observation[1][0],
                                         self.observation[1][1],
                                         self.observation[1][2]])

        done = False

        if (self.number - self.range * 0.01) < action < (self.number + self.range * 0.01):
            reward += ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2
            done = True
            if reward == 1:
                reward += 10

        self.guess_count += 1
        if self.guess_count >= self.guess_max:
            done = True
        #print("here\/")
        #print(self.observation)
        return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        
        obj = number_tracking()
        #self.observation_view_of_alternative_variables
        self.variables = obj.other_variables_in_list() #all of the other variables that need to be observed, np array with 1 x 3
        
        self.number = obj.new_df() #theses are numbers for the GIC#self.np_random.uniform(-self.range, self.range)
        #print(self.number)
        self.guess_count = 0
        self.observation = np.array([0, self.variables]) # zero because no other observation
        self.observation = np.array([self.observation[0],
                                    self.observation[1][0],
                                    self.observation[1][1],
                                    self.observation[1][2]])
        #self.observation = 0
        return self.observation
