#open AI type of enviroment

#The agent can maximize its reward by following the proper path of the GIC directly
#The agent loses points when it falls off course.

#if the distance between the actual and predicted is greater than 1 the agent losses more points

##import numpy as np
##import matplotlib.pyplot as plt
##
##Class enviorment(object):
##    def __init__(self, m, n, magic):
##        self.grid = np
##        
##


import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)


from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

print(x)
