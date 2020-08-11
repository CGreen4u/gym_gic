#Create a q-table
import gym
from number_tracker import number_tracking
import pandas as pd
import time
import os
env = gym.make('GuessingGame-v0')
import numpy as np
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")
#print((env.action_space[1]))
q_table = np.zeros([env.observation_space.n, 20]) #with our GIC being contious we need to make a descret table. 

"""Training the agent"""

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    print(state)
    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        print(action)
        #next_state, reward, done, info = env.step(action) 
        #reward, done, info = env.step(action)
        #next_state = 1
        guess = dichotomize(upper_bound, lower_bound)
        #print("[{}, {}, {}]".format(lower_bound, guess, upper_bound))

        observation, reward, done, info = env.step(np.array([guess]))

        observation, reward, done, info = env.step(np.array([guess]))

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")