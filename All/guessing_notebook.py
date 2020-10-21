import numpy as np
import gym
from number_tracker import number_tracking
import pandas as pd
import os
env = gym.make('GuessingGame-v0')


def does_the_csv_exist():
    # if file does not exist write header 
    if not os.path.isfile(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\guess.csv'):
        d = {'guess': []}
        df = pd.DataFrame(data=d)
        df.to_csv('guess.csv',index=False)
    else: # else it exists so append without writing the header
        pass

def new_guess():
    env.reset()

    # applying dichotomy # we constantly cut the number in half until we get close
    def dichotomize(upper, lower):
        return (upper + lower) / 2.0

    upper_bound = 15
    lower_bound = -15

    done = False


    # lst = [-1,5,3,4,7,10,9]
    # for x in lst:
    target = env.env.number
    while not done:
        guess = dichotomize(upper_bound, lower_bound)
        #print("[{}, {}, {}]".format(lower_bound, guess, upper_bound))

        observation, reward, done, info = env.step(np.array([guess]))

        if observation == 1:
            # Guess is lower than the target
            lower_bound = guess
        elif observation == 2:
            # Guess is equal to the target
            print("You Got it")
        elif observation == 3:
            # Guess is higher than the target
            upper_bound = guess
        else:
            raise("Problem")


    print("Guess: {} | Target: {}".format(guess, target))

    dr = pd.read_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\guess.csv') # open guess csv
    dr = dr.append({'guess': guess}, ignore_index=True) # add value
    dr.to_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\guess.csv',index=False) # save and close updated csv

# does_the_csv_exist()

# new_guess()