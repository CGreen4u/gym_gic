import os
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22")
import numpy as np
import gym
from number_tracker import number_tracking



env = gym.make('GuessingGame-v0')
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
