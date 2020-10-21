import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import os
import time
import pandas as pd
from pandas.io.common import EmptyDataError
from guessing_notebook import does_the_csv_exist, new_guess

os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")
style.use('fivethirtyeight')

#you need to delete the conter at the start of each run
if os.path.exists("counter.csv"):
  os.remove("counter.csv")
else:
  print("The file does not exist")

if os.path.exists("guess.csv"):
  os.remove("guess.csv")
else:
  print("The file does not exist")

fig, ax = plt.subplots()
does_the_csv_exist()
##fig = plt.figure()
#ax = fig.add_subplot(1,1,1)


graph_data = pd.read_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\example2.txt')
date = []
point = []
point_all = []
guess = []


class creater:
    def __init__(self):
        self.count = 0

    def animate(self):
        count = len(point_all)
        new_guess() # calling our file to run to make a new guess
        ax.set_xticklabels(date, rotation=45)
        guessing_file = pd.read_csv(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\guess.csv')
        guess.append(guessing_file.iat[count,0]) #we are calling the x column in the len or count of times script ran to go to next row
        #self.count += 1
        for row1, index1 in graph_data.iterrows():
            print(index1[1])
            #time.sleep(10)
            date.append(index1[0])
            point.append(float(index1[1]))
            point_all.append(float(index1[1]))
            break
        if len(date) > 30:
            date.remove(date[0])
            point.remove(point[0])
            guess.remove(guess[0])
        graph_data.drop(row1, inplace=True)
        print(guess)
        ax.clear()
        ax.plot(date, point, 'b')
        ax.plot(date, guess, 'g')
        fig.autofmt_xdate()


ani = animation.FuncAnimation(fig, creater.animate,frames=10, interval=1200) #milliseconds every second it updates
plt.show()




##def __init__(self):
##    self.date = []
##    self.point = []
##    self.original_file = pd.read_csv('example.txt')
##    self.graph_data = pd.read_csv('example2.txt')
##    
##    os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")
##    style.use('fivethirtyeight')
##
##def animate(self):
##    fig, ax = plt.subplots()
##    self.ax.set_xticklabels(date, rotation=45)
##    for row1, index1 in self.graph_data.iterrows():
##        print(index1[1])
##        #time.sleep(10)
##        self.date.append(index1[0])
##        self.point.append(float(index1[1]))
##        break
##    if len(date) > 30:
##        self.date.remove(date[0])
##        self.point.remove(point[0])
##    self.graph_data.drop(row1, inplace=True)
##    self.ax.clear()
##    self.ax.plot(date, point)
##
##
##ani = animation.FuncAnimation(fig, animate,frames=10, interval=1000) #milliseconds every second it updates
##plt.show()


    #    self.observation_space = spaces.Discrete(4)

    #     self.number = 0
    #     self.guess_count = 0
    #     self.guess_max = 200
    #     self.observation = 0

    #     self.seed()
    #     self.reset()
    #     #print("break-this is in source code its the initial reset being called so the number comes alive at this point")

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def step(self, action):
    #     #assert self.action_space.contains(action)

    #     if action < self.number:
    #         self.observation = 1

    #     elif action == self.number:
    #         self.observation = 2

    #     elif action > self.number:
    #         self.observation = 3

    #     reward = 0
    #     done = False

    #     if (self.number - self.range * 0.10) < action < (self.number + self.range * 0.10):
    #         reward = 1
    #         done = True

    #     self.guess_count += 1
    #     if self.guess_count >= self.guess_max:
    #         done = True

    #     return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    # def reset(self):
        
    #     obj = number_tracking()
        
    #     self.number = obj.new_df() #theses are numbers for the GIC#self.np_random.uniform(-self.range, self.range)
    #     #print(self.number)
    #     self.guess_count = 0
    #     self.observation = 0
    #     return self.observation
