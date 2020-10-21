import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22")
from number_tracker import number_tracking

''' 
description
 Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

'''
variables = ['AE', 'SymH', 'Bz', 'gic']
df = pd.read_csv(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\combined_csv.csv", usecols=variables)
print(df.head())
df= df.replace([99999.9, 9999.99, 999.99, 999999.0 ,999999.00,-99999.990000, '#REF!'], np.nan)
print(df.describe())
#shift_in_date = 1 # it counts by one the whole time
#first order derivative
#df['lag_1'] = df['gic'] - df['gic'].shift(1)/(shift_in_date)
os.chdir(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all')
df.to_csv('example_data_in_order.csv',index=False)

def get_sample_of_data(df):
    '''we are just getting sample data within a range'''
    row_sample = df.sample(n=1)#, random_state=1) #getting random sample 
    index_number = row_sample.index[0] # getting the index number
    x = df[index_number:(index_number + 100)] # range of numbers staring at random sample up to n number of selections
    return x, index_number #returning the values in range plus the index starting point

sample_date, index_number = get_sample_of_data(df)
print(sample_date)
print(index_number) 
# '''
# oberservation
#     type: box
#     continous action space
#     ['AE', 'SymH', 'Bz', 'gic'] = [(-0.04-0.04),(2.061795), (-99999.99), (-11.376)


#         Num     Observation               Min                     Max
#         0       Bz                       -11.22680               11.09640
#         1       AE                        53.833333              58.000000
#         2       SymH                     -52.68000               34.64000
#         3       GIC                      -4.530000               1.630000
#         4       lag_1                    -4.450000               4.180000

#                 gic            Bz          AE          SymH         lag_1
# count  2.674783e+06  2.612157e+06  181.000000  2.674783e+06  2.674782e+06
# mean  -7.994705e-04 -4.765889e-01   55.843002 -6.905632e+00  3.738622e-08
# std    1.794034e-01  2.416170e+00    1.213410  1.192425e+01  2.574833e-02
# min   -4.530000e+00 -1.122680e+01   53.833333 -5.268000e+01 -4.450000e+00
# 25%   -9.000000e-02 -1.840058e+00   54.791667 -1.417200e+01 -2.000000e-02
# 50%    1.000000e-02 -4.460800e-01   55.805556 -6.520000e+00  0.000000e+00
# 75%    9.000000e-02  9.887800e-01   56.875000  1.957333e+00  1.000000e-02
# max    1.630000e+00  1.109640e+01   58.000000  3.464000e+01  4.180000e+00


# Starting State:
#          The position of the game is assigned a uniform random value in
#          [-0.6 , -0.4].
#          The starting velocity of the car is always assigned to 0.

#          Unlike this game our system does not need to always start at the same position. it is based on past information. 
#          if there is no value avaliable we can have it start at a random position, if not it needs to continue from the last 
#          position.

#     '''
# def __init__(self, goal_gic, min_):

#     #the goal needs to be the actual gic?
# def __init__(self, goal_gic): #goal_velocity=0): #we are explaining the limits of the enviorment here
    
#     self.guess_count = 0
#     self.guess_max = 8
#     '''We are telling the machine what the env look like'''
#     self.min_Bz = -11.22680
#     self.max_Bz = 11.09640
#     self.min_AE = 53.833333
#     self.max_AE = 58.000000
#     self.min_SymH = -52.68000
#     self.max_SymH = 34.64000
#     self.max_change = 4.45
#     #self.goal_position = 0.5
#     self.goal_gic = goal_gic
#     #self.goal_velocity = goal_velocity
# #the gic always has the disire to settle at or near zero - how can I make this a variable.

#     self.low = np.array([self.min_Bz,self.min_AE,self.min_SymH, -self.max_change], dtype=np.float32)
#     self.high = np.array([self.max_Bz,self.max_AE,self.max_SymH, self.max_change], dtype=np.float32)

