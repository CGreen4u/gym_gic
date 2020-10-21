import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
os.chdir(r"C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all")

dfx = pd.read_csv('combined_csv.csv')
dfx.index = pd.to_datetime(dfx[['year','month','day' ,'hour','minute','second']])
#dfx = dfx.drop(columns=["year", "month", "day", "hour", "minute", "second"])
#dfx= dfx.replace([99999.9, 9999.99, 999.99, 999999.0 ,999999.00,-99999.990000, '#REF!'], np.nan)
#dfx = dfx.fillna(method='backfill')
#del dfx['AE']

def calculate_derivative(df, new_column_name, column_to_derive, shift_in_date = 1):
    df[new_column_name] = (df[column_to_derive])-(df[column_to_derive].shift(1))/(shift_in_date)
    return df[new_column_name]

new_column_name = 'FOD' #first order derivative
column_to_derive = 'gic'
calculate_derivative(dfx, new_column_name, column_to_derive, shift_in_date = 1)

new_column_name = 'SOD' #second order derivative
column_to_derive = 'FOD'
calculate_derivative(dfx, new_column_name, column_to_derive, shift_in_date = 1)

prices = dfx['gic']
del prices['index']
prices = prices.reset_index(drop=True)


#Functions to buy, sell and wait
def buy(btc_price, btc, money):
    if(money != 0):
        btc = (1 / btc_price ) * money
        money = 0
    return btc, money


def sell(btc_price, btc, money):
    if(btc != 0):
        money = btc_price * btc
        btc = 0
    return btc, money


def wait(btc_price, btc, money):
    # do nothing
    return btc, money


#Create actions, states tables

np.random.seed(1)

# set of actions that the user could do
actions = { 'buy' : buy, 'sell': sell, 'wait' : wait}

actions_to_nr = { 'buy' : 0, 'sell' : 1, 'wait' : 2 }
nr_to_actions = { k:v for (k,v) in enumerate(actions_to_nr) }

nr_actions = len(actions_to_nr.keys())
nr_states = len(prices)

# q-table = reference table for our agent to select the best action based on the q-value
q_table = np.random.rand(nr_states, nr_actions)


#Functions to get rewards and act upon action
def get_reward(before_btc, btc, before_money, money):
    reward = 0
    if(btc != 0):
        if(before_btc < btc):
            reward = 1
    if(money != 0):
        if(before_money < money):
            reward = 1
            
    return reward

def choose_action(state):
    if np.random.uniform(0, 1) < eps:
        return np.random.randint(0, 2)
    else:
        return np.argmax(q_table[state])

def take_action(state, action):
    return actions[nr_to_actions[action]](prices[state], btc, money)


def act(state, action, theta):
    btc, money = theta
    
    done = False
    new_state = state + 1
    
    before_btc, before_money = btc, money
    btc, money = take_action(state, action)
    theta = btc, money
    
    reward = get_reward(before_btc, btc, before_money, money)
    
    if(new_state == nr_states):
        done = True
    
    return new_state, reward, theta, done

reward = 0
btc = 0
money = 100

theta = btc, money

# exploratory
eps = 0.3

n_episodes = 20
min_alpha = 0.02

# learning rate for Q learning
alphas = np.linspace(1.0, min_alpha, n_episodes)

# discount factor, used to balance immediate and future reward
gamma = 1.0


#Steps for Q-network learning
##Agent starts in a state=0 takes an action and receives a reward
##Agent selects action by referencing Q-table with highest value (max) OR by random (epsilon, ε)
##Update q-values


rewards = {}

for e in range(n_episodes):
    
    total_reward = 0
    
    state = 0
    done = False
    alpha = alphas[e]
    
    while(done != True):

        action = choose_action(state)
        next_state, reward, theta, done = act(state, action, theta)
        
        total_reward += reward
        
        if(done):
            rewards[e] = total_reward
            print(f"Episode {e + 1}: total reward -> {total_reward}")
            break
        
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma *  np.max(q_table[next_state]) - q_table[state][action])

        state = next_state


plt.ylabel('Total Reward')
plt.xlabel('Episode')
plt.plot([rewards[e] for e in rewards.keys()])

state = 0
acts = np.zeros(nr_states)
done = False

while(done != True):

        action = choose_action(state)
        next_state, reward, theta, done = act(state, action, theta)
        
        acts[state] = action
        
        total_reward += reward
        
        if(done):
            break
            
        state = next_state

buys_idx = np.where(acts == 0)
wait_idx = np.where(acts == 2)
sell_idx = np.where(acts == 1)

plt.figure(figsize=(15,15))
plt.plot(buys_idx[0], prices[buys_idx[0]], 'bo', markersize=2)
plt.plot(sell_idx[0], prices[sell_idx[0]], 'ro', markersize=2)
plt.plot(wait_idx[0], prices[wait_idx[0]], 'yo', markersize=2)
