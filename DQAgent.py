import os
#turning on local GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#importing dependencies


import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

env = gym.make('GuessingGame-v0') # initialise environment

#we need to change the four from hard code later
#this is just the way we plan to move.
state_size = 4 #env.observation_space.n
print(state_size)

action_size = env.action_space.shape[0]
print(action_size)

batch_size = 32 #running thing concurently really matters if running gpu

n_episodes = 1001 # n games we want agent to play (default 1001) the more games we play the more data we get for training
#each episode we learn little bit and we pass it on to train with.

output_dir = 'model_output/guessinggame/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def does_the_csv_exist():
    # if file does not exist write header 
    if not os.path.isfile(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\guess.csv'):
        d = {'guess': []}
        df = pd.DataFrame(data=d)
        df.to_csv('guess.csv',index=False)
    else: # else it exists so append without writing the header
        pass
does_the_csv_exist()

'''Define the Agent'''

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end 
        #memory saved from episodes. saves us from going over the full episode and looks at samples randomly thought game rather than contious
        #sampling will also add diversity so it doesnt always start with going up or down. deque is a list that cuts down last 2000 memories
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 400 actions, so 400 output neurons: -4 and 4 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis = 0)
        act_values = self.model.predict(np.array(state)) # if not acting randomly, predict reward value based on current state
        #AttributeError'int' object has no attribute 'ndim'
        return np.argmax(act_values[0]) # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                #print(next_state)
                #print(type(next_state))
                #next_state = np.dtype('int64').type(next_state)
                #print(next_state)
                next_state = np.expand_dims(next_state, axis = 0) #expected dense_1_input to have 2 dimensions, # Remove this line below, as it would set back shape to 3
                #print(self.model.predict(np.array(next_state)))
                target = (reward + self.gamma * np.amax(self.model.predict(np.array(next_state))[0])) # (maximum target Q based on future action a')
            state = np.expand_dims(state, axis = 0) # repeat process to add dim
            #print(self.model.predict(np.array(state))) #slap on np.array to convert from int
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


'''initalize agen'''
agent = DQNAgent(state_size, action_size) # initialise agent
'''Interact with environment'''
done = False #default is game has not ended.
for e in range(n_episodes): # iterate over new episodes of the game
    state = env.reset() # reset state at start of each new episode of the game
    print(state)
    #state = np.reshape(state, [0, state_size]) #just changing from a row to a column, simple transpose
    for time in range(5000):  # time represents a frame of the game; goal is to stay within the 1% range. once its done it can go on for only 5000 steps e.g., 500 or 5000 timesteps
#         env.render() #moving my animation to render
        action = agent.act(state) # action is 0 - 4 (move up, down, equal or no guess); decide on one or other here
        next_state, reward, done, _ = env.step(action) #env.step(np.array([guess])) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
        reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        #next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state        
        if done: # episode ends if agent is outside 1% or we reach timestep 5000
            print("episode: {}/{}, score: {}, e: {:.2}" # print the episode's score and agent's epsilon 
                  .format(e, n_episodes, time, agent.epsilon)) #if agent isnt working well epsoline is where we should look
            break # exit loop
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5") #saving every 50 episode. we can decide which model weight to hold onto