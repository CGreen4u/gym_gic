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
from keras.callbacks import TensorBoard 
import tensorflow as tf
import time
from tqdm import tqdm
import datetime

env = gym.make('GuessingGame-v0') # initialise environment

#we need to change the four from hard code later
#this is just the way we plan to move.
'''Data to include  would say: AL, AU, SymH, Bz, and Kp might be a good initial, include local time-magnetic local time, lat/long, in correlation to the earth and sun'''


state_size = 4 #env.observation_space.n #we changed this to one because ther is only one input for state as of now
# in order to get the four the states need to be gic, and three other variables proabaly
print(state_size)

action_size = env.action_space.shape[0]
print(action_size)

batch_size = 32 #32 #running thing concurently really matters if running gpu, makes running the system more efficent by updating weights for more at one time

n_episodes = 1001 #1001 # n games we want agent to play (default 1001) the more games we play the more data we get for training
#each episode we learn little bit and we pass it on to train with.

output_dir = 'model_output/guessinggame/'

# tensorboard --logdir=C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\logs
#covnet dementions
MODEL_NAME = "24X4"
MODEL_NAME_NN = "basic_NN"
ep_rewards = [-2]
MIN_REWARD = -2 #for model save

#stats settings
AGGREGATE_STATS_EVERY = 50 #EPISODES


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

# Own Tensorboard class
#this saves us from creating a new file. we just need to update the file
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        #self.log_dir = f"logs/{MODEL_NAME}-{int(time.time())}"
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    #physically writes to the log files
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                #tf.summary.scalar('loss',stats['loss'], step=self.step)
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                #tf.summary.scalar('loss',stats['loss'], step=self.step)
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


'''Define the Agent'''

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000) # double-ended queue; acts like list, but elements can be added/removed from either end 
        #memory saved from episodes. saves us from going over the full episode and looks at samples randomly thought game rather than contious
        #sampling will also add diversity so it doesnt always start with going up or down. deque is a list that cuts down last 2000 memories
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 #0.99975 #0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.001 #0.0385 #0.01 # minimum amount of random exploration permitted
        self.learning_rate = 1e-3 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method below (basically a regular net)

        self.tensorboard = ModifiedTensorBoard(log_dir="logs\{}-{}".format(MODEL_NAME, int(time.time())))#(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0


    def _build_model(self):
        #LSTM model https://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory.pdf
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # input layer; states as input
        model.add(Dense(24, activation='relu')) # hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 400 actions, so 400 output neurons: -4 and 4 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']) #original loss='mse'
        
        #log_dir= f'models/{MODEL_NAME_NN}__{int(time.time())}.models'
        
        #log_dir=f"logs/{MODEL_NAME_NN}-{int(time.time())}"
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
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
                #print(state)
                next_state = np.expand_dims(next_state, axis = 0) #expected dense_1_input to have 2 dimensions, # Remove this line below, as it would set back shape to 3
                #print(self.model.predict(np.array(next_state)))
                target = (reward + self.gamma * np.amax(self.model.predict(np.array(next_state))[0])) # (maximum target Q based on future action a')
            state = np.expand_dims(state, axis = 0) # repeat process to add dim
            #print(self.model.predict(np.array(state))) #slap on np.array to convert from int
            target_f = self.model.predict(state) #current qs list # approximately map current state to future discounted reward
            target_f[0][action] = target
            #log_dir= 'models/' + MODEL_NAME_NN + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #str(log_dir)
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.tensorboard]) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


'''initalize agent'''
agent = DQNAgent(state_size, action_size) # initialise agent
'''Interact with environment'''
done = False #default is game has not ended.
for e in tqdm(range(n_episodes)): # iterate over new episodes of the game

    agent.tensorboard.step = e
    episode_reward = 0
    step = 1

    state = env.reset() # reset state at start of each new episode of the game
    #print(state)
    #state = np.reshape(state, [0, state_size]) #just changing from a row to a column, simple transpose
    for time in range(5000):  # time represents a frame of the game; goal is to stay within the 1% range. once its done it can go on for only 5000 steps e.g., 500 or 5000 timesteps
#         env.render() #moving my animation to render
        action = agent.act(np.array(state)) # action is 0 - 4 (move up, down, equal or no guess); decide on one or other here
        next_state, reward, done, info = env.step(np.array(action)) #env.step(np.array([guess])) # agent interacts with env, gets feedback; 4 state data points, e.g., high or low - observation - later the variables and  done, reward        
        #print(info)
        #reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        #next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state        
        if done: # episode ends if agent is outside 1% or we reach timestep 5000
            #print("episode: {}/{}, number of guesses-score: {}, e: {:.2}" # print the episode's score and agent's epsilon 
            #      .format(e, n_episodes, time, agent.epsilon)) #if agent isnt working well epsoline is where we should look
            break # exit loop
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
    #if e % 100 == 0:
    #    agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5") #saving every 50 episode. we can decide which model weight to hold onto

    #append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(reward)
    if not e % AGGREGATE_STATS_EVERY or e == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon)

        #save model, but only when min reward is greater or equal a set value
        try:
            if min_reward >=average_reward: #>= MIN_REWARD:
                #agent.model.save((f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'))
                agent.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        except AttributeError as e:
            print(e)
            pass