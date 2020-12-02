import os
#turning on local GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # o refers to one device, 1 refers to 2 gpus

#importing dependencies
#new
#F:\Astra\conda2\envs\ReinforcementLearning\Lib\site-packages\gym\envs\toy_text
#env
#C:\Users\cgree\.conda\envs\keras-gpu\Lib\site-packages\gym\envs\toy_text
import random
import gym
#from gym import wrappers # import stack_frames
#from gym.wrappers import stack_frames
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import os # for creating directories
from keras.callbacks import TensorBoard 
import keras.backend.tensorflow_backend as backend
import tensorflow as tf
#import time
from tqdm import tqdm
import datetime
import os
import gym
#from gym import wrappers 
#from gym.wrappers.stack_frames import FrameStack
import pandas as pd

#just turning off a warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#os.chdir(r'C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all')

env = gym.make('GuessingGame-v0') # initialise environment
#env = FrameStack(env, 1) #I actually built this into the enviorment a while back but can find my change as I would rather us the wrapper. I will update this later. env shows 3 max/min
#we need to change the four from hard code later
#this is just the way we plan to move.
'''Data to include  would say: AL, AU, SymH, Bz, and Kp might be a good initial, include local time-magnetic local time, lat/long, in correlation to the earth and sun'''


state_size = 10 #with ravel #4 #env.observation_space.n #we changed this to one because ther is only one input for state as of now
# in order to get the four the states need to be gic, and three other variables proabaly
print(state_size)

action_size = env.action_space.shape[0]
print(action_size)

batch_size = 32 #480 #32 #running thing concurently really matters if running gpu, makes running the system more efficent by updating weights for more at one time

n_episodes = 5001 #1001 # n games we want agent to play (default 1001) the more games we play the more data we get for training
#each episode we learn little bit and we pass it on to train with.

output_dir = r'model_output/guessinggame/'

# tensorboard --logdir=C:\Users\cgree\Documents\Astra\Space_weather5_22\weakley_all\logs
#covnet dementions
MODEL_NAME = "24X4"
MODEL_NAME_NN = "basic_NN"
ep_rewards = [-2]
MIN_REWARD = -2 #for model save
MIN_REPLAY_MEMORY_SIZE = 1_000
DISCOUNT = 0.99 # DISCOUNT FUTURE REWARDS
UPDATE_TARGET_EVERY = 5
MINIBATCH_SIZE = 64

#stats settings
AGGREGATE_STATS_EVERY = 50 #EPISODES
Z = datetime.datetime.now()
Z = Z.strftime("%f")

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

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
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
                tf.summary.scalar(name, tf.reshape(value, []), step=index)
                self.step += 1
                self.writer.flush()
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                #tf.summary.scalar('loss',stats['loss'], step=self.step)
                tf.summary.scalar(name, tf.reshape(value, []), step=index)
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
        #this method gets trained every step
        self.model = self._build_model() # private method below (basically a regular net)

        #we are going to have two models due to this.
        #This model goes all over the place because it fits for every single step. 
        #this model we will do a .predict every model instead of doing a .fit every step like the self.model
        #this model predicts everytime to give the model some type of consistency to leverage off of.
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = ModifiedTensorBoard(log_dir="X:\\Final_RL\\logs\\{}-{}".format(MODEL_NAME, int(datetime.datetime.now().strftime("%f"))))#log_dir="logs\{}-{}".format(MODEL_NAME, int(Z)))#(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def _build_model(self):
        #LSTM model https://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory.pdf
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(6, input_dim=self.state_size, activation='relu')) # input layer; states as input
        model.add(Dropout(0.2)) #dropout
        model.add(Dense(12, activation='relu')) # hidden layer OG 24
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear')) # 400 actions, so 400 output neurons: -4 and 4 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy']) #original loss='mse'
        model.summary()
        #log_dir= f'models/{MODEL_NAME_NN}__{int(time.time())}.models'
        
        #log_dir=f"logs/{MODEL_NAME_NN}-{int(time.time())}"
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return model

    #replay memory add steps data to memory replay araray with daques
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
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatchsample
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
            #print(np.amax(self.model.predict(np.array(next_state))[0]))
            #print('prediciton****************')
            #print(np.amax(self.model.predict(np.array(next_state))))
            state = np.expand_dims(state, axis = 0) # repeat process to add dim
            #print(self.model.predict(np.array(state))) #slap on np.array to convert from int
            target_f = self.model.predict(state) #current qs list # approximately map current state to future discounted reward
            print(target_f.size)
            target_f = {action:target}
            target_f = list(target_f.keys())[0]
            target_f = np.array(target_f).reshape((1,))

            #target_f[action] = target

            #target_f[0][action] = target
            #log_dir= 'models/' + MODEL_NAME_NN + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #str(log_dir)
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.tensorboard]) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # #start training only if certain number of sample is already saved
        # if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
        #     return
        # minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        # #get current states from minibatch, then query NN model for Q values
        # #print(minibatch[0],minibatch[1],minibatch[3],minibatch[4])

        # state = np.array([transition[0] for transition in minibatch])
        # #state = np.expand_dims(state, axis = 0)
        # #state = state.ravel()
        # #print(state)
        # current_qs_list = self.model.predict(state)
        # #print(current_qs_list)
        # #current_qs_list = self.model.predict(np.array(state))
        # #current_qs_list = self.model.predict(state)(list(state), dtype=np.float)

        # new_current_states = np.array([transition[3] for transition in minibatch])
        # future_qs_list = self.target_model.predict(new_current_states)
        # x = []
        # y = []

        # #for index, (state, action, reward, new_current_states, done) in range(len(minibatch)):
        # for index, (state, action, reward, new_current_states, done) in enumerate(minibatch):

        #     if not done:
        #         max_future_q = np.max(future_qs_list[index])
        #         new_q = reward + DISCOUNT * max_future_q
        #     else:
        #         new_q = reward

        #     current_qs = current_qs_list[action]
        #     # print(current_qs_list)
        #     # print(type(current_qs_list))
        #     # print("goat")
        #     # print(type(current_qs))
        #     # print(current_qs.shape)
        #     # print(current_qs)
        #     # #current_qs = current_qs.tolist()
        #     #current_qs = new_q
        #     #current_qs[(int(round(action, 0)))] = new_q   #You may need to adjust back form int. just remove the int() but we also may need float
        #     #action = (action*-1)
        #     current_qs = new_q

        #     x.append(state)
        #     y.append(current_qs)
        #     print(y)

        #     self.model.fit(np.array(x), np.array(y), batch_size = MINIBATCH_SIZE, 
        #     verbose=0, shuffle=False, callbacks=[self.tensorboard])

        #     #updating to derermin if we want to update target_model yet
        #     if self.target_update_counter > UPDATE_TARGET_EVERY:
        #         self.target_model.set_weights(self.model.get_weights)
        #         self.target_update_counter = 0

        #     if self.epsilon > self.epsilon_min:
        #         self.epsilon *= self.epsilon_decay
        '''
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                next_state = np.expand_dims(next_state, axis = 0) #expected dense_1_input to have 2 dimensions, # Remove this line below, as it would set back shape to 3
                #print(self.model.predict(np.array(next_state)))
                target = (reward + self.gamma * np.amax(self.model.predict(np.array(next_state))[0])) # (maximum target Q based on future action a')
            state = np.expand_dims(state, axis = 0) # repeat process to add dim
            #print(self.model.predict(np.array(state))) #slap on np.array to convert from int
            #update Q value for given state
            target_f = self.model.predict(state) #current qs list # approximately map current state to future discounted reward
            target_f[0][action] = target
            #log_dir= 'models/' + MODEL_NAME_NN + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            x.append(state)
            y.append(target_f)
            #str(log_dir)
            #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            print(state)
            print(target_f)
            print(x)
            print(y)
            print(np.array)
            #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            self.model.fit(x, y, batch_size = 64, epochs=1, verbose=0, shuffle=False, callbacks=[self.tensorboard])
            #self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.tensorboard]) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        '''

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


'''initalize agent'''
agent = DQNAgent(state_size, action_size) # initialise agent
'''Interact with environment'''
done = False #default is game has not ended.
reward_total = 0
for e in tqdm(range(1, n_episodes + 1 ), ascii=True, unit='eposodes'): # iterate over new episodes of the game

    # Update tensorboard step every episode
    agent.tensorboard.step = e
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # reset state at start of each new episode of the game
    state = env.reset() 
    
    a = 480 # 4.80
    b = -480 #-4.80

    range_high = 480
    range_low = -480
    #print(state)
    #state = np.reshape(state, [0, state_size]) #just changing from a row to a column, simple transpose
    for time in range(5000):  # time represents a frame of the game; goal is to stay within the 1% range. once its done it can go on for only 5000 steps e.g., 500 or 5000 timesteps
        #         env.render() #moving my animation to render
        action = np.random.uniform(range_high, range_low)
        #
        #action = (a + b)/2
        action = round(action)
        next_state, reward, done, info = env.step(np.array([action]))
        reward_total = reward_total + reward
        #Chaing parameters depending on if the guess is higher or lower
        if(next_state[0]==1):
            print(action," is lower than the target")
            range_low=action
            #b=action
        if(next_state[0]==2):
            print("Correct Value reached:",action)
            break
        if(next_state[0]==3):
            print(action," is higher than the target")
            range_high=action
            #a=action

        # action = agent.act(np.array(state)) # action is 0 - 4 (move up, down, equal or no guess); decide on one or other here
        # next_state, reward, done, info = env.step(np.array(action)) 
        # #action = agent.act(state)
        # # print("action between 0-4. it should only be 0 for the first run due to not seeing the state")
        # #next_state, reward, done, info = env.step(np.array(action)) 


        agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
        state = next_state # set "current state" for upcoming iteration to the current next state
        env.render(action)        
        if done: # episode ends if agent is outside 1% or we reach timestep 5000
            
            #print("episode: {}/{}, number of guesses-score: {}, e: {:.2}" # print the episode's score and agent's epsilon 
            #      .format(e, n_episodes, time, agent.epsilon)) #if agent isnt working well epsoline is where we should look
            break # exit loop
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
    #if e % 100 == 0:
    #    agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5") #saving every 50 episode. we can decide which model weight to hold onto
    # print("next state all the data")
    # print(state)
    #append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(reward)
    if not e % AGGREGATE_STATS_EVERY or e == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=agent.epsilon, reward_total=reward_total)

        #save model, but only when min reward is greater or equal a set value
        # try:
        if min_reward >= MIN_REWARD: #>= MIN_REWARD:
            #print((f'X:\\Final_RL\\models\\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(datetime.datetime.now().strftime("%f"))}.model'))
            #agent.model.save(f'X:\\Final_RL\\models\\x.model')
            max_reward = int(max_reward)
            average_reward = int(average_reward)
            min_reward = int(min_reward)
            reward_total = int(reward_total)
                #agent.model.save((f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'))
            #print(str(f'X:\\Final_RL\\models\\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(datetime.datetime.now().strftime("%f"))}.model'))
            agent.model.save(f'X:\\Final_RL\\models\\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}reto_{reward_total:_>7.2f}min__{int(datetime.datetime.now().strftime("%f"))}.model')
        # except AttributeError as e:
        #     print(e)
        #     pass


        #tensorboard --port=6006 --logdir=X:\Final_RL\models
        #tensorboard --port=6007 --logdir=X:\Final_RL\logs

