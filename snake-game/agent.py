import os
import random
import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


class Agent:
    def __init__(self, hyperparams):
        self.epsilon = hyperparams['epsilon']
        self.epsilon_min = hyperparams['epsilon_min']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.gamma = hyperparams['gamma']
        self.batch_size = hyperparams['batch_size']
        self.learning_rate = hyperparams['learning_rate']
        self.kernel_size = hyperparams['kernel_size']
        self.input_shape = hyperparams['input_shape']
        self.action_space = np.array([0, 1, 2, 3])
        self.memory = deque(maxlen=100000) if 'memory' not in hyperparams else hyperparams['memory']

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        #hyperparam
        self.target_update_freq = hyperparams['target_update_freq']
        
        self.target_update_counter = 0
        

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=(self.kernel_size, self.kernel_size),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(Conv2D(filters=32,
                         kernel_size=(self.kernel_size, self.kernel_size),
                         activation='relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(4))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_dqn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch])
        # actions = np.array([sample[1] for sample in minibatch])
        # rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        # dones = np.array([sample[4] for sample in minibatch])

        # rewards = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)

        # y = self.model.predict_on_batch(states)

        # index = np.array([i for i in range(self.batch_size)])
        # y[[index], [actions]] = rewards
        
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        self.model.fit(states, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_reward(self, state):
        return self.model.predict(np.array([state]))

    def policy(self, state):
        expected_reward = self.predict_reward(state)
        best_Q_action = np.argmax(expected_reward)

        p = [self.epsilon / (len(self.action_space)-1) for _ in range(len(self.action_space))]
        p[best_Q_action] = 1 - self.epsilon

        return np.random.choice(self.action_space, p=p)
    
    def increase_target_update_counter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    

    # don't use
    def save(self, model_path):
        pass
