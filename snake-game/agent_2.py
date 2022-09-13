import os
import random
import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


class Agent2:
    def __init__(self, hyperparams):
        self.state_space = hyperparams['state_space']
        self.epsilon = hyperparams['epsilon']
        self.epsilon_min = hyperparams['epsilon_min']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.gamma = hyperparams['gamma']
        self.batch_size = hyperparams['batch_size']
        self.learning_rate = hyperparams['learning_rate']
        self.layer_sizes = hyperparams['layer_sizes']
        self.action_space = 4
        self.memory = deque(maxlen=2500) if 'memory' not in hyperparams else hyperparams['memory']

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def set_model(self, model):
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, self.state_space))
        next_state = np.reshape(next_state, (1, self.state_space))
        self.memory.append((state, action, reward, next_state, done))

    def train_dqn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        rewards = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1-dones)
        y = self.model.predict_on_batch(states)

        index = np.array([i for i in range(self.batch_size)])
        y[[index], [actions]] = rewards

        self.model.fit(states, y, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict_reward(self, state):
        return self.model.predict(np.array(np.reshape(state, (1, self.state_space))))

    def policy(self, state):
        expected_reward = self.predict_reward(state)[0]
        best_Q_action = np.argmax(expected_reward)

        p = [self.epsilon / (self.action_space-1) for _ in range(self.action_space)]
        p[best_Q_action] = 1 - self.epsilon

        return np.random.choice(range(self.action_space), p=p)

    def save(self, model_path):
        index = 1
        while True:
            model_file_path = model_path + f'model_{index}/model.h5'
            if not os.path.isfile(model_file_path):
                self.model.save(model_file_path)

                model_hyperparams_path = model_path + f'model_{index}/hyperparams.pkl'
                hyperparams = {
                    'state_space': self.state_space,
                    'epsilon': self.epsilon,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay,
                    'gamma': self.gamma,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'layer_sizes': self.layer_sizes,
                    'memory': self.memory
                }
                with open(model_hyperparams_path, 'wb') as pickle_file:
                    pickle.dump(hyperparams, pickle_file)
                break
            index += 1
