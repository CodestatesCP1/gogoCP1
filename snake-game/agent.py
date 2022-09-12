import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


class Agent:
    def __init__(self, hyperparams):
        self.epsilon = hyperparams['epsilon']
        self.gamma = hyperparams['gamma']
        self.batch_size = hyperparams['batch_size']
        self.learning_rate = hyperparams['learning_rate']
        self.kernel_size = hyperparams['kernel_size']
        self.input_shape = hyperparams['input_shape']
        self.action_space = np.array([0, 1, 2, 3])
        self.memory = deque(maxlen=2500)

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=(self.kernel_size, self.kernel_size),
                         padding='same',
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=64,
                         kernel_size=(self.kernel_size, self.kernel_size),
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=64,
                         kernel_size=(self.kernel_size, self.kernel_size),
                         padding='same',
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_dqn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, 1 if done else 0)

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        if not done:
            rewards = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)

        y = self.model.predict_on_batch(states)

        index = np.array([i for i in range(self.batch_size)])
        y[[index], [actions]] = rewards

        self.model.fit(states, y, epochs=1, verbose=0)

    # 아직 학습된 모델이 없으므로 랜덤 예측
    def predict_reward(self, state):
        return self.model.predict(np.array([state]))

    def policy(self, state):
        expected_reward = self.predict_reward(state)[0]
        best_Q_action = np.argmax(expected_reward)

        p = [self.epsilon / (len(self.action_space)-1) for _ in range(len(self.action_space))]
        p[best_Q_action] = 1 - self.epsilon

        return np.random.choice(self.action_space, p=p)
