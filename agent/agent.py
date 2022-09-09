""" 
CNN 구성을 위한 라이브러리 추가 import 
keras.layers  -> Conv2D, MaxPooling2D, Flatten
tensorflow
"""

import random
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

import os
import sys

""" 경로 지정 후 snake game 모듈 불러오기...는 나중에
game.py와 gameclasses.py에서 뱀의 위치(a), 뱀 머리(b), 먹이 위치(c)를 반환하는 함수를 만들면
agent에서 import해서 state = (a, b, c)로 가져와서 
CNN에 넣을 수 있을 것 같습니다.
"""
# sys.path.append('/gogoCP1/snake_game')
                
# from snake_game import game 
# from snake_game import gameclasses 

# from plot_script import plot_result
# from snake_env import Snake


""" 
1. DQN 클래스 구성
    - init
    - 모델 구성 함수 (CNN 사용)
    - 행동 value 반환 함수
    - Q 함수 및 Q 함수 학습 함수
2. DQN 훈련 함수
3. if __name__ == '__main__': 학습 실행

"""
class DQN():
    
    # 게임에서 state 정보를 가져오고, 설정한 모델 불러오기
    def __init__(self, env, params):
        
        """ env에서 가져오기"""
        # action : U, L, D, R
        self.action_space = 4
        
        # state_space : (뱀 있음?, 뱀 머리임?, 먹이임?)
        #self.state_space = env.state_space  -- 
        
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        
        self.memory = deque(maxlen=2500)
        self.model = self.build_CNN()
        
        
    # CNN 모델 구성
    def build_CNN(self):
        model = Sequential()
        
        """ CNN 모델 만들기
        board가 길이 30*30으로 이루어진 한 칸이 가로 세로 20개씩 구성되어 있고,
        state의 상태를 3개 채널로 입력받으므로
        input_shape = (20, 20, 3)
        
        나머지 filters, kernel_size, strides, padding은 임의로 지정.
        은닉층의 개수도 일단 임의로 지정. (향후 필요에 따라 추가합시다.)
        
        compile의 loss는 일단 레퍼런스 코드에 있는거 그대로 가져옴.
        """
        # 입력층.   input_shape()???
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(20, 20, 3)))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # 은닉층
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # 출력층 : 출력층은 U, D, L, R의 확률값을 각각 출력해야 하므로, action_space(4)에 activation은 softmax
        model.add(Dense(self.action_space, activation='softmax'))
        
        # loss 함수는 뭘로 해야할까요??
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    # 상태, 행동, 보상을 저장하기 위한 함수
    
    def remember(self, state, action, reward, next_state):
        
        # 레퍼런스 코드에서는 done은 벽이나 몸에 닿았을 때 True가 됨.
        # game.py에서는 reward=-1로 죽는걸 나타냈으므로 이에 맞게 수정?        
        self.memeory.append((state, action, reward, next_state, done))
    
    # state 상태를 받아 다음 action value를 도출하는 함수
    def act(self, state):
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    
    # Q 함수 업데이트 함수
    def q_function(self):
        
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1-dones)
        
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# agemt 훈련 함수 -- episode : 훈련을 진행할 에피소드, env는 snake 게임 run하는 객체
def train_agent(episode, env):
    
    sum_of_rewards = []
    agent = DQN(env, params)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        
        max_steps = 10000
        for i in range(max_steps):
            action = agent.act(state)
            prev_state = state
            
            next_state, reward, done, _ = env.step(action)
            score += reward
            
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if params['batch_size'] > 1:
                agent.replay()
            
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {e+1}/{episode}, score: {score}')
                break
        
        sum_of_rewards.append(score)

    return sum_of_rewards



"""
이 밑에서는 game.py를 가져와서 실행시킨 다음, 
학습 파라미터를 설정 
+
agent가 플레이하고 그 결과를 반환하는 코드를 구현해야 함!    

(+ 동영상으로 추출하는 코드도 추가?)
"""

if __name__ == '__main__':
    
    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    
    results = dict()
    ep = 50
    
    # env : 게임 실행 객체  -- 불러오려면 class?
    # env = 
    
    sum_of_rewards = train_agent(ep, env)
    results[params['name']] = sum_of_rewards
    
    # plot_result(results, direct=True, k=20) : 학습 결과 matplotlib으로 시각화