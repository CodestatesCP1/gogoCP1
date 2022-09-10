import numpy as np


class Agent:
    def __init__(self):
        pass

    # 아직 학습된 모델이 없으므로 랜덤 예측
    def predict_reward(self, state):
        return [np.random.randint(-10, 11),
                np.random.randint(-10, 11),
                np.random.randint(-10, 11),
                np.random.randint(-10, 11)]

    def policy(self, state):
        expected_reward = self.predict_reward(state)
        return np.argmax(expected_reward)
