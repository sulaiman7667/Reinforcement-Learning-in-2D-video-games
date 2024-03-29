import gym

import matplotlib.pyplot as plt

from DQN_Agent import DQN_Agent


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQN_Agent()
    agent.QTrainer(env)

