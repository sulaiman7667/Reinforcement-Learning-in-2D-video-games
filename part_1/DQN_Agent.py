import copy
import time

import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import Linear_Qnet


class DQN_Agent:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_dec = 1e-3
        self.lr = 0.001
        self.mem_size = 10000
        self.batch_size = 64

        self.model = Linear_Qnet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        self.batch_index = np.arange(self.batch_size, dtype=np.int32)
        self.mem_counter = 0
        self.input_size = 4
        self.n_actions = 2
        self.scores = []

        self.state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminated_memory = np.zeros(self.mem_size, dtype=bool)


    def plot(self, scores):
        plt.clf()
        plt.xlabel('Number of Episodes')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.pause(.01)
        plt.savefig('Training results.pdf')


    def save(self, state, action, reward, new_state, terminated):
        index = self.mem_counter % self.mem_size
        self.mem_counter += 1

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminated_memory[index] = terminated


    def get_action(self, state):
        state = state.astype('float32')
        if np.random.random() > self.epsilon:
            state = T.tensor(state)
            actions = self.model(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.randint(self.n_actions)

        return action


    def random_batch(self):
        memory_len = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(memory_len, self.batch_size)

        state_batch = T.tensor(self.state_memory[batch])
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch])
        new_state_batch = T.tensor(self.new_state_memory[batch])
        terminated_batch = T.tensor(self.terminated_memory[batch])

        return state_batch, action_batch, reward_batch, new_state_batch, terminated_batch


    def QTrainer(self, env):
        avg_scores = []
        episodes = 500
        total_score = 0
        iteration = 0
        start = time.time()
        for episode in range(episodes):
            state = env.reset()
            while True:
                env.render()

                action = self.get_action(state)

                new_state, reward, terminated, _ = env.step(action)

                self.save(state, action, reward, new_state, terminated)

                state_batch, action_batch, reward_batch, new_state_batch, terminated_batch = self.random_batch()

                q_pred = self.model(state_batch)[self.batch_index, action_batch]
                q_new_states = self.model(new_state_batch)

                q_target = []
                for i in range(self.batch_size):
                    if terminated_batch[i]:
                        q_target.append(reward_batch[i])
                    else:
                        action = T.argmax(q_new_states[i])
                        q_target.append(reward_batch[i] + self.gamma * (q_new_states[i][action]))
                q_target = T.stack(q_target)

                self.optimizer.zero_grad()
                loss = self.criterion(q_target, q_pred)
                loss.backward()
                self.optimizer.step()

                self.epsilon = self.epsilon - self.eps_dec
                if self.epsilon < 0:
                    self.epsilon = 0
                total_score += reward
                iteration += 1
                if terminated:
                    self.scores.append(total_score)
                    self.plot(self.scores)
                    print("iteration:", iteration, "elapsed time:", int(time.time() - start), "episode:", episode, "Score:", total_score, "epsilon:", self.epsilon)
                    total_score = 0
                    break
                else:
                    state = new_state

