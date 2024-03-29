import copy
import os
import random
import cv2
import gym
import gym_super_mario_bros
import time

import torch as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from nes_py.wrappers import JoypadSpace
from model import Conv_Qnet
from collections import deque


class Double_DQN_Agent:
    def __init__(self):
        self.model = self.convert_to_cuda(Conv_Qnet())
        self.target_model = self.convert_to_cuda(Conv_Qnet())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025, eps=1e-4)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.replay_memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.replace_target = 10000
        self.number_of_iterations = 2000000

        self.scores = []
        self.number_of_actions = 2
        self.batch_index = np.arange(0, self.batch_size)
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, [["right"], ["right", "A"]])


    def plot(self, scores):
        plt.clf()
        plt.xlabel('Number of Episodes / 10')
        plt.ylabel('Average Score')
        plt.title("Double DQN Super Mario")
        plt.plot(scores)
        plt.ylim(ymin=0)
        plt.grid()
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.pause(.01)
        plt.savefig('Double DQN Agent Result.pdf')


    def convert_to_cuda(self, input):
        if T.cuda.is_available():
            input = input.cuda()
        return input


    def refactor_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[40:220, 0:256]
        img = cv2.resize(img, (84, 84))
        img = img/255
        img = T.tensor(img, dtype=T.float32)
        img = self.convert_to_cuda(img)
        return img


    def get_action(self, state):
        if random.random() < self.epsilon:
            action = np.random.randint(self.number_of_actions)
        else:
            output = self.model(state.unsqueeze(0))
            action = T.argmax(output).item()
        return action


    def step(self, action):
        total_reward = 0.0
        terminated = False
        for i in range(4):
            image_pixels, reward, terminated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return image_pixels, total_reward, terminated, info


    def play_action(self, action, state):
        image_pixels, reward, terminated, info = self.step(action)
        image_pixels = self.refactor_img(image_pixels).unsqueeze(0)
        state = state[:][1:4]
        new_state = T.cat((state, image_pixels))

        return new_state, reward, terminated, info


    def initial_state(self):
        image_pixels = self.env.reset()
        image_pixels = self.refactor_img(image_pixels)
        state = []
        for i in range(4):
            state.append(image_pixels)
        state = T.stack(state)

        return state


    def reset(self, state):
        image_pixels = self.env.reset()
        image_pixels = self.refactor_img(image_pixels).unsqueeze(0)
        state = state[:][1:4]
        state = T.cat((state, image_pixels))
        return state


    def random_batch(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        batch_cols = zip(*batch)
        state, action, reward, new_state, terminated = map(T.stack, batch_cols)
        return self.convert_to_cuda(state), self.convert_to_cuda(action.squeeze()), self.convert_to_cuda(reward.squeeze()), self.convert_to_cuda(new_state), self.convert_to_cuda(terminated.squeeze())


    def QTrainer(self):
        state = self.initial_state()
        tot_reward = 0
        start = time.time()
        episode = 0
        for iteration in range(self.number_of_iterations):
            action = self.get_action(state)
            self.env.render()
            new_state, reward, terminated, info = self.play_action(action, state)

            self.replay_memory.append((state, T.tensor(action), T.tensor(reward), new_state, T.tensor(terminated)))


            tot_reward += reward
            if(iteration < self.batch_size):
                continue


            state_batch, action_batch, reward_batch, new_state_batch, terminated_batch = self.random_batch()

            q_pred = self.model(state_batch)[self.batch_index, action_batch]

            q_new_states = self.model(new_state_batch)
            q_targ_states = self.target_model(new_state_batch)

            q_target = []
            for i in range(self.batch_size):
                if terminated_batch[i]:
                    q_target.append(reward_batch[i])
                else:
                    action = T.argmax(q_new_states[i])
                    q_target.append((reward_batch[i] + self.gamma * (q_targ_states[i][action])))
            q_target = T.stack(q_target).detach()


            self.optimizer.zero_grad()
            loss = self.criterion(q_target, q_pred)
            loss.backward()
            self.optimizer.step()


            state = new_state
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

            if iteration % self.replace_target == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if iteration % 100000 == 0:
                T.save(self.model, "pretrained_models/model_" + str(iteration) + "_iterations_double.pth")

            if terminated:
                episode += 1
                state = self.reset(state)
                if episode % 10 == 0:
                    self.scores.append(tot_reward/10)
                    self.plot(self.scores)
                    scores = []
                    print("iteration:", iteration, "elapsed time:", int(time.time() - start), "episode:", episode, "Average score:", int(tot_reward/10), "epsilon:", self.epsilon)
                    tot_reward = 0


    def QTest(self, model):
        state = self.initial_state()
        self.epsilon = 0
        self.model = model
        total_reward = 0
        episode = 0
        win_rate = 0

        while True:
            self.env.render()

            action = self.get_action(state)

            new_state, reward, terminated, info = self.play_action(action, state)

            total_reward += reward
            state = new_state
            if terminated:
                episode += 1
                state = self.reset(state)
                print("Episode:", str(episode) + "/100")
                if info['flag_get'] == True:
                    win_rate += 1
                if episode % 100 == 0:
                    avg_score = total_reward/100
                    total_reward = 0
                    return win_rate, avg_score
