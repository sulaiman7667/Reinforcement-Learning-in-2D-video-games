import random
import time
import cv2

import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import Conv_Qnet
from collections import deque


class DQN_Agent:
    def __init__(self, game_env):
        self.replay_memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.1
        self.final_epsilon = 0.01
        self.lr = 1e-6
        self.number_of_iterations = 2000000
        self.batch_size = 32
        self.number_of_frames = 4

        self.model = self.convert_to_cuda(Conv_Qnet())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.number_of_actions = 2
        self.game_env = game_env
        self.decay = np.linspace(self.epsilon, self.final_epsilon, self.number_of_iterations)
        self.batch_index = np.arange(0, self.batch_size)
        self.scores = []


    def plot(self, scores):
        plt.clf()
        plt.xlabel('Number of Episodes / 100')
        plt.ylabel('Average Score')
        plt.title("DQN FlappyBird")
        plt.plot(scores)
        plt.ylim(ymin=0)
        plt.grid()
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.pause(.01)
        plt.savefig('DQN Agent results.pdf')


    def convert_to_cuda(self, input):
        if T.cuda.is_available():
            input = input.cuda()
        return input


    def refactor_img(self, img):
        img = cv2.resize(img, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 0] = 255
        img = np.expand_dims(img, axis = 0)
        img = T.tensor(img, dtype = T.float32)
        img = self.convert_to_cuda(img)
        return img


    def get_action(self, state):
        action = T.FloatTensor([0,0])
        action = self.convert_to_cuda(action)
        if random.random() < self.epsilon:
            action[random.randint(0, 1)] = 1
        else:
            output = self.model(state)
            action[T.argmax(output)] = 1

        return action


    def play_action(self, action, state):
        image_pixels, reward, terminated = self.game_env.frame_step(action)
        image_pixels = self.refactor_img(image_pixels)
        state = state[0][:][1:self.number_of_frames]
        new_state = T.cat((state, image_pixels)).unsqueeze(0)
        reward = T.tensor([reward]).unsqueeze(0)
        return new_state, reward, terminated


    def initial_state(self):
        action = T.FloatTensor([1,0])
        image_pixels, reward, terminated = self.game_env.frame_step(action)
        image_pixels = self.refactor_img(image_pixels)
        state = []
        for i in range(self.number_of_frames):
            state += tuple(image_pixels)
        state = T.stack(state).unsqueeze(0)
        return state


    def random_batch(self):
        sample_size = min(len(self.replay_memory), self.batch_size)
        batch = random.sample(self.replay_memory, sample_size)
        batch_cols = list(zip(*batch))
        state_batch = self.convert_to_cuda(T.stack(batch_cols[0]).squeeze(1))
        action_batch = self.convert_to_cuda(T.stack(batch_cols[1]).squeeze(1))
        reward_batch = self.convert_to_cuda(T.stack(batch_cols[2]).squeeze(1))
        new_state_batch = self.convert_to_cuda(T.stack(batch_cols[3]).squeeze(1))
        terminated_batch = batch_cols[4]

        return state_batch, action_batch, reward_batch, new_state_batch, terminated_batch


    def QTrainer(self):
        total_score = 0
        score = 0
        episode = 0
        state = self.initial_state()
        start = time.time()

        for iteration in range(self.number_of_iterations):
            action = self.get_action(state)

            new_state, reward, terminated = self.play_action(action, state)

            self.replay_memory.append((state, action.unsqueeze(0), reward, new_state, terminated))

            state_batch, action_batch, reward_batch, new_state_batch, terminated_batch = self.random_batch()

            q_pred = self.model(state_batch)
            q_pred = q_pred * action_batch
            q_pred = T.sum(q_pred, dim=1)


            q_new_states = self.model(new_state_batch)

            q_target = []
            for i in range(len(state_batch)):
                if terminated_batch[i]:
                    q_target += reward_batch[i]
                else:
                    action = T.argmax(q_new_states[i])
                    q_target += (reward_batch[i] + self.gamma * (q_new_states[i][action]))
            q_target = T.stack(q_target)

            self.optimizer.zero_grad()
            loss = self.criterion(q_target, q_pred)
            loss.backward()
            self.optimizer.step()

            self.epsilon = self.decay[iteration]

            state = new_state

            if iteration % 100000 == 0:
                T.save(self.model, "pretrained_models/model_" + str(iteration) + "_iterations.pth")

            if self.game_env.score > score:
                score = self.game_env.score

            if terminated:
                total_score += score
                score = 0
                episode += 1
                if episode % 100 == 0:
                    self.scores.append(total_score/100)
                    self.plot(self.scores)
                    print("iteration:", iteration, "elapsed time:", int(time.time() - start), "episode:", episode, "Average score:", total_score/100, "epsilon:", self.epsilon)
                    total_score = 0


    def QTest(self, model):
        state = self.initial_state()
        self.epsilon = 0
        self.model = model
        max_score = 0
        tot_score = 0
        episode = 0
        score = 0

        while True:
            action = self.get_action(state)

            new_state, reward, terminated = self.play_action(action, state)

            state = new_state

            if self.game_env.score > score:
                score = self.game_env.score

            if (self.game_env.score > max_score):
                max_score = self.game_env.score

            if terminated:
                episode += 1
                tot_score += score
                score = 0
                print("Episode:", str(episode) + "/20")
                if episode == 20:
                    avg_score = tot_score/20
                    return max_score, avg_score
