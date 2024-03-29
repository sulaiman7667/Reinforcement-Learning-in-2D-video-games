import os
import sys
import time

import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Conv_Qnet
from DQN_Agent import DQN_Agent
from Double_DQN_Agent import Double_DQN_Agent
from Dueling_DQN_Agent import Dueling_DQN_Agent


def plot(scores_1, scores_2, scores_3):
    plt.clf()
    plt.xlabel('Number of Episodes / 10')
    plt.ylabel('Average Score')
    plt.plot(scores_1, color='r', label='DQN')
    plt.plot(scores_2, color='g', label='Double DQN')
    plt.plot(scores_3, color='b', label='Dueling DQN')
    plt.ylim(ymin=0)
    plt.legend()
    plt.savefig('Training results.pdf')

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == 'test':
        # select model type
        if T.cuda.is_available():
            model_1 = T.load('pretrained_models/model_1900000_iterations.pth').eval()
            model_2 = T.load('pretrained_models/model_1900000_iterations_double.pth').eval()
            model_3 = T.load('pretrained_models/model_1900000_iterations_dueling.pth').eval()
        else:
            pass
            # model = T.load('pretrained_models/model_5000000_iterations.pth', map_location = 'cpu').eval()
            # model = T.load('pretrained_models/model_5000000_iterations_double.pth', map_location = 'cpu').eval()
            # model = T.load('pretrained_models/model_5000000_iterations_dueling.pth', map_location = 'cpu').eval()


        # select agent
        agent_1 = DQN_Agent()
        agent_2 = Double_DQN_Agent()
        agent_3 = Dueling_DQN_Agent()

        model_1 = agent_1.convert_to_cuda(model_1)
        model_2 = agent_2.convert_to_cuda(model_2)
        model_3 = agent_3.convert_to_cuda(model_3)

        print("Testing, Wait for results...")

        win_rate, avg_score = agent_1.QTest(model_1)
        print("Agent: DQN", "Win rate:", win_rate, "Average Score:", avg_score)

        win_rate, avg_score = agent_2.QTest(model_2)
        print("Agent: Double DQN", "Win rate:", win_rate, "Average Score:", avg_score)

        win_rate, avg_score = agent_3.QTest(model_3)
        print("Agent: Dueling DQN", "Win rate:", win_rate, "Average Score:", avg_score)

    elif mode == 'train':
        if not os.path.exists('pretrained_models/'):
            os.mkdir('pretrained_models/')
            
        agent = DQN_Agent()
        agent.QTrainer()
        scores_1 = agent.scores

        agent = Double_DQN_Agent()
        agent.QTrainer()
        scores_2 = agent.scores

        agent = Dueling_DQN_Agent()
        agent.QTrainer()
        scores_3 = agent.scores

        plot(scores_1, scores_2, scores_3)