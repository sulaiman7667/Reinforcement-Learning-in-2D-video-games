import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Linear_Qnet(nn.Module):
    def __init__(self):
        super(Linear_Qnet, self).__init__()

        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        actions = self.layer3(x)
        
        return actions