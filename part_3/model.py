import torch.nn as nn

class Conv_Qnet(nn.Module):

    def __init__(self):
        super(Conv_Qnet, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size()[0], -1)
        x = self.linear4(x)
        x = self.relu4(x)
        actions = self.linear5(x)

        return actions

class Dueling_Conv_Qnet(nn.Module):

    def __init__(self):
        super(Dueling_Conv_Qnet, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size()[0], -1)
        x = self.linear4(x)
        x = self.relu4(x)
        V = self.V(x)
        A = self.A(x)

        return V, A