import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_RELU_6_100(nn.Module):
    def __init__(self, name = "mnist_relu_6_100"):
        super(MNIST_RELU_6_100, self).__init__()
        self.name = name
        self.layer1 = nn.Linear(784, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.layer5 = nn.Linear(100, 100)
        self.layer6 = nn.Linear(100, 100)
        self.layer7 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = self.layer7(x)

        return x
