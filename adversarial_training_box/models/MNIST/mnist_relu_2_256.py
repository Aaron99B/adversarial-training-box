import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_RELU_2_256(nn.Module):
    def __init__(self, name="mnist_net_2_256"):
        super(MNIST_RELU_2_256, self).__init__()
        self.name = name
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x