import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_RELU_4_1024(nn.Module):
    def __init__(self):
        super(MNIST_RELU_4_1024, self).__init__()
        self.name = "mnist_relu_4_1024"
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return x