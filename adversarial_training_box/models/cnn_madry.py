import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture from Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks

class CNNMADRY(torch.nn.Module):

    def __init__(self):
        self.name = "cnn_madry"
        super(CNNMADRY, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x