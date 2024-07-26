import torch.nn as nn
import torch.nn.functional as F

class CIFAR_6_500(nn.Module):
    def __init__(self):
        super(CIFAR_6_500, self).__init__()
        self.name = "cifar_6_500"

        self.fc1 = nn.Linear(3072, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 500)
        self.fc7 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x