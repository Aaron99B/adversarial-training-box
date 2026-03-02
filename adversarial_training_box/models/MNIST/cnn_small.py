import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_SMALL(nn.Module):
    def __init__(self, input_height=28, input_width=28, num_classes=10,name="cnn_small"):
        """
        Basic convolutional neural network with a convultional connection and a linear connection
            input layer: 1 for loading the whole image
            hidden layer: 50
            output layer: for MNIST 10 (digits 0 to 9)

        Parameters:
            input_height & input_width: int
                The size of the image
            num_classes: int
                The number of classes we want to predict
        """
        super(CNN_SMALL, self).__init__()
        self.name = name
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU()) # Convolutional layer with 32 filters
        self.fc1 = nn.Linear(32 * input_height * input_width, 128)
        self.fcoutput = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = x.view(-1, 1, 28, 28)  # Reshape input to [batch_size, 1, 28, 28]
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.fcoutput(x)
        return x