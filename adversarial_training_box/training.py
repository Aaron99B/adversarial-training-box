import torch
import torch.optim as optim
import torchvision
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

from adversarial_training_box.models.cnn_dropout import Net
from adversarial_training_box.experiment_tracker import ExperimentTracker
from adversarial_training_box.attribute_dict import AttributeDict

def train(training_parameters: AttributeDict, criterion: torch.nn.Module, network: torch.nn.Module, experiment_tracker: ExperimentTracker):
    
    experiment_tracker.watch(network, criterion, log_option="all", log_frequency=10)

    network.train()
    for epoch in range(0,training_parameters.epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            experiment_tracker.log({"loss": loss.item(), "epoch" : epoch})

    experiment_tracker.save_model(network, data)


def test(network: torch.nn.Module, criterion: torch.nn.Module):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = network(data)
            test_loss += criterion(output, label, size_average=False).item()
            _, predicted = output.data.max(1, keepdim=True)
            correct += predicted.eq(label.data.view_as(predicted)).sum()
            total += label.size(0)
            test_loss /= len(test_loader.dataset)
            experiment_tracker.log({"test_loss": test_loss})
            
    experiment_tracker.log({"test_accuracy": 100. * correct / total})


if __name__ == "__main__":

    training_parameters = AttributeDict(epochs = 1,
        batch_size = 64,
        learning_rate = 0.01,
        momentum = 0.5,
        log_interval = 10,
        random_seed = 1)
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=training_parameters.learning_rate,
            momentum=training_parameters.momentum)
    criterion = F.nll_loss

    torch.manual_seed(training_parameters.random_seed)

    mnist_standard_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,))
                    ])

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=True, download=True,
                    transform=mnist_standard_transform),
    batch_size=training_parameters.batch_size, shuffle=True)

    experiment_tracker = ExperimentTracker("robust-training", Path("../generated"), training_parameters)

    train(training_parameters, criterion, network, experiment_tracker)

    batch_size_test = 100

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                                transform=mnist_standard_transform),
    batch_size=batch_size_test, shuffle=True)


    test(network, criterion)