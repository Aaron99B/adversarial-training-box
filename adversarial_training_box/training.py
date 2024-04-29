import torch
import torch.optim as optim
import torchvision
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

from adversarial_training_box.models.cnn_dropout import Net
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict

def train(training_parameters: AttributeDict, criterion: torch.nn.Module, network: torch.nn.Module, train_loader: torch.utils.data.DataLoader, experiment_tracker: ExperimentTracker):
    
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


def test(network: torch.nn.Module, criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
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

    experiment_tracker.save_model(network, data)


def test_accuracy_class_wise(network: torch.nn.Module, test_loader: torch.utils.data.DataLoader, experiment_tracker: ExperimentTracker):
    classes = [0,1,2,3,4,5,6,7,8,9]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    class_accuracies = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracies.append([classname, round(accuracy, 1)])

    experiment_tracker.save_class_accuracy_table(class_accuracies)


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

    train(training_parameters, criterion, network, train_loader, experiment_tracker)

    batch_size_test = 100

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                                transform=mnist_standard_transform),
    batch_size=batch_size_test, shuffle=True)


    test(network, criterion, test_loader)
    #test_accuracy_class_wise(network, test_loader, experiment_tracker)