import torch
import torch.optim as optim
import torchvision
from pathlib import Path
import torch.nn.functional as F

from adversarial_training_box.adversarial_attack.clever_hans_attack import CleverHansAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.adversarial_attack.foolbox_attack import FoolboxAttack
from adversarial_training_box.models.cnn_dropout import Net
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline


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


    batch_size_test = 128

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                                transform=mnist_standard_transform),
    batch_size=batch_size_test, shuffle=True)


    pipeline = Pipeline(experiment_tracker, train_loader, test_loader, training_parameters, criterion, optimizer)

    pipeline.train(network)

    pipeline.test_normal_accuracy(network)

    pipeline.test_robust_accuracy(network, epsilons=[0, 0.1, 0.2], attacks=[FGSMAttack(), CleverHansAttack()])