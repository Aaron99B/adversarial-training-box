from random import shuffle
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from pathlib import Path
import optuna
from optuna.trial import TrialState
import argparse
import importlib
from adversarial_training_box.pipeline.early_stopper import EarlyStopper

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.models.MNIST.mnist_relu_4_1024 import MNIST_RELU_4_1024
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule

def get_network_class(network_name):
    """Dynamically import and return network class based on name."""
    try:
        module = importlib.import_module(f"adversarial_training_box.models.MNIST.{network_name.lower()}")
        return getattr(module, network_name)
    except (ImportError, AttributeError):
        raise ValueError(f"Network {network_name} not found. Make sure it exists in the models directory.")


if __name__ == "__main__":
    # Experiment configuration
    parser = argparse.ArgumentParser(description='Training script with configurable network and experiment name')
    parser.add_argument('--network', type=str, default='MNIST_RELU_4_1024',
                       help='Network architecture name (default: MNIST_RELU_4_1024)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name (default: {network}-standard-training)')
    args = parser.parse_args()  # Add this line to actually parse the arguments

    training_parameters = AttributeDict(
        learning_rate = 0.001,
        weight_decay = 1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.98,
        attack_epsilon=0.3, 
        patience_epochs=6, 
        overhead_delta=0.0,
        batch_size=256)
    
    experiment_name = args.experiment_name
    NetworkClass = get_network_class(args.network)
    network = NetworkClass()

    # Training configuration
    optimizer = getattr(optim, 'Adam')(network.parameters(), lr=training_parameters.learning_rate, weight_decay=training_parameters.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters.scheduler_step_size, gamma=training_parameters.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=training_parameters.patience_epochs, delta=training_parameters.overhead_delta)

    # Train, validation and test dataset
    dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset,validation_dataset, = torch.utils.data.random_split(dataset, (0.8, 0.2))
    test_dataset = torchvision.datasets.MNIST('../data',train=False, download=True, transform=torchvision.transforms.ToTensor())

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Validation module
    validation_module = StandardTestModule(criterion=criterion)

    # Training modules stack
    training_stack = []
    training_stack.append((300, StandardTrainingModule(criterion=criterion)))

    # Testing modules stack
    testing_stack = [StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.1),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.2),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.3),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.1),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.2),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3),
    ]
    
    # Convert complex objects to JSON-serializable format
    def serialize_training_stack(stack):
        return [{"epochs": epochs, "module_type": type(module).__name__, 
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for epochs, module in stack]
    
    def serialize_testing_stack(stack):
        return [{"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for module in stack]
    
    def serialize_validation_module(module):
        return {"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)}

    training_objects = AttributeDict(criterion=str(criterion), 
                                     optimizer=str(optimizer), 
                                     network=str(network), 
                                     scheduler=str(scheduler), 
                                     training_stack=serialize_training_stack(training_stack),
                                     testing_stack=serialize_testing_stack(testing_stack),
                                     validation_module=serialize_validation_module(validation_module))

    # Setup experiment
    experiment_tracker = ExperimentTracker(experiment_name, Path("./generated"), login=True)
    experiment_tracker.initialize_new_experiment("", training_parameters=training_parameters | training_objects)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)

    # Train
    pipeline.train(train_loader, network, training_stack, early_stopper=early_stopper, 
                   validation_loader=validation_loader,
                   validation_module=validation_module
                   )

    # Test
    network = experiment_tracker.load_trained_model(network)
    pipeline.test(network, test_loader, testing_stack=testing_stack)
    experiment_tracker.export_to_onnx(network, test_loader)