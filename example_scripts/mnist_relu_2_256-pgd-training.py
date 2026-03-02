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

from adversarial_training_box.models.MNIST.mnist_relu_2_256 import MNIST_RELU_2_256
from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule

def objective(trial):
    network = MNIST_RELU_2_256() 

    optimizer_name = "Adam"
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 1, 10, log=True)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.01, 1, log=True)
    attack_epsilon = 0.3
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    dataset = torchvision.datasets.MNIST('../data', train=True, download=False,
                    transform=torchvision.transforms.ToTensor())

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2))
    train_sampler = torch.utils.data.RandomSampler(data_source=train_dataset, num_samples=8000)
    validation_sampler = torch.utils.data.RandomSampler(data_source=validation_dataset, num_samples=2000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, sampler=validation_sampler)

    training_module = StandardTrainingModule(criterion=criterion, attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=attack_epsilon)

    for epoch in range(0,40):
        train_accuracy, robust_accuracy = training_module.train(train_loader, network, optimizer)
        scheduler.step()

        network.eval()
        test_module = StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3)
        attack, epsilon, test_accuracy, test_robust_accuracy, valid_loss = test_module.test(validation_loader, network)

        trial.report(test_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_accuracy

if __name__ == "__main__":
    # torch.manual_seed(0)

    # study = optuna.create_study(direction="maximize", storage="sqlite:///pgd_training.db")
    # study.optimize(objective, n_trials=300, timeout=6000)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))

    # training_parameters = AttributeDict(learning_rate=trial.params["lr"], 
    #                                     weight_decay=trial.params["weight_decay"], 
    #                                     scheduler_step_size=trial.params["scheduler_step_size"], 
    #                                     scheduler_gamma=trial.params["scheduler_gamma"],
    #                                     attack_epsilon=0.3,
    #                                     early_stopper_min_delta=0.002,
    #                                     patience_epochs=5, 
    #                                     batch_size=256)

    # Experiment configuration
    training_parameters = AttributeDict(
    learning_rate = 0.001,
    weight_decay = 1e-4,
    scheduler_step_size=10,
    scheduler_gamma=0.98,
    attack_epsilon=0.3,
    patience_epochs=5,
    overhead_delta=0.0,
    batch_size=256)
    
    experiment_name = "mnist_relu_2_256-PGD-training"
    network = MNIST_RELU_2_256()

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
    validation_module = StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3, criterion=criterion)

    # Training modules stack
    training_stack = []
    training_stack.append((200, StandardTrainingModule(criterion=criterion, attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3)))

    # Testing modules stack
    testing_stack = [
        StandardTestModule(),
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
    experiment_tracker.initialize_new_experiment("PGD_Training", training_parameters=training_parameters | training_objects)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)

    # Train
    pipeline.train(train_loader, network, training_stack, early_stopper=None, 
                   validation_loader=validation_loader,
                   validation_module=validation_module
                   )

    # Test
    network = experiment_tracker.load_trained_model(network)
    pipeline.test(network, test_loader, testing_stack=testing_stack)
    experiment_tracker.export_to_onnx(network, test_loader)