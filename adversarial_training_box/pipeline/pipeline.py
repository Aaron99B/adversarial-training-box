
import torch

from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.pipeline.training_module import TrainingModule
from adversarial_training_box.pipeline.test_module import TestModule
from adversarial_training_box.pipeline.early_stopper import EarlyStopper


class Pipeline:
    def __init__(self, experiment_tracker: ExperimentTracker, training_parameters: AttributeDict, criterion: torch.nn.Module, optimizer: torch.optim, scheduler: torch.optim=None) -> None:
        self.experiment_tracker = experiment_tracker
        self.training_parameters = training_parameters
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, network):
        self.experiment_tracker.save_model(network)

    def train(self, train_loader: torch.utils.data.DataLoader, network: torch.nn.Module, training_stack: list[int, TrainingModule], validation_module: TestModule = None, in_training_validation_loader: torch.utils.data.DataLoader = None, early_stopper: EarlyStopper = None):

        for epochs, module in training_stack:
            for epoch in range(0, epochs):
                network.train()
                train_accuracy = module.train(train_loader, network, self.optimizer, self.experiment_tracker)
                validation_accuracy = train_accuracy

                if not self.experiment_tracker is None:
                    self.experiment_tracker.log({"train_accuracy" : train_accuracy})

                if not self.scheduler is None:
                    self.scheduler.step()
                
                if validation_module:
                    network.eval()
                    _, _, validation_accuracy = validation_module.test(in_training_validation_loader, network)
                    network.zero_grad()
                    self.experiment_tracker.log({"training_validation_accuracy" : validation_accuracy})
                
                if early_stopper:
                    should_stop = early_stopper.early_stop(validation_accuracy)
                    if should_stop:
                        print(f"early stopped at epoch: {epoch}")
                        break

                self.save_model(network)


    def test(self, network: torch.nn.Module, test_loader: torch.utils.data.DataLoader, testing_stack: list[TestModule]):

        network.eval()
        for module in testing_stack:

            print(f'testing for attack: {module.attack} and epsilon: {module.epsilon}')

            attack, epsilon, accuracy = module.test(test_loader, network)
            self.experiment_tracker.log_test_result({"epsilon" : epsilon, "attack" : str(attack), "accuracy" : accuracy})

        self.experiment_tracker.log_table_result_table_online()


    def test_accuracy_class_wise(self, network: torch.nn.Module):
        classes = [0,1,2,3,4,5,6,7,8,9]
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in self.test_loader:
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

        self.experiment_tracker.save_class_accuracy_table(class_accuracies)