
import torch
from tqdm import tqdm
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.database.experiment_tracker import ExperimentTracker


class Pipeline:
    def __init__(self, experiment_tracker: ExperimentTracker, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, training_parameters: AttributeDict, criterion: torch.nn.Module, optimizer: torch.optim) -> None:
        self.experiment_tracker = experiment_tracker
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.training_parameters = training_parameters
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, network: torch.nn.Module):
    
        self.experiment_tracker.watch(network, self.criterion, log_option="all", log_frequency=10)

        network.train()
        for epoch in range(0,self.training_parameters.epochs):
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                output = network(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.experiment_tracker.log({"loss": loss.item(), "epoch" : epoch})


    def test_normal_accuracy(self, network: torch.nn.Module):
        network.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in self.test_loader:
                output = network(data)
                test_loss += self.criterion(output, label, size_average=False).item()
                _, predicted = output.data.max(1, keepdim=True)
                correct += predicted.eq(label.data.view_as(predicted)).sum()
                total += label.size(0)
                test_loss /= len(self.test_loader.dataset)
                self.experiment_tracker.log({"test_loss": test_loss})
                
        self.experiment_tracker.log({"test_accuracy": 100. * correct / total})

        self.experiment_tracker.save_model(network, data)


    def test_robust_accuracy(self, network: torch.nn.Module, epsilons: list[float], attacks: list[AdversarialAttack]):
        
        results = []

        for attack in attacks:

            for epsilon in epsilons:
                correct = 0
                total = 0

                for data, target in tqdm(self.test_loader):
                    data.requires_grad = True

                    output = network(data)

                    loss = self.criterion(output, target)

                    network.zero_grad()

                    loss.backward()

                    perturbed_data = attack.compute_perturbed_image(network=network, data=data, labels=target, epsilon=epsilon)

                    output = network(perturbed_data)

                    _, final_pred = output.data.max(1, keepdim=True)

                    correct += final_pred.eq(target.data.view_as(final_pred)).sum().item()
                    total += target.size(0)

                final_acc = 100 * correct / total

                results.append({"epsilon" : epsilon, "attack" : attack.name, "robust_accuracy" : final_acc})
        self.experiment_tracker.log_table("robust_accuracy", results)


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