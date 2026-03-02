import torch
import numpy as np

from adversarial_training_box.pipeline.test_module import TestModule
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTestModule(TestModule):

    def __init__(self, attack: AdversarialAttack = None, epsilon: float = None, criterion:torch.nn.Module = None) -> None:
        self.attack = attack
        self.epsilon = epsilon
        self.criterion = criterion

    def test(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module) -> None:
        network.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        correct_adversarial = 0
        correct_benign = 0
        total = 0
        valid_losses = []

        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            total += target.size(0)

            output = network(data)

            if self.criterion: 
                loss = self.criterion(output, target)
                valid_losses.append(loss.item())

            _, pred = output.data.max(1, keepdim=True)
            correct_benign += pred.eq(target.data.view_as(pred)).sum().item()

            # Calculate adversarial accuracies
            if not self.attack is None:
                # Filter out predictions that are correctly predicted in unperturbed case, as others will be incorrect anyways
                correct_predictions = data[pred.eq(target.data.view_as(pred)).view_as(target)]
                labels_for_correct_predictions = target[pred.eq(target.data.view_as(pred)).view_as(target)]
                perturbed_data = self.attack.compute_perturbed_image(network=network, data=correct_predictions, labels=labels_for_correct_predictions, epsilon=self.epsilon)

                output = network(perturbed_data)
                _, adv_pred = output.data.max(1, keepdim=True)

                correct_adversarial += adv_pred.eq(labels_for_correct_predictions.data.view_as(adv_pred)).sum().item()

        # Calculate accuracies
        robust_accuracy = None
        if not self.attack is None:
            robust_accuracy = correct_adversarial / total
        test_accuracy = correct_benign / total

        # Calculate loss
        valid_loss = None 
        if self.criterion:
            valid_loss = np.average(valid_losses)

        return self.attack, self.epsilon, test_accuracy, robust_accuracy, valid_loss

    def __str__(self) -> str:
        return f"test_module_filtered_adv_acc{self.attack}_{self.epsilon}"
    