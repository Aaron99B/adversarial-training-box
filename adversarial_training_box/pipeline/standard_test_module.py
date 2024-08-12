import torch

from adversarial_training_box.pipeline.test_module import TestModule
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTestModule(TestModule):

    def __init__(self, attack: AdversarialAttack = None, epsilon: float = None) -> None:
        self.attack = attack
        self.epsilon = epsilon


    def test(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        correct = 0
        total = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = network(data)

            _, pred = output.data.max(1, keepdim=True)

            total += target.size(0)

            if not self.attack is None:
                correct_predictions = data[pred.eq(target.data.view_as(pred)).view_as(target)]
                labels_for_correct_predictions = target[pred.eq(target.data.view_as(pred)).view_as(target)]
                perturbed_data = self.attack.compute_perturbed_image(network=network, data=correct_predictions, labels=labels_for_correct_predictions, epsilon=self.epsilon)

                output = network(perturbed_data)
                _, adv_pred = output.data.max(1, keepdim=True)

                correct += adv_pred.eq(labels_for_correct_predictions.data.view_as(adv_pred)).sum().item()

            else:

                correct += pred.eq(target.data.view_as(pred)).sum().item()

        final_acc = correct / total

        return self.attack, self.epsilon, final_acc

    def __str__(self) -> str:
        return f"test_module_filtered_adv_acc{self.attack}_{self.epsilon}"
    