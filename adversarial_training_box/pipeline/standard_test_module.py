import torch

from adversarial_training_box.pipeline.test_module import TestModule
from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class StandardTestModule(TestModule):

    def __init__(self, attack: AdversarialAttack = None, epsilon: float = None) -> None:
        self.attack = attack
        self.epsilon = epsilon


    def test(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module) -> None:

        correct = 0
        total = 0
        for data, target in data_loader:

            if not self.attack is None:
                data = self.attack.compute_perturbed_image(network=network, data=data, labels=target, epsilon=self.epsilon)

            output = network(data)

            _, final_pred = output.data.max(1, keepdim=True)

            correct += final_pred.eq(target.data.view_as(final_pred)).sum().item()
            total += target.size(0)

            final_acc = 100 * correct / total

        return self.attack, self.epsilon, final_acc

    def __str__(self) -> str:
        return f"test_module_{self.attack}_{self.epsilon}"
    