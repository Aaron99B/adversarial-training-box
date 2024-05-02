import torch
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack


class FoolboxAttack(AdversarialAttack):
    def __init__(self) -> None:
        super().__init__("foolbox")
        
    def compute_perturbed_image(self, model: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        fmodel = PyTorchModel(model, bounds=(0, 1), device="cpu")
        attack = LinfPGD()

        raw_advs, adversaries, success = attack(fmodel, data, labels, epsilons=epsilon)

        return adversaries