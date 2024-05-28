from torch.nn.modules.module import Module
import torch
import numpy as np
from autoattack import AutoAttack


from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class AutoAttackModule(AdversarialAttack):
    def __init__(self) -> None:
        super().__init__("AutoAttack")
        
    def compute_perturbed_image(self, network: Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network.to(device)
        data.to(device)
        adversary = AutoAttack(network, norm="Linf", eps=epsilon, version="standard", device=device)
        x = adversary.run_standard_evaluation(data, labels)
        return x