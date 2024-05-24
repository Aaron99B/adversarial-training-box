import numpy as np
import torch
from torch.nn.modules.module import Module
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack


class CleverHansAttack(AdversarialAttack):

    def __init__(self) -> None:
        super().__init__("PGD")
        
    def compute_perturbed_image(self, network: Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        x = projected_gradient_descent(model_fn=network, x=data, eps=epsilon, eps_iter=0.01, nb_iter=30, norm=np.inf)

        return x