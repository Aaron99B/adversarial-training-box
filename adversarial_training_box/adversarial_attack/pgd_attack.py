import numpy as np
import torch
from torch.nn.modules.module import Module
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack


class PGDAttack(AdversarialAttack):

    def __init__(self, epsilon_step_size: float, number_iterations: int, random_init: bool) -> None:
        super().__init__("PGD")
        self.epsilon_step_size = epsilon_step_size
        self.number_iterations = number_iterations
        self.random_init = random_init
        
    def compute_perturbed_image(self, network: Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        x = projected_gradient_descent(model_fn=network, x=data, eps=epsilon, eps_iter=self.epsilon_step_size, nb_iter=self.number_iterations, norm=np.inf, rand_init=self.random_init)

        return x
    
    def __str__(self) -> str:
        return f"{self.name}_{self.epsilon_step_size}_{self.number_iterations}_{self.random_init}"