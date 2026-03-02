import torch
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack


class PGDAttack(AdversarialAttack):
    def __init__(self, epsilon_step_size: float, number_iterations: int, random_init: bool) -> None:
        super().__init__("PGD_foolbox")
        self.epsilon_step_size = epsilon_step_size
        self.number_iterations = number_iterations
        self.random_init = random_init
        
    def compute_perturbed_image(self, network: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        network.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fmodel = PyTorchModel(network, bounds=(0,1), device=device)
        attack = LinfPGD(abs_stepsize=self.epsilon_step_size, steps=self.number_iterations, random_start=self.random_init)

        raw_advs, adversaries, success = attack(fmodel, data, labels, epsilons=epsilon)

        return adversaries.detach()
    
    def __str__(self) -> str:
        return f"{self.name}_{self.epsilon_step_size}_{self.number_iterations}_{self.random_init}"