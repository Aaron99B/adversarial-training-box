import torch
import torchvision
import numpy as np
from torch import nn

from adversarial_training_box.adversarial_attack.adversarial_attack import AdversarialAttack

class RandomFGSMAttack(AdversarialAttack):

    def __init__(self, alpha: float) -> None:
        super().__init__("FGSM_random_init")
        self.alpha = alpha

    def compute_perturbed_image(self, network: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        delta = torch.zeros_like(data).uniform_(-epsilon, epsilon).cpu()
        delta.requires_grad = True
        output = network(data + delta)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-data, delta.data), 0-data)
        delta = delta.detach()

        return torch.clamp(data + delta, 0, 1)
        
    
    def __str__(self) -> str:
        return self.name