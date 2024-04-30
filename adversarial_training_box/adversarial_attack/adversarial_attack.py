from abc import ABC, abstractmethod
import torch

class AdversarialAttack(ABC):
    @abstractmethod
    def compute_perturbed_image(network: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        pass