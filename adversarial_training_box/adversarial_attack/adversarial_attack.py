from abc import ABC, abstractmethod
import torch

class AdversarialAttack(ABC):

    def __init__(self, name) -> None:
        self.name = name
        
    @abstractmethod
    def compute_perturbed_image(self, network: torch.nn.Module, data: torch.tensor, labels: torch.tensor, epsilon: float) -> torch.tensor:
        pass