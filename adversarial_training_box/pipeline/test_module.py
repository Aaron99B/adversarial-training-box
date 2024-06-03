from abc import ABC, abstractmethod
import torch

class TestModule(ABC):

    @abstractmethod
    def test(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module) -> None:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass