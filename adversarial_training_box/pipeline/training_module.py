from abc import ABC, abstractmethod
import torch

from adversarial_training_box.database.experiment_tracker import ExperimentTracker

class TrainingModule(ABC):

    @abstractmethod
    def train(self, data_loader: torch.utils.data.DataLoader, network: torch.nn.Module, optimizer: torch.optim, scheduler: torch.optim=None, experiment_tracker: ExperimentTracker = None) -> None:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass