import torch

from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
class EarlyStopper:
    def __init__(self, validation_loader: torch.utils.data.DataLoader, validation_module: StandardTestModule, patience: int = 1, min_delta: float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.validation_loader = validation_loader
        self.validation_module = validation_module

    def early_stop(self, network):

        validation_loss = self.validation_module(self.validation_loader, network)
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
